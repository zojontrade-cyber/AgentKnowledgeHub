"""
文档解析 Agent — 多模态文档解析，支持 PDF / 图片 / 表格 / 纯文本

核心能力:
  1. PDF 解析（文字 + 嵌入图片 + 表格）
  2. 图片 OCR + LLM 视觉理解
  3. 表格结构化提取
  4. 文档分块（Chunking）与元数据标注
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

from config import settings
from services.llm_factory import LazyLLM


class DocType(str, Enum):
    PDF = "pdf"
    IMAGE = "image"
    TABLE = "table"
    TEXT = "text"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


@dataclass
class DocumentChunk:
    """
    文档块

    三层内容分离（这是本次重构的核心设计）
    --------------------------------------
    早期实现把「标题路径 + 正文」拼成一个字符串，既用于 embedding
    又用于生成，导致两个问题：

      1. **embedding 被污染**：`【员工考勤管理制度 > 1. 文档说明】\n# 员工考勤
         管理制度\n\n## 1. 文档说明\n\n本制度规定…` 这样的文本里，真正
         有区分度的只有一句"标准工作时间为…"，其余是通用套话。
         结果含答案的块相似度反而低于无关块（实测 0.42 vs 0.57），
         出现「文档里明明有却搜不出来」。

      2. **分块目标错位**：优化目标应是「一个 chunk 表达一个检索意图」，
         而非「chunk 大小合适」。把标题、文档说明、工作时间混在一块，
         这一块就不再是完整的 answer unit。

    因此拆成三层：

      content          —— **只用于 embedding**：章节标题 + 本章节正文，
                          不含上级标题、不含其他章节
      parent_content   —— **只用于生成**：所属文档的完整内容（或相邻章节），
                          检索命中后交给 LLM 作为上下文
      display          —— 用于引用展示：标题路径 + 内容
    """

    content: str                      # 检索用（进入 embedding）
    doc_id: str
    chunk_index: int
    doc_type: DocType
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}#chunk-{self.chunk_index}"

    @property
    def section_title(self) -> str:
        """本块所属章节的标题（如「2. 工作时间」）"""
        return self.metadata.get("section_title") or ""

    @property
    def title_path(self) -> str:
        """完整标题路径（如「员工考勤管理制度 > 2. 工作时间」），仅用于展示"""
        return self.metadata.get("heading_path") or self.metadata.get("title") or ""

    @property
    def parent_content(self) -> str:
        """
        父块内容 —— 小-大分块（small-to-big）的「大块」，**仅用于生成**。

        检索用小而聚焦的 content 精确定位，生成时把 parent_content
        一起交给 LLM，兼顾精度与上下文完整性。
        """
        return self.metadata.get("parent_content") or self.content

    @property
    def parent_id(self) -> str:
        """父块标识（同一文档的块共享）"""
        return self.metadata.get("parent_id") or self.chunk_id

    @property
    def display(self) -> str:
        """用于引用展示的文本：标题路径 + 内容"""
        if self.title_path and self.title_path not in self.content:
            return f"【{self.title_path}】{self.content}"
        return self.content


class DocParserAgent:
    """
    文档解析 Agent

    工作流:
      classify → parse → chunk → enrich_metadata → output
    """

    SUPPORTED_EXTENSIONS: dict[str, DocType] = {
        ".pdf": DocType.PDF,
        ".png": DocType.IMAGE,
        ".jpg": DocType.IMAGE,
        ".jpeg": DocType.IMAGE,
        ".csv": DocType.TABLE,
        ".xlsx": DocType.TABLE,
        ".xls": DocType.TABLE,
        ".txt": DocType.TEXT,
        ".md": DocType.MARKDOWN,
    }

    # 分块参数
    # ────────────────────────────────────────────────────────
    # CHUNK_SIZE 现在是**上限**而非固定长度：
    # 旧实现按固定 512 字硬切，导致 880 字的文档被切成
    # 「512 + 368」两块，第二块从句子中间开始（如「次按考勤制度处理」），
    # 既污染向量语义，也让引用内容残缺。
    # 新实现优先在语义边界（章节 → 段落 → 句子）切分，只在必要时
    # 才按字符截断，因此不会出现断头文本。
    CHUNK_SIZE = 512          # 单块上限（软约束）
    CHUNK_OVERLAP = 64        # 仅在被迫按字符切分时使用
    MIN_CHUNK_SIZE = 80       # 小于此长度的块会尝试与相邻块合并

    def __init__(self) -> None:
        # 惰性构造：不在 __init__ 里创建 ChatOpenAI，否则未配置 Key 时
        # 连导入/启动都会失败（详见 services/llm_factory.py）
        self.llm = LazyLLM(temperature=0)

    # ── public API ───────────────────────────────────────────

    async def parse(self, file_path: str) -> list[DocumentChunk]:
        """解析单个文件，返回文档块列表"""
        doc_type = self._classify(file_path)
        doc_id = self._make_doc_id(file_path)

        raw_texts: list[str] = []
        if doc_type == DocType.PDF:
            raw_texts = await self._parse_pdf(file_path)
        elif doc_type == DocType.IMAGE:
            raw_texts = await self._parse_image(file_path)
        elif doc_type == DocType.TABLE:
            raw_texts = await self._parse_table(file_path)
        elif doc_type in (DocType.TEXT, DocType.MARKDOWN):
            raw_texts = self._parse_text(file_path)
        else:
            raw_texts = self._parse_text(file_path)

        chunks = self._chunk_texts(raw_texts, doc_id, doc_type, file_path)
        return chunks

    async def parse_batch(self, file_paths: list[str]) -> list[DocumentChunk]:
        """批量解析多个文件"""
        all_chunks: list[DocumentChunk] = []
        for fp in file_paths:
            all_chunks.extend(await self.parse(fp))
        return all_chunks

    # ── classification ───────────────────────────────────────

    def _classify(self, file_path: str) -> DocType:
        ext = os.path.splitext(file_path)[1].lower()
        return self.SUPPORTED_EXTENSIONS.get(ext, DocType.UNKNOWN)

    @staticmethod
    def _make_doc_id(file_path: str) -> str:
        return hashlib.sha256(file_path.encode()).hexdigest()[:16]

    # ── PDF parsing ──────────────────────────────────────────

    async def _parse_pdf(self, file_path: str) -> list[str]:
        """
        PDF 多模态解析:
          1. 提取文字页面
          2. 如果页面包含图片 / 表格，调用 LLM 视觉理解
        """
        texts: list[str] = []
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    texts.append(page_text.strip())
        except Exception:
            texts.append(f"[PDF 解析失败] {file_path}")

        if not texts:
            texts = await self._pdf_vision_fallback(file_path)

        return texts

    async def _pdf_vision_fallback(self, file_path: str) -> list[str]:
        """当 PDF 纯文本提取失败时，使用 LLM 视觉能力"""
        try:
            from pdf2image import convert_from_path

            images = convert_from_path(file_path, dpi=150, first_page=1, last_page=5)
            texts: list[str] = []
            for img in images:
                description = await self._describe_image_with_llm(img)
                texts.append(description)
            return texts
        except Exception:
            return [f"[PDF 视觉解析失败] {file_path}"]

    # ── image parsing ────────────────────────────────────────

    async def _parse_image(self, file_path: str) -> list[str]:
        """图片解析: OCR + LLM 视觉理解"""
        texts: list[str] = []
        ocr_text = self._ocr(file_path)
        if ocr_text.strip():
            texts.append(ocr_text)

        from PIL import Image
        img = Image.open(file_path)
        description = await self._describe_image_with_llm(img)
        texts.append(description)
        return texts

    @staticmethod
    def _ocr(file_path: str) -> str:
        try:
            import pytesseract
            from PIL import Image
            return pytesseract.image_to_string(Image.open(file_path), lang="chi_sim+eng")
        except Exception:
            return ""

    async def _describe_image_with_llm(self, image: Any) -> str:
        """调用 LLM 多模态能力描述图片内容"""
        import base64
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        messages = [
            SystemMessage(content="你是一个专业的文档分析助手，请详细描述图片中的内容，包括文字、表格、图表信息。"),
            HumanMessage(content=[
                {"type": "text", "text": "请描述这张图片的所有内容："},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]),
        ]
        resp = await self.llm.ainvoke(messages)
        return resp.content

    # ── table parsing ────────────────────────────────────────

    async def _parse_table(self, file_path: str) -> list[str]:
        """表格解析: CSV / Excel → 结构化文本"""
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".csv":
                return self._parse_csv(file_path)
            else:
                return self._parse_excel(file_path)
        except Exception:
            return [f"[表格解析失败] {file_path}"]

    @staticmethod
    def _parse_csv(file_path: str) -> list[str]:
        import csv
        texts: list[str] = []
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            rows: list[str] = []
            for row in reader:
                rows.append(" | ".join(f"{h}: {row.get(h, '')}" for h in headers))
            for i in range(0, len(rows), 20):
                batch = rows[i : i + 20]
                texts.append(f"表头: {' | '.join(headers)}\n" + "\n".join(batch))
        return texts or ["[空 CSV]"]

    @staticmethod
    def _parse_excel(file_path: str) -> list[str]:
        try:
            import openpyxl
            wb = openpyxl.load_workbook(file_path, read_only=True)
            texts: list[str] = []
            for sheet in wb.worksheets:
                rows = list(sheet.iter_rows(values_only=True))
                if not rows:
                    continue
                headers = [str(c) if c else "" for c in rows[0]]
                data_rows: list[str] = []
                for row in rows[1:]:
                    data_rows.append(" | ".join(
                        f"{headers[j]}: {row[j]}" if j < len(headers) else str(row[j])
                        for j in range(len(row))
                    ))
                for i in range(0, len(data_rows), 20):
                    batch = data_rows[i : i + 20]
                    texts.append(f"工作表: {sheet.title}\n表头: {' | '.join(headers)}\n" + "\n".join(batch))
            return texts or ["[空 Excel]"]
        except Exception:
            return [f"[Excel 解析失败] {file_path}"]

    # ── text / markdown ──────────────────────────────────────

    # ── 元数据清洗 ───────────────────────────────────────────

    # 企业文档常见的两类元数据（实测于「云启科技」制度文档集）：
    #   1. HTML 注释块：<!-- 文档编号: YQ-INST-021 类别: 制度类 ... -->
    #   2. Markdown 引用行：> 文档编号：YQ-INST-021 / > 版本：v1.4
    #
    # 为什么必须剥离：
    #   分块是「从第 0 字符起按 512 字硬切」的，元数据位于文首，
    #   因此**第一个分块会被元数据主导**。而所有文档的元数据措辞
    #   几乎一致（文档编号/类别/来源/获取方式...），导致不同主题的
    #   文档产生高度相似的向量，检索时互相抢占，用户提问命中的
    #   往往是「文档编号」这类样板文字而非实际业务内容。
    _META_REF_KEYS = (
        "文档编号", "版本", "生效日期", "责任人", "权限级别",
        "类别", "来源", "获取方式", "更新日期", "负责人", "状态", "备注",
    )

    @classmethod
    def strip_metadata(cls, text: str) -> str:
        """剥离文档头尾的模板化元数据，保留标题与正文"""
        import re

        # 1. HTML 注释块（含跨行）
        text = re.sub(r"<!--.*?-->", "", text, flags=re.S)

        # 2. 文首连续的 Markdown 引用元数据行
        keys = "|".join(re.escape(k) for k in cls._META_REF_KEYS)
        text = re.sub(
            rf"(?m)^\s*>\s*({keys})\s*[:：].*$\n?", "", text
        )

        # 3. 尾部「附则」段（各文档措辞雷同，同样会造成向量趋同）
        text = re.sub(r"(?ms)^\s*#{1,3}\s*附则\s*$.*\Z", "", text)

        # 4. 模板声明句
        text = re.sub(r"(?m)^.*本(制度|办法|规范|规定|细则|预案)为模拟文档.*$\n?", "", text)

        # 压缩多余空行
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @classmethod
    def extract_title(cls, text: str) -> str:
        """提取文档标题：优先一级标题，其次首个非空行"""
        import re

        m = re.search(r"^#\s+(.+)$", text, flags=re.M)
        if m:
            return m.group(1).strip()
        for line in text.split("\n"):
            s = line.strip().lstrip("#").strip()
            if s:
                return s
        return ""

    @staticmethod
    def _parse_text(file_path: str) -> list[str]:
        with open(file_path, encoding="utf-8") as f:
            return [f.read()]

    # ── 语义切分 ─────────────────────────────────────────────

    # 按优先级排列的分隔符：越靠前越优先在此处切分。
    # Markdown 标题优先，其次是段落/换行，再次是中文句读。
    _SEPARATORS = [
        "\n## ",      # 二级标题（章节）
        "\n### ",     # 三级标题（小节）
        "\n\n",       # 段落
        "\n",         # 行
        "。", "！", "？", "；",   # 中文句末
        ". ", "! ", "? ", "; ",   # 英文句末
        "，", ",",    # 从句
        " ",          # 词
    ]

    @classmethod
    def _split_recursive(cls, text: str, max_size: int) -> list[str]:
        """
        递归按语义边界切分文本。

        策略（对应方案 1 + 2）：
          1. 文本不超限 -> 直接返回（**不强行凑长度**，短章节自成一块）
          2. 依次尝试各层级分隔符，在该层级能切出多段时**立即采用**
             （所以 Markdown 标题会优先命中，章节天然独立成块）
          3. 切出的片段若仍超限，对片段递归
          4. 所有分隔符都无效时，才退化为按字符硬切

        关键设计：**不做跨片段合并**。
        早期版本在此处把短片段累加合并到接近 max_size，结果 6 个章节
        被拼回 2 块（章节 1+2+3 累加 439 字 < 512 就继续拼），
        语义单元反而被破坏了。合并只在 _merge_fragments 里针对
        真正的碎片处理。
        """
        text = text.strip()
        if not text:
            return []

        # 结构化文档（含 Markdown 标题）始终按标题切分，**无论长度**。
        # 否则一篇 476 字、含 5 个章节的文档会因「未超限」被整篇返回，
        # 多个主题挤在一块的向量里，检索精度下降。
        import re

        if re.search(r"(?m)^#{1,3}\s+\S", text):
            # 在标题前插入分隔符后切分，标题归属其后的内容
            marked = re.sub(r"(?m)^(#{1,3}\s+)", "\n\x00\\1", text)
            blocks = [b for b in marked.split("\x00") if b.strip()]
            if len(blocks) > 1:
                out: list[str] = []
                for b in blocks:
                    b = b.strip()
                    if not b:
                        continue
                    # 单个章节仍超限时，对其内部继续递归
                    if len(b) > max_size:
                        out.extend(cls._split_recursive(b, max_size))
                    else:
                        out.append(b)
                return out

        if len(text) <= max_size:
            return [text]

        for sep in cls._SEPARATORS:
            if sep not in text:
                continue
            parts = text.split(sep)
            if len(parts) < 2:
                continue

            # 把分隔符还原到片段末尾，保持原文可读
            pieces = [
                (p + sep) if i < len(parts) - 1 else p
                for i, p in enumerate(parts)
            ]
            pieces = [p for p in pieces if p.strip()]
            if len(pieces) < 2:
                continue

            # 对仍然超限的片段递归
            out: list[str] = []
            for p in pieces:
                if len(p) > max_size:
                    out.extend(cls._split_recursive(p, max_size))
                else:
                    out.append(p.strip())
            return out

        # 兜底：按字符硬切（保留重叠，维持上下文连贯）
        out = []
        start = 0
        while start < len(text):
            end = start + max_size
            piece = text[start:end].strip()
            if piece:
                out.append(piece)
            if end >= len(text):
                break
            start = end - cls.CHUNK_OVERLAP
        return out

    @staticmethod
    def _merge_fragments(chunks: list[str], min_size: int, max_size: int) -> list[str]:
        """
        合并过短的碎片 —— **但章节块永不参与合并**。

        这里修的是一个真实踩过的坑：
          旧实现只在「新片段以标题开头」时 flush，但没考虑
          「当前 buf 是一个过短的标题块」的情况。于是：

              buf = "# 员工考勤管理制度"        (10 字, 过短)
              + "## 1. 文档说明"               (66 字) -> buf=76, 仍过短
              + "## 2. 工作时间"               (257 字) -> buf=337, 停止

          结果「工作时间」这一节被并进了「标题+文档说明」的大杂烩块，
          它的向量被通用套话（"本制度规定…适用于…"）主导，
          导致用户问"标准工作时间"时**反而搜不到这一块**
          （实测相似度 0.42，低于无关文档的 0.57）。

        新规则（按你的建议）：
          - 以 Markdown 标题开头的片段 = 章节，**永远独立成块**
          - 只有「不以标题开头」的碎片（真正的续行/残句）才考虑并入前块
          - 章节块若自身过短，也保留 —— 语义完整优于长度达标
        """
        import re

        if not chunks:
            return []

        is_heading = lambda s: bool(re.match(r"^\s*#{1,3}\s+\S", s))

        out: list[str] = []
        buf = ""

        for c in chunks:
            c = c.strip()
            if not c:
                continue

            if is_heading(c):
                # 章节块：先落盘累积的 buf，再让本章节独立成为新 buf
                if buf:
                    out.append(buf)
                buf = c
                continue

            # 非章节内容（续行/残句）：
            # 若 buf 是过短的章节块，则并入它（补全该章节）
            if buf and len(buf) < min_size and len(buf) + len(c) + 2 <= max_size:
                buf = buf + "\n\n" + c
            else:
                if buf:
                    out.append(buf)
                buf = c

        if buf:
            out.append(buf)

        return [c for c in out if c.strip()]

    @staticmethod
    def _heading_path(text: str, upto: int) -> str:
        """
        计算位置 upto 处所属的标题路径，如「员工考勤管理制度 > 请假流程」。

        用于给每个块标注它在文档结构中的位置，既提升向量检索的
        区分度，也让引用来源对用户更可读。
        """
        import re

        segments: list[str] = []
        for m in re.finditer(r"(?m)^(#{1,3})\s+(.+)$", text):
            if m.start() >= upto:
                break
            level = len(m.group(1))
            title = m.group(2).strip()
            # 截断到对应层级
            segments = segments[: level - 1]
            segments.append(title)
        return " > ".join(segments)

    @staticmethod
    def _sections_with_spans(text: str) -> list[tuple[int, int, str]]:
        """
        切出各章节的 (起始位置, 结束位置, 章节全文)。

        用于小-大分块：小块检索命中后，可据此回溯到它所属的完整章节。
        章节以 Markdown 标题（#, ##, ###）为边界；无标题时整篇视为一节。
        """
        marks = list(re.finditer(r"(?m)^#{1,3}\s+.+$", text))
        if not marks:
            return [(0, len(text), text.strip())]

        spans: list[tuple[int, int, str]] = []
        # 标题之前若有前言，单独成节
        if marks[0].start() > 0 and text[: marks[0].start()].strip():
            spans.append((0, marks[0].start(), text[: marks[0].start()].strip()))

        for i, m in enumerate(marks):
            start = m.start()
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            body = text[start:end].strip()
            if body:
                spans.append((start, end, body))
        return spans

    # ── 树形解析 ─────────────────────────────────────────────

    @dataclass
    class Section:
        """文档中的一个章节（Markdown 天然是树结构）"""
        level: int
        title: str
        body: str            # 该章节**直接**内容（不含子章节）
        start: int           # 在原文中的起始偏移
        end: int

    @classmethod
    def parse_tree(cls, text: str) -> tuple[str, list["DocParserAgent.Section"]]:
        """
        把 Markdown 直接解析成 (文档标题, 章节列表) —— **不再先切字符串再补救**。

        为什么改成树形
        --------------
        旧流程是「先按分隔符切成字符串数组，再靠正则把标题还原回去」，
        结果产生了 33 个纯标题块（如 `"员工考勤管理制度"` 8 个字），
        它们向量化后互相高度相似且含最通用的词（"制度""办法"），
        在检索时集体占据前排，把真正含答案的章节块挤到第 8 位。

        根因是 split 与 merge 两个阶段职责混乱：split 产生标题行，
        merge 又要判断"标题不能吞掉后续章节"，两处规则互相打架。

        Markdown 本身就是树（标题 = 节点），直接解析出结构即可，
        结构明确后：
          - 文档标题 -> 作为每个 chunk 的上下文前缀
          - 章节标题 + 章节正文 -> chunk 的 embedding 内容
          - 纯标题行不再单独成为 chunk
        """
        marks = list(re.finditer(r"(?m)^(#{1,6})\s+(.+)$", text))

        # 文档标题 = 第一个一级标题（没有则取首个非空行）
        doc_title = ""
        for m in marks:
            if len(m.group(1)) == 1:
                doc_title = m.group(2).strip()
                break
        if not doc_title:
            doc_title = cls.extract_title(text)

        if not marks:
            return doc_title, [
                cls.Section(1, doc_title, text.strip(), 0, len(text))
            ] if text.strip() else []

        sections: list[DocParserAgent.Section] = []

        # 标题之前的前言（若有实质内容）
        if marks[0].start() > 0:
            pre = text[: marks[0].start()].strip()
            if pre:
                sections.append(cls.Section(1, doc_title or "前言", pre, 0, marks[0].start()))

        for i, m in enumerate(marks):
            level = len(m.group(1))
            title = m.group(2).strip()
            start = m.start()
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            body = text[m.end():end].strip()

            # 文档标题（一级）本身不作为独立章节，其 body 归入后续
            if level == 1 and not body:
                continue

            sections.append(cls.Section(level, title, body, start, end))

        return doc_title, sections

    # ── 分块质量闸门 ─────────────────────────────────────────

    # 入库前的最小内容长度。低于此值且**没有实质正文**的块会被拒绝。
    MIN_INDEXABLE_CHARS = 30

    # ── 「文档说明」类低价值块 ───────────────────────────────
    #
    # 问题不是 embedding 污染，而是**低价值高频模板词**：
    #   「本办法规范…」「为了加强…」「制定本制度…」「适用于…」
    # 这类块几乎不含判别性信息，却含最通用的词（制度/管理/流程/规范），
    # 于是**稠密和 BM25 都会被它占满 top-k**。
    #
    # 实测（159 章节）："请假要走什么流程" 的 top1 曾是
    # 「离职交接管理办法 / 1. 文档说明」——"请假"根本不在该块中出现。
    #
    # 处理方式：**不删除**，标记 searchable=False 使其退出第一阶段召回，
    # 但 parent_content 保留 → 回答阶段仍可通过 small-to-big 拿到
    # 制度名称 / 适用范围 / 制定目的（这些确实来自文档说明）。
    _OVERVIEW_TITLE_PATTERNS = (
        r"^\s*\d*[\.、]?\s*文档说明\s*$",
        r"^\s*\d*[\.、]?\s*制度说明\s*$",
        r"^\s*\d*[\.、]?\s*概述\s*$",
        r"^\s*\d*[\.、]?\s*总则\s*$",
        r"^\s*\d*[\.、]?\s*引言\s*$",
        r"^\s*\d*[\.、]?\s*前言\s*$",
        r"^\s*\d*[\.、]?\s*说明\s*$",
        r"^\s*\d*[\.、]?\s*目的\s*$",
        r"^\s*\d*[\.、]?\s*编制目的\s*$",
    )
    # 注意：**「适用范围」不在此列**。
    #   「适用于与云启科技建立劳动关系的正式员工，试用期满后开始计算…」
    #   这类章节含**实质规则**（试用期、实习生、劳务派遣的差异处理），
    #   排除它会丢失真实答案。实测 159 章节中有 2 个这样的块，
    #   若按标题一刀切会被误删 —— 已从模式中去掉。

    # 模板套话信号词 —— 用于兜底识别**标题不规范**的文档说明块
    _BOILERPLATE_MARKERS = (
        "本办法规范", "本规定规范", "本制度规范", "本规范明确",
        "本制度规定", "本办法明确", "本规定明确", "本制度建立",
        "本管理办法明确", "本规定建立", "本办法建立",
    )

    @classmethod
    def classify_section(cls, section_title: str, body: str) -> str:
        """
        章节类型分类：``content`` | ``doc_overview``

        doc_overview 判定（满足其一）：
          1. 标题是「N. 文档说明 / 概述 / 总则 / 前言…」这类元信息标题
          2. 正文开头是「本办法规范…」这类模板套话 **且** 篇幅很短
             （长正文即使开头是套话，通常仍含实质内容，不应整块排除）

        返回 ``doc_overview`` 的块会被标记 searchable=False，
        退出第一阶段召回，但保留在 parent_content 中可展开。
        """
        title = (section_title or "").strip()
        for pat in cls._OVERVIEW_TITLE_PATTERNS:
            if re.match(pat, title):
                return "doc_overview"

        # 兜底：标题不含元信息，但正文是典型套话 **且极短**。
        #
        # 阈值定在 120 字是有意的保守取值：
        #   实测「文档说明」正文 40-48 字，而「适用范围」56 字、
        #   任何含真实规则的章节都远超 120 字。
        # 宁可漏判（少排除一个块），不可误判（丢失真实答案）。
        body_s = (body or "").strip()
        if len(body_s) <= 120 and any(
            body_s.startswith(m) for m in cls._BOILERPLATE_MARKERS
        ):
            return "doc_overview"

        return "content"

    @classmethod
    def is_indexable(cls, content: str) -> bool:
        """
        质量闸门：判断一个块是否值得进入向量库。

        拒绝的对象：
          - 纯标题块（如 "员工考勤管理制度"、"2. 工作时间"）
            这类块不含可回答问题的实质内容，却因为含最通用的词
            （"制度""办法""管理"）而在检索时抢占前排 —— 实测它们
            是「文档里有答案却搜不出来」的直接原因。

        放行的对象：
          - 任何含实质正文的块
        """
        if not content:
            return False
        # 去掉 markdown 标题行与纯标题性质的短行，看剩余正文长度
        body = re.sub(r"(?m)^\s*#{1,6}\s+.+$", "", content)
        # 去掉「章节：xxx」这类前缀行
        body = re.sub(r"(?m)^\s*章节[：:].+$", "", body)
        body = body.strip()
        return len(body) >= cls.MIN_INDEXABLE_CHARS

    # ── chunking ─────────────────────────────────────────────

    def _chunk_texts(
        self,
        texts: list[str],
        doc_id: str,
        doc_type: DocType,
        source: str,
    ) -> list[DocumentChunk]:
        """
        基于树形结构的切分。

        embedding 文本结构（按设计）：
            文档标题
            章节：<章节标题>
            <章节正文>

        即 document_title + section_title + content 三段式。
        不包含：文档说明/通用套话、全文 parent。
        """
        chunks: list[DocumentChunk] = []
        idx = 0

        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(
                f"CHUNK_OVERLAP({self.CHUNK_OVERLAP}) 必须小于 "
                f"CHUNK_SIZE({self.CHUNK_SIZE})，否则分块无法终止"
            )

        for text in texts:
            text = self.strip_metadata(text)
            if not text:
                continue

            doc_title, sections = self.parse_tree(text)
            if not sections:
                continue

            # 父块 = 整篇文档（用于生成，不进 embedding）
            parent_id = f"{doc_id}#doc"

            for sec in sections:
                # 章节正文超长时，在章节内部继续按语义切
                pieces = self._split_recursive(sec.body, self.CHUNK_SIZE) if sec.body else []
                pieces = [p for p in pieces if p.strip()]

                if not pieces:
                    # 章节没有正文（纯标题）-> 跳过，不产生块
                    continue

                for piece in pieces:
                    # ── 章节类型 ──────────────────────────
                    # doc_overview 的块不进第一阶段召回（searchable=False），
                    # 但保留在 parent_content 中供回答阶段展开。
                    sec_type = self.classify_section(sec.title, sec.body)
                    searchable = sec_type != "doc_overview"

                    # ── embedding 三段式 ──────────────────
                    #
                    # 为什么要带文档标题：
                    #   只给"工作时间为9:00至18:00"，模型不知道这是
                    #   哪家公司的什么制度；加上文档标题后可显著提升
                    #   制度类文档的区分度（实测 section_title 缺失时，
                    #   含答案块的相似度低于无关块）。
                    #
                    # 为什么不用【路径】或 markdown 标记：
                    #   实测 `【员工考勤管理制度 > 1. 文档说明】\n# 员工考勤管理制度
                    #   \n\n## 1. 文档说明\n\n本制度规定…` 会把
                    #   通用套话权重放大，稀释真正的区分信息。
                    lines = []
                    if doc_title:
                        lines.append(doc_title)
                    if sec.title and sec.title != doc_title:
                        lines.append(f"章节：{sec.title}")
                    lines.append(piece.strip())
                    content = "\n".join(lines)

                    # ── 质量闸门 ─────────────────────────
                    if not self.is_indexable(piece):
                        logger.debug(
                            "跳过低质量块（无实质正文）",
                            extra={"extra_fields": {
                                "doc": doc_title, "section": sec.title,
                                "len": len(piece),
                            }},
                        )
                        continue

                    chunks.append(DocumentChunk(
                        content=content,
                        doc_id=doc_id,
                        chunk_index=idx,
                        doc_type=doc_type,
                        metadata={
                            "source": source,
                            "title": doc_title,
                            "heading_path": (
                                f"{doc_title} > {sec.title}"
                                if sec.title and sec.title != doc_title
                                else doc_title
                            ),
                            "section_title": sec.title,
                            "char_start": sec.start,
                            "char_end": sec.end,
                            "parent_id": parent_id,
                            "parent_content": text,
                            # 检索过滤依据：Chroma metadata 只接受标量，
                            # 布尔统一转成 0/1，避免 True 被存储层拒绝
                            "searchable": 1 if searchable else 0,
                            "section_type": sec_type,
                        },
                    ))
                    idx += 1

        return chunks
