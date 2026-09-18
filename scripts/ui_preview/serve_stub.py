"""索引台前端预览用桩服务（仅开发，不参与生产部署）。

为什么需要它
------------
UI 的视觉与交互验收不应依赖后端与向量库全部就绪。
本脚本用标准库起一个静态服务，并按**真实响应 schema** 造桩数据：

  GET  /api/health                 -> HealthResponse
  GET  /api/ui/overview            -> api/ui_routes.py::overview
  GET  /api/ui/documents           -> api/ui_routes.py::list_documents
    POST /api/qa/ask                 -> QuestionResponse（含四种 source_type）
  POST /api/ingest/upload          -> JobAcceptedResponse（202）
  GET  /api/jobs/{doc_id}          -> JobStatusResponse（状态机推进）

字段名一律照抄源码，避免"桩数据好看但前端接错字段"这类假阳性。

用法：
    python scripts/ui_preview/serve_stub.py --port 8099
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
STATIC = ROOT / "python" / "api" / "static"

# ── 桩数据 ───────────────────────────────────────────────────────

OVERVIEW = {
    "vector_store": {"backend": "chroma", "mode": "persistent", "total_vectors": 128},
    "uploaded_files": 3,
    "config": {
        "chat_model": "deepseek-chat",
        "embedding_model": "BAAI/bge-m3",
        "chroma_mode": "persistent",
        "environment": "dev",
    },
}

HEALTH = {
    "status": "ok",
    "service": "AgentKnowledgeHub",
    "dependencies": {"vector_store": "ok", "reranker": "disabled"},
}

NOW = time.time()
DOCUMENTS = {
    "total": 3,
    "documents": [
        {
            "display_name": "2024年度薪酬与加班费管理规定.pdf",
            "name": "2024年度薪酬与加班费管理规定.pdf",
            "source_path": "/uploads/9f2c1ab7c0e14d3e8a55.pdf",
            "status": "COMMITTED", "chunks_count": 86, "acl_scope": "public",
            "size": 4194304, "size_human": "4.0 MB", "modified": NOW - 5400,
        },
        {
            "display_name": "离职交接与门禁权限回收流程.md",
            "name": "离职交接与门禁权限回收流程.md",
            "source_path": "/uploads/31ba77e0c9f24a6fbb10.md",
            "status": "PROCESSING", "chunks_count": 0, "acl_scope": "hr",
            "size": 18432, "size_human": "18.0 KB", "modified": NOW - 240,
        },
        {
            "display_name": "门禁权限申请表（模板）.xlsx",
            "name": "门禁权限申请表（模板）.xlsx",
            "source_path": "/uploads/c07e5a1d4b3f4e2b9d61.xlsx",
            "status": "FAILED", "chunks_count": 0, "acl_scope": "public",
            "size": 96256, "size_human": "94.0 KB", "modified": NOW - 86400,
        },
    ],
}

# 语料主题取自 data/ 里的中文制度文本：薪酬 / 加班 / 离职 / 门禁
SEEDS = [
    "薪酬制度", "加班费", "离职交接", "门禁权限", "岗位职级",
    "考勤管理", "绩效评估", "差旅报销", "保密协议", "培训体系",
]
NEIGHBORS = [
    ("工作日加班", "Event", "RELATED_TO"), ("小时工资", "Artifact", "DEPENDS_ON"),
    ("法定节假日", "Event", "RELATED_TO"), ("人力资源部", "Organization", "BELONGS_TO"),
    ("薪酬系统", "Artifact", "USES"), ("试用期", "Event", "APPLIES_TO"),
    ("绩效工资", "Artifact", "PART_OF"), ("张伟", "Person", "RESPONSIBLE_FOR"),
    ("总部办公区", "Location", "LOCATED_IN"), ("权限回收单", "Artifact", "REQUIRES"),
    ("交接清单", "Artifact", "CONTAINS"), ("直属主管", "Person", "REPORTS_TO"),
    ("财务部", "Organization", "APPROVED_BY"), ("离职证明", "Artifact", "REQUIRES"),
    ("年假折算", "Event", "RELATED_TO"), ("门禁卡", "Artifact", "PROHIBITS"),
    ("信息安全部", "Organization", "REGULATES"), ("劳动合同", "Artifact", "DEFINES"),
]
CONCEPT_SEEDS = [("手术", 42), ("糖尿病", 38), ("医院", 31), ("药品", 27), ("治疗", 24)]
CONCEPT_RELS = ["Causes", "CapableOf", "AtLocation", "HasA", "HasSubevent", "HasProperty"]


QA_SOURCES = [
    {
        "type": "path", "score": 1.0625,
        "content": "推理路径: 张伟 -[RESPONSIBLE_FOR]-> 薪酬制度 -[DEPENDS_ON]-> 加班费",
        "metadata": {"from": "张伟", "to": "加班费"},
    },
    {
        "type": "vector", "score": 0.8625,
        "content": "薪酬制度 —[RELATED_TO]→ 工作日加班；工作日加班 —[DEPENDS_ON]→ 小时工资；"
                   "薪酬制度 —[USES]→ 薪酬系统",
        "metadata": {"entity": "薪酬制度", "hops": 2, "template": "neighbors_2hop"},
    },
    {
        "type": "community", "score": 0.9,
        "content": "该子图围绕「薪酬制度」展开：制度由人力资源部维护，"
                   "工作日延长工作时间按小时工资的 1.5 倍计发加班费，"
                   "法定节假日按 3 倍计发，相关取值记录在《薪酬与加班费管理规定》第 3 章。",
        "metadata": {"type": "community_summary"},
    },
    {
        "type": "vector", "score": 0.78,
        "content": "工作日安排劳动者延长工作时间的，支付不低于工资的百分之一百五十的工资报酬；"
                   "法定休假日安排工作的，支付不低于百分之三百的工资报酬。",
        "metadata": {
            "source": "2024年度薪酬与加班费管理规定.pdf",
            "heading_path": "第3章 加班费计算 > 3.2 计发标准",
            "section_title": "3.2 计发标准",
            "chunk_index": 14, "section_type": "text",
        },
    },
    {
        "type": "vector", "score": 0.71,
        "content": "小时工资 = 月工资收入 ÷ (21.75 × 8)。加班费基数按劳动合同约定的工资标准确定，"
                   "不包含年终奖金与一次性补贴。",
        "metadata": {
            "source": "2024年度薪酬与加班费管理规定.pdf",
            "heading_path": "第3章 加班费计算 > 3.1 计算基数",
            "section_title": "3.1 计算基数",
            "chunk_index": 11, "section_type": "text",
        },
    },
    {
        "type": "vector", "score": 0.64,
        "content": "员工离职时应在最后一个工作日完成门禁卡、办公设备与系统账号的回收，"
                   "由直属主管与信息安全部共同确认。",
        "metadata": {
            "source": "离职交接与门禁权限回收流程.md",
            "heading_path": "交接流程 > 权限回收",
            "section_title": "权限回收",
            "chunk_index": 6, "section_type": "text",
        },
    },
]

QA_ANSWER = (
    "按《2024年度薪酬与加班费管理规定》，加班费按以下方式计发：\n\n"
    "1. 工作日延长工作时间，按不低于工资的 150% 支付；法定休假日安排工作，按不低于 300% 支付。\n"
    "2. 小时工资基数为「月工资收入 ÷ (21.75 × 8)」，基数按劳动合同约定的工资标准确定，"
    "不含年终奖金与一次性补贴。\n"
    "3. 该制度由人力资源部维护，薪酬与加班费的取值关系记录在薪酬系统中。\n\n"
    "此外，员工离职时需在最后一个工作日完成门禁卡与系统账号回收，由直属主管与信息安全部共同确认。"
)

REASONING = [
    "意图识别：涉及制度取值与责任主体",
    "实体链接：识别出「薪酬制度」「加班费」「张伟」三个候选实体",
    "BM25 多查询召回：命中 12 个候选章节（薪酬制度 / 工作日加班 / 小时工资）",
    "最短路径检索：张伟 → 薪酬制度 → 加班费",
    "社区摘要：对子图生成高层概括",
    "交叉重排：路径 ×1.25、子图 ×1.15、摘要 ×1.10、向量 ×1.0 后合并排序",
    "答案生成：基于以上 6 条上下文合成",
]

# 上传任务状态机：与 api/main.py 的 progress 映射一致
JOB_STEPS = [
    ("PENDING", 5), ("PROCESSING", 35), ("VECTOR_DONE", 65),
    ("GRAPH_DONE", 90), ("COMMITTED", 100),
]
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


def job_snapshot(doc_id: str) -> dict:
    with JOBS_LOCK:
        job = JOBS.setdefault(doc_id, {"created": time.time(), "name": "上传文档"})
        elapsed = time.time() - job["created"]
        # 每 ~2.2 秒推进一档，约 9 秒走完全流程
        idx = min(len(JOB_STEPS) - 1, int(elapsed / 2.2))
        status, progress = JOB_STEPS[idx]
        done = status == "COMMITTED"
        return {
            "doc_id": doc_id,
            "status": status,
            "progress": progress,
            "file_name": job["name"],
            "chunks_count": 62 if done else 0,
            "entities_count": 48 if done else 0,
            "relations_count": 137 if done else 0,
            "retry_count": 0,
            "error": "",
            "acl_scope": "public",
            "created_at": job["created"],
            "updated_at": time.time(),
        }


# ── HTTP ─────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    server_version = "AKH-Stub/1.0"

    def log_message(self, fmt, *args):  # 安静点
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload, code: int = 200):
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _file(self, path: Path):
        if not path.is_file():
            self._send(404, b"not found", "text/plain; charset=utf-8")
            return
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        if path.suffix == ".js":
            ctype = "application/javascript; charset=utf-8"
        elif path.suffix == ".css":
            ctype = "text/css; charset=utf-8"
        elif path.suffix == ".html":
            ctype = "text/html; charset=utf-8"
        elif path.suffix == ".svg":
            ctype = "image/svg+xml"
        self._send(200, path.read_bytes(), ctype)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        u = urlparse(self.path)
        path, q = u.path, parse_qs(u.query)

        if path in ("/", "/index.html"):
            return self._file(STATIC / "index.html")
        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            target = (STATIC / rel).resolve()
            if STATIC.resolve() not in target.parents and target != STATIC.resolve():
                self._send(403, b"forbidden", "text/plain; charset=utf-8")
                return
            return self._file(target)

        if path == "/api/health":
            return self._json(HEALTH)
        if path == "/api/ui/overview":
            return self._json(OVERVIEW)
        if path == "/api/ui/documents":
            return self._json(DOCUMENTS)
        if path.startswith("/api/jobs/"):
            return self._json(job_snapshot(path.rsplit("/", 1)[-1]))

        self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self):
        u = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""

        if u.path == "/api/qa/ask":
            return self._json({
                "answer": QA_ANSWER,
                "confidence": 0.94,
                "intent": "制度取值与责任主体",
                "sources": QA_SOURCES,
                "reasoning_steps": REASONING,
            })

        if u.path == "/api/ingest/upload":
            # 真实接口是 202 + JobAcceptedResponse，不是 200 + IngestResponse
            name = "上传文档"
            if b"filename=" in raw:
                try:
                    seg = raw.split(b"filename=", 1)[1].split(b"\r\n", 1)[0].strip(b'"')
                    if seg:
                        name = seg.decode("utf-8", "ignore")
                except Exception:
                    pass
            doc_id = "stub-" + str(int(time.time() * 1000) % 100000)
            with JOBS_LOCK:
                JOBS[doc_id] = {"created": time.time(), "name": name}
            return self._json({
                "doc_id": doc_id, "file_name": name, "status": "PENDING",
                "duplicate": False,
                "message": "已接收，正在后台处理，请通过 GET /api/jobs/{doc_id} 查询进度",
            }, 202)

        self._send(404, b"not found", "text/plain; charset=utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[stub] http://{args.host}:{args.port}  static={STATIC}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
