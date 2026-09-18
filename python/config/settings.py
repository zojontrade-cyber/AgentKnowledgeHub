"""
应用配置 — 通过环境变量或 .env 文件加载

重构要点
--------
原实现所有字段都是裸 str/int 带默认值，**零校验器**。后果：
  - 漏配 OPENAI_API_KEY → 应用正常启动，首次调用才 401
  - VECTOR_STORE_TYPE 拼错 → 静默落入 pgvector 分支（vector_store.py:36-40）

现在用 pydantic 校验：关键配置非法时**启动即失败**，而不是运行期静默降级。
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

# 明确拒绝的占位/弱默认值
_PLACEHOLDER_SECRETS = {
    "", "password", "changeme", "your-password", "sk-your-api-key-here",
    "sk-your-key", "secret", "admin",
}


class Settings(BaseSettings):
    # ── LLM（对话/推理）───────────────────────────────────────
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"
    # LLM 调用超时与重试（避免请求无限挂起）
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2

    # ── Embedding（向量化）────────────────────────────────────
    # 为什么要与对话模型分开配置：
    #   部分服务商只提供对话能力而不提供 embedding 接口
    #   （典型如 DeepSeek，调用 embeddings 端点一律 404）。
    #   因此这里允许 embedding 使用**独立**的 endpoint 与 Key。
    #
    # 留空时自动回落（继承）对话端的配置，保持向后兼容：
    #   EMBEDDING_API_KEY  未设置 -> 用 OPENAI_API_KEY
    #   EMBEDDING_BASE_URL 未设置 -> 用 OPENAI_BASE_URL
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    # embedding 请求超时（批量入库时单次可能较大）
    embedding_timeout_seconds: float = 60.0

    # ── 问答模式（Pipeline vs ReAct）─────────────────────────
    #
    # pipeline = 现状：LLM 只在入口做一次意图路由（5 选 1），
    #            之后按静态计划**硬编码**并发调用检索器。
    #            决策颗粒度：入口 1 次。
    #
    # react    = LLM 逐步决策：每轮自己选工具与参数，看到 Observation
    #            后再决定继续或停止。决策颗粒度：每一步。
    #
    # 为什么保留两者可切换：架构已冻结，且本项目已实测过
    # 「更智能的机制 ≠ 更好」—— 因此 ReAct 必须先经 A/B 验证，
    # 而不是直接替换主链路。
    qa_mode: Literal["pipeline", "react"] = "pipeline"

    # 生成 Prompt 的结构。
    #   stable（默认）= 稳定 system 段（常量，所有问题逐字节相同）
    #                 + 动态段固定顺序 metadata → context → query，
    #                 上下文标签不带分数/类型（见 services/prompt_builder.py）
    #   legacy        = 改前的结构（意图风格插进 system、标签带类型与分数）
    #
    # 为什么保留 legacy：改造生成 Prompt 会影响答案质量，而"改前"的基线必须
    # 能重跑才能做 A/B。旧评测脚本已失效、历史 JSON 也无法确认同源，
    # 因此把旧结构固化在 prompt_builder 里，靠这个开关回到改前行为。
    answer_prompt_mode: Literal["stable", "legacy"] = "stable"

    # ReAct 循环上限（含首次检索）。
    # 4 轮可覆盖：单跳 1 轮 / 双跳 2 轮 / 需补充证据 3 轮 / 复杂多跳 4 轮。
    # 上限过高会让 Agent 为"更全面"做无效检索，只增延迟不增质量。
    qa_max_steps: int = 4

    # ── 对话历史（展示保留）────────────────────────────────
    # 存每一轮问答的快照，用于刷新 / 重开后恢复界面。
    # **不注入 prompt** —— 问答链路仍是单轮无记忆，见 docs/conversation-history.md
    #
    # 保留天数。0 = 永久保留（不清理）。
    qa_history_retention_days: int = 30
    # 清理任务间隔（小时）。retention_days <= 0 时不启动该任务。
    qa_history_purge_interval_hours: int = 6

    # ── Reranker（检索第二阶段：精排）─────────────────────────
    # 检索已由 BM25 解决召回（Chunk R@5 97.4%），瓶颈转为**排序**
    # （Chunk@1 仅 76.3%）。Reranker 是 cross-encoder，对 (query, doc)
    # 逐对精算相关性，正是作用于"候选已在池中但没排第一"。
    #
    # 实测（38 题 exact，data/eval/p2_rerank.json）：
    #         Chunk@1   Chunk@5    MRR     耗时
    #   关闭   76.3%     97.4%    0.852     1ms
    #   开启   86.8%     97.4%    0.917    548ms
    #   -> Top1 +10.5pp，Recall@5 无变化，MRR +0.065
    #
    # 关键：rerank 只改善**排序**，不改善**召回**。若 R@5 本身低，
    # 应先去修召回（换检索通道），而不是开 reranker。
    rerank_enabled: bool = True
    # 留空则跳过重排（向后兼容）。硅基流动可用：BAAI/bge-reranker-v2-m3
    rerank_model: str = ""
    rerank_timeout_seconds: float = 30.0
    # 粗召回数量（重排的输入规模）。
    # 实测 pool=20 与 pool=50 指标完全相同（86.8%/97.4%/0.917），
    # 但耗时 548ms vs 884ms —— 说明 20 已足够，加宽只增加延迟。
    rerank_candidates: int = 20

    # ── 数据层（SQLite）──────────────────────────────────────
    # 单一事实源：文档状态机 + Outbox + 文件版本 + 实体别名 + 审计日志。
    # 用 SQLite 是为了零外部依赖；迁移 Postgres 时只需替换
    # services/database.py 里的连接与方言，业务代码不变。
    sqlite_path: str = "./data/agenthub.db"
    # 任务最大重试次数（超过则标记 FAILED，不再自动重试）
    max_retries: int = 3
    # 后台 worker 轮询间隔（秒）
    worker_poll_interval: float = 2.0
    # 是否在 API 进程内启动后台 worker（单机部署用；多副本时应独立部署）
    worker_enabled: bool = True
    # uvicorn 热重载开关。
    #
    # **默认关闭，生产/开发都不建议开启。**
    #
    # 原因：reload=True 时 uvicorn 会监控文件变化并重启 worker。若在
    # startup 过程中有文件变动（改代码、写日志、生成缓存等），会在
    # lifespan 尚未完成时触发重启，导致 workflows 里持有的
    # vector_store 处于**半初始化状态**：
    #   - /api/health 报 ok
    #   - /api/ui/overview 能正确报告 159 条向量
    #   - 但 search() 返回空 -> 智能问答拿不到任何检索结果
    #
    # 实测（2026-09）：同代码同数据下，reload=True 的实例对
    # "请假要走什么流程" 返回「无法回答」，reload=False 的实例
    # 正确返回考勤管理制度的完整审批链。
    #
    # 需要热重载时显式设置 API_RELOAD=true，并确保改动完成后
    # 再发请求。
    api_reload: bool = False


    # ── 检索模式（主召回通道）─────────────────────────────────
    # bm25   = BM25 字面召回（**默认**，实测 Chunk R@5 89.1%）
    # dense  = 稠密向量召回（旧默认，实测 Chunk R@5 仅 23.6%）
    # hybrid = BM25+稠密 RRF 融合（实测低于纯 BM25，仅供对照）
    #
    # 依据：在 159 制度章节 / 78 题文档级 / 55 题 chunk 级评测集上，
    # retrieval-only（无 reranker）实测见 data/eval/p1_results.json。
    # 制度语料含大量判别性低频词项（请假/年假/保密/印章），
    # BM25 视其为强信号，稠密模型将其稀释。
    retrieval_mode: Literal["bm25", "dense", "hybrid"] = "bm25"
    # hybrid 模式下 BM25 的融合权重（实测 0.7 仍不及纯 BM25）
    hybrid_bm25_weight: float = 0.7

    # ── Vector Store ─────────────────────────────────────────
    # 用 Literal 约束，拼错立即报错而不是静默走错分支
    vector_store_type: Literal["chroma", "pgvector"] = "chroma"
    # chroma 连接方式：
    #   persistent = 嵌入式本地文件模式（无需 Docker，数据落 chroma_path）
    #   http       = 连接独立 Chroma 服务端（需 Docker 或远程实例）
    chroma_mode: Literal["persistent", "http"] = "persistent"
    chroma_path: str = "./chroma_data"
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    pgvector_dsn: str = "postgresql://postgres:postgres@localhost:5432/knowledge"


    # ── API ──────────────────────────────────────────────────
    # 默认绑回环而非 0.0.0.0：避免默认暴露到所有网卡
    api_host: str = "127.0.0.1"
    api_port: int = 8080

    # ── 认证与限流 ───────────────────────────────────────────
    # 逗号分隔的 API Key 列表；为空则鉴权关闭（仅供本地开发）
    api_keys: str = ""
    admin_api_keys: str = ""
    rate_limit_per_minute: int = 60
    auth_enabled: bool = True

    # ── 数据级 ACL（访问控制）────────────────────────────────
    # 格式：key=scope1|scope2,key2=scope3
    #   public  —— 所有人可见（检索始终包含）
    #   其他值   —— 需调用方 Key 的 scope 匹配才可见
    # 示例：API_KEY_SCOPES=dev-key-1=public|hr,dev-key-2=public
    # 未配置的 Key 默认只能看到 public 文档。
    api_key_scopes: str = ""
    # 新上传文档的默认可见范围
    default_acl_scope: str = "public"

    @property
    def key_scope_map(self) -> dict[str, list[str]]:
        """解析 API_KEY_SCOPES 为 {key: [scope, ...]}"""
        out: dict[str, list[str]] = {}
        for item in (self.api_key_scopes or "").split(","):
            item = item.strip()
            if not item or "=" not in item:
                continue
            k, scopes = item.split("=", 1)
            out[k.strip()] = [s.strip() for s in scopes.split("|") if s.strip()]
        return out

    # ── 上传限制 ─────────────────────────────────────────────
    upload_dir: str = "./uploads"
    max_upload_bytes: int = 20 * 1024 * 1024  # 20MB

    # ── 运行环境 ─────────────────────────────────────────────
    # dev: 允许宽容启动；prod: 严格 fail-fast
    environment: Literal["dev", "prod"] = "dev"
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # ── 校验 ─────────────────────────────────────────────────

    @field_validator("openai_base_url")
    @classmethod
    def _check_base_url(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError(
                f"OPENAI_BASE_URL 必须是 http(s) URL，当前为 {v!r}"
            )
        return v.rstrip("/")

    @field_validator("embedding_base_url")
    @classmethod
    def _check_embedding_base_url(cls, v: str) -> str:
        """embedding 端 URL 校验（允许为空 -> 回落对话端）"""
        v = (v or "").strip()
        if not v:
            return ""
        if not v.startswith(("http://", "https://")):
            raise ValueError(
                f"EMBEDDING_BASE_URL 必须为空或 http(s) URL，当前为 {v!r}"
            )
        return v.rstrip("/")

    # ── embedding 配置解析（含回落逻辑）──────────────────────

    @property
    def effective_embedding_api_key(self) -> str:
        """embedding 实际使用的 Key：独立配置优先，否则继承对话端"""
        return self.embedding_api_key or self.openai_api_key

    @property
    def effective_embedding_base_url(self) -> str:
        """embedding 实际使用的 endpoint：独立配置优先，否则继承对话端"""
        return self.embedding_base_url or self.openai_base_url

    @property
    def embedding_uses_separate_endpoint(self) -> bool:
        """embedding 是否使用了与对话不同的服务商"""
        return bool(self.embedding_base_url) and (
            self.embedding_base_url != self.openai_base_url
        )

    @field_validator("max_upload_bytes")
    @classmethod
    def _check_upload_limit(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("MAX_UPLOAD_BYTES 必须为正整数")
        return v

    @field_validator("rate_limit_per_minute")
    @classmethod
    def _check_rate_limit(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("RATE_LIMIT_PER_MINUTE 必须为正整数")
        return v

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def admin_key_set(self) -> set[str]:
        return {k.strip() for k in self.admin_api_keys.split(",") if k.strip()}

    @property
    def is_prod(self) -> bool:
        return self.environment == "prod"

    @model_validator(mode="after")
    def _fail_fast_on_insecure_prod(self) -> "Settings":
        """
        生产环境下拒绝弱配置 —— 启动即失败，而不是带病运行。

        这是重构前最危险的问题之一：默认密码 + 空 API Key 会让服务
        "看起来正常"地跑起来，直到出事。
        """
        if not self.is_prod:
            return self

        problems: list[str] = []

        if not self.openai_api_key or self.openai_api_key.lower() in _PLACEHOLDER_SECRETS:
            problems.append("OPENAI_API_KEY 未配置或仍为占位值")

    
        if self.auth_enabled and not self.api_key_set:
            problems.append("ENVIRONMENT=prod 且 AUTH_ENABLED=true，但 API_KEYS 为空")

        if self.auth_enabled and not self.admin_key_set:
            problems.append("ENVIRONMENT=prod 但 ADMIN_API_KEYS 为空（/admin 无管理员可达）")

        if problems:
            raise ValueError(
                "生产环境配置校验失败，拒绝启动:\n  - " + "\n  - ".join(problems)
            )

        return self


settings = Settings()
