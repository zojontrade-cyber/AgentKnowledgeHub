FROM python:3.12-slim

# 依赖锁定：优先使用 lock 文件（由 pip-compile / uv 生成），缺失时回落到
# requirements.txt 并发出警告。生产构建应始终提供 requirements.lock。
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-chi-sim poppler-utils curl \
    && rm -rf /var/lib/apt/lists/*

# 先拷依赖清单，利用 Docker 层缓存
COPY requirements.txt requirements.lock* ./

# 有 lock 用 lock（可复现），没有则用 requirements.txt（不推荐用于生产）
RUN if [ -f requirements.lock ]; then \
        echo "使用 requirements.lock 安装（可复现）" && \
        pip install --no-cache-dir --require-hashes -r requirements.lock; \
    else \
        echo "警告: 未找到 requirements.lock，使用 requirements.txt（不可复现）" && \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# 创建非 root 用户 —— 原镜像以 root 运行，叠加任意文件写入即为容器逃逸级风险
RUN groupadd --gid 10001 appuser \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin appuser

COPY --chown=appuser:appuser . .

RUN mkdir -p /app/uploads && chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

# 健康检查：命中真实探测端点（依赖不可用时返回 503）
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/api/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
