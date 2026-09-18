# 运行手册：启动 / 停止 / 端口冲突

## 启动

```bat
start-api.bat
```

等价于：

```bat
cd /d "%~dp0python"
set PYTHONIOENCODING=utf-8
python -m api.main
```

- 默认监听 `http://127.0.0.1:8080`（`API_HOST` / `API_PORT` 可覆盖）
- 文档页 `http://127.0.0.1:8080/docs`
- API Key：`dev-key-1`（管理员 `dev-admin-key-1`）
- 不需要任何外部服务（嵌入式 Chroma + SQLite）

## 停止

```bat
stop-api.bat
```

按 `API_PORT`（默认 8080）找到 LISTENING 的进程并结束。
**若进程是提权启动的**，普通 shell 杀不掉 —— 脚本会提示你改用管理员命令行：

```bat
taskkill /PID <pid> /F
```

## 端口被占用时会发生什么

`api/main.py` 在 `uvicorn.run()` **之前**做一次端口预检（`_preflight_port`）。

### 情况一：占用者是本服务

```
端口 8080 上已有本服务在运行：http://127.0.0.1:8080
无需重复启动。若要重启：先运行 stop-api.bat，再启动。
```

**退出码 0** —— 目标状态（服务在跑）已经达成，这不是失败。

### 情况二：占用者是别的程序

```
端口 8080 已被其它程序占用（PID 25104），服务无法启动。

  查看占用：netstat -ano | findstr :8080
  结束进程：taskkill /PID <pid> /F   （若为提权进程，需以管理员身份执行）
  换端口  ：set API_PORT=8090 && python -m api.main
  停已有实例：stop-api.bat
```

**退出码 1**，并且**不会初始化任何东西**。

## 为什么需要这个预检（历史教训）

uvicorn 的顺序是：**先把 `lifespan` 整个跑完，再去 bind 端口**。
所以端口被占时，真实发生的是：

```
向量库初始化成功            ← Chroma 已经打开了
编排流水线就绪
后台 worker 已启动
对话历史清理任务已启动
Application startup complete
ERROR: [Errno 10048] ... 通常每个套接字地址只允许使用一次   ← 到这里才失败
后台 worker 已停止
服务已关闭
```

两个问题：

1. **原因被淹没**：日志以"服务已关闭"收尾，看起来像"启动失败"，但看不出为什么
2. **有副作用**：两个进程会短暂同时打开同一个 Chroma 目录 / SQLite 文件

预检把这两点都消掉了：先判断端口，再决定要不要初始化。

### 适用范围

预检挂在 `api/main.py` 的 `__main__` 里，因此覆盖 **`python -m api.main`（即 `start-api.bat` 的入口）**。

若你直接用 `uvicorn api.main:app --port 8080` 启动，预检**不会**执行，
仍会看到 uvicorn 原生的 `[Errno 10048]` —— 此时用上面的 `netstat` / `taskkill` 排查。

## 提交前的自检

```bat
cd python
python -m pytest tests/ -q
```

HTTP 与浏览器端到端（需要服务已启动）：

```bat
python scripts\verify_chat_metrics.py
python scripts\verify_chat_history.py
python scripts\verify_ui_metrics.py
```

> **本机注意**：这些脚本用 `httpx.Client(trust_env=False)`。
> 本机注册表配了系统代理，httpx 默认会把 `127.0.0.1` 的请求也走代理而**挂死**
> （表现为 ReadTimeout，但服务端其实 6.8s 内就正常返回了 200）。
> 不要把它误判成服务端卡死。
