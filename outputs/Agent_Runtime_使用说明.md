# Agent Runtime 使用说明

本文说明如何在本机启动 Agent Runtime、创建和继续对话，以及使用内置工具。当前项目使用自行实现的 Runtime；模型通过 OpenAI Chat Completions 兼容接口接入 DeepSeek。

## 打开当前服务

默认本地示例使用 `http://127.0.0.1:8000`：

- Swagger 接口页面：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/healthz>

在 Swagger 中可以直接试用接口。先调用 `POST /sessions` 创建一个 session，再将返回的 `id` 用于 `POST /sessions/{session_id}/messages`。

## 首次安装

需要 Python 3.11 或更新版本。在项目根目录运行：

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## 本机启动

项目从进程环境变量读取模型配置，不会自动读取 `.env` 文件。当前机器已将配置保存在 Windows 当前用户环境变量中。打开新的 PowerShell 后，可将这些变量加载到当前终端再启动服务：

```powershell
$env:OPENAI_API_KEY = [Environment]::GetEnvironmentVariable('OPENAI_API_KEY', 'User')
$env:OPENAI_MODEL = [Environment]::GetEnvironmentVariable('OPENAI_MODEL', 'User')
$env:OPENAI_BASE_URL = [Environment]::GetEnvironmentVariable('OPENAI_BASE_URL', 'User')
$env:AGENT_DB_PATH = 'agent.sqlite3'

if ([string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) {
    throw 'OPENAI_API_KEY is not configured for this Windows user.'
}

.\.venv\Scripts\python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
```

若 8000 端口已被占用，可改用空闲端口，例如 `8002`，并将示例请求中的地址同步改为该端口。

密钥只保存在本机环境变量中。不要把真实密钥写进 README、示例配置或提交到仓库。`.env` 虽被 Git 忽略，但应用不会自动加载它。

## 创建和继续对话

每个窗口应使用独立的 `session_id`。同一个窗口的后续消息继续使用同一 ID；新窗口再次调用 `POST /sessions` 创建新的 ID。消息、摘要和 Todo 都按 session 隔离并保存在 SQLite 数据库中。

下面的 PowerShell 示例创建窗口 1 并发送消息：

```powershell
$baseUrl = 'http://127.0.0.1:8000'
$session = Invoke-RestMethod `
    -Method Post `
    -Uri "$baseUrl/sessions" `
    -ContentType 'application/json' `
    -Body (@{ title = '窗口 1' } | ConvertTo-Json)

$sessionId = $session.id
$body = @{ content = '计算 12 * (3 + 1)' } | ConvertTo-Json
$reply = Invoke-RestMethod `
    -Method Post `
    -Uri "$baseUrl/sessions/$sessionId/messages" `
    -ContentType 'application/json' `
    -Body $body

$reply | ConvertTo-Json -Depth 8
```

继续同一窗口时复用 `$sessionId`：

```powershell
$body = @{ content = '再算一次，结果是多少？' } | ConvertTo-Json
Invoke-RestMethod `
    -Method Post `
    -Uri "$baseUrl/sessions/$sessionId/messages" `
    -ContentType 'application/json' `
    -Body $body
```

读取窗口历史：

```powershell
Invoke-RestMethod -Method Get -Uri "$baseUrl/sessions/$sessionId/messages"
```

创建第二个窗口时再次请求 `POST /sessions`，并保存它返回的另一个 `id`。不要在窗口之间混用 session ID。

## 内置工具

Agent 会根据消息自行判断是否调用工具。通常直接用自然语言描述任务即可。

| 工具 | 示例输入 | 行为 |
| --- | --- | --- |
| `calculator` | `计算 12 * (3 + 1)` | 支持数字、括号、加减乘除；不执行 Python 代码。 |
| `search` | `搜索 agent runtime` | 返回固定 mock 数据，不会访问互联网。 |
| `todo` | `记一个待办：周五前发周报` | 在当前 session 中新增待办。 |
| `todo` | `列出当前窗口的待办` | 列出当前 session 的待办。 |
| `todo` | `完成刚才那个待办` | 根据当前对话中的待办信息完成相应项目。 |

Todo 按 session 隔离。窗口 1 新增的待办不会出现在窗口 2。Todo 数据以及完整消息历史会写入 `AGENT_DB_PATH` 指定的 SQLite 文件；默认文件是项目目录下的 `agent.sqlite3`。

## 查看会话和运行结果

- `GET /sessions`：列出当前开发身份的会话。
- `GET /sessions/{session_id}`：查看单个会话。
- `GET /sessions/{session_id}/messages`：查看完整消息历史，包括工具结果。
- `POST /sessions/{session_id}/messages`：发送新消息并运行 Agent。

聊天响应中的 `status` 一般为 `completed`。若是 `runtime_not_configured`，启动服务的进程没有读到 `OPENAI_API_KEY`；环境变量变更后需要重启服务。`llm_error` 表示模型请求失败；`invalid_response` 表示模型响应无法解析；`max_turns` 表示本轮达到 Runtime 的模型决策轮数上限。工具调用和工具结果可在消息历史中查看。

## 自动化测试

测试使用 fake LLM 和临时 SQLite 数据库，不会调用 DeepSeek，也不要求 API Key：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 本地运行限制

当前身份是固定的开发身份 `local-user`，接口没有登录认证。服务只绑定到 `127.0.0.1`，适用于本机开发；在部署到其他设备或公开网络前，必须接入真实身份认证并限制访问。
