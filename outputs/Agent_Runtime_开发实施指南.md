# 自研 Agent Runtime 开发实施指南

## 1. 目标与边界

实现一个由项目自身控制的 Agent Runtime：接收用户消息，调用 LLM 判断直接回答或请求工具，校验并执行工具，再把结果交回 LLM，直到得到最终答案或达到限制。核心循环、工具注册、参数校验、执行、session 与上下文管理都由本项目实现，不使用 LangGraph、OpenHands、OpenClaw 等 Agent 框架。

LLM SDK 可以作为模型通信客户端；它只负责发送请求和接收响应，不负责替代本项目的循环与状态管理。建议先将模型通信封装为 `LLMClient`，这样以后可以更换供应商，也便于用假客户端测试 Runtime。

本文按 Python 3.11+、FastAPI、SQLite 给出一个适合 MVP 的落地方案。若项目已有语言或 Web 框架，可以保留接口和状态设计，替换技术栈。

## 2. MVP 范围

第一版实现以下内容即可：

- 一个聊天接口：创建 session、继续指定 session。
- 一个自行编写的 `AgentRuntime.run()` 循环，默认最多 5 次模型决策。
- 工具注册表；每个工具有名称、描述、JSON Schema 参数、执行函数。
- 三个工具：`calculator`、可 mock 的 `search`、必须持久化的 `todo`。
- LLM 可以通过工具调用格式选择工具；Runtime 校验参数并负责执行。
- session 历史持久化，两个窗口使用不同 `session_id`。
- 基础上下文压缩：保留最近消息，将更早的对话折叠为一段摘要。
- 单元测试覆盖工具、解析、循环上限和 session 隔离。

第一版不做多 Agent、计划执行框架、复杂长期记忆、向量数据库、流式中断恢复或复杂摘要评估。

## 3. 推荐模块结构

```text
app/
  api.py                 # HTTP 接口
  runtime.py             # Agent 循环与轮次限制
  llm_client.py          # 模型供应商适配层
  response_parser.py     # 统一解析模型响应
  context.py             # 上下文组装与基础压缩
  sessions.py            # session 读写
  tools/
    base.py              # Tool 定义、注册表、JSON Schema 校验
    calculator.py
    search.py
    todo.py
  storage.py             # SQLite 初始化与查询
tests/
  test_runtime.py
  test_tools.py
  test_sessions.py
```

模块边界的关键是：模型只能提出工具调用，不能直接执行工具；只有 Runtime 能从注册表查找、校验并调用工具。

## 4. 一次请求如何运行

1. API 根据 `user_id` 和 `session_id` 读取或创建 session，将新用户消息写入历史。
2. Context Builder 选取系统规则、工具定义、历史摘要、最近消息和当前用户输入。
3. Runtime 调用 LLM。LLM 返回最终答案，或返回一个/多个工具调用。
4. Parser 将供应商响应归一化为内部类型。Runtime 检查轮次限制、工具是否注册、参数是否符合 Schema。
5. Runtime 执行工具，把每个工具结果作为 `tool` 消息写入同一 session，再回到第 2 步调用 LLM。
6. LLM 返回最终答案后，Runtime 保存 assistant 消息并返回 API。
7. 如果超出最大轮次、模型失败或工具连续失败，返回清楚的错误/部分结果；不要无限重试。

建议区分“模型请求次数”和“工具执行次数”。例如 `max_model_turns=5`，每次模型回复最多允许 3 个工具调用。每个 HTTP 请求也可设总超时和 token 上限。

伪代码：

```python
async def run(user_id: str, session_id: str, user_text: str) -> str:
    session = await sessions.get_owned(user_id, session_id)
    await sessions.append(session_id, role="user", content=user_text)

    for turn in range(MAX_MODEL_TURNS):
        context = await context_builder.build(session_id)
        response = await llm.respond(context, tools=registry.schemas())
        decision = parse_response(response)

        if decision.kind == "final":
            await sessions.append(session_id, "assistant", decision.text)
            return decision.text

        await sessions.append_assistant_tool_calls(session_id, decision.tool_calls)
        for call in decision.tool_calls:
            result = await registry.execute(call.name, call.arguments)
            await sessions.append_tool_result(session_id, call, result)

    return "我已达到本轮工具调用上限，请缩小任务范围或继续提问。"
```

实际实现还需要捕获模型错误、Schema 错误和工具异常，按统一格式写入工具结果或返回可诊断错误。不要把异常堆栈原样暴露给用户。

## 5. 工具注册与 Schema

每个工具至少定义：稳定的 `name`、给模型看的 `description`、参数 JSON Schema、异步执行函数。启动时集中注册，重复名称应直接报错。

```python
@registry.register(
    name="todo",
    description="创建、列出或完成当前会话的待办事项。",
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "list", "complete"]},
            "text": {"type": "string", "description": "待办内容；add 时必填"},
            "todo_id": {"type": "string", "description": "complete 时必填"}
        },
        "required": ["action"],
        "additionalProperties": False
    }
)
async def todo_tool(args: dict, context: ToolContext) -> dict:
    ...
```

工具 Schema 是模型决策时可见的契约，不是安全边界。服务端仍需验证类型、必填项、枚举、长度、权限和业务规则。Schema 校验可使用 `jsonschema` 或 Pydantic；校验失败时不执行函数，并将简短、可修复的错误反馈给模型。

### 必须实现的工具

| 工具 | 建议参数 | MVP 行为 |
| --- | --- | --- |
| `calculator` | `expression: string` | 解析有限的四则运算、括号和小数；拒绝任意 Python 代码、变量和函数调用。避免直接使用 `eval`。 |
| `search` | `query: string` | 返回固定 mock 结果，包含标题、摘要和来源标识；以后替换成真实搜索实现时保持同一 Schema。 |
| `todo` | `action`, `text?`, `todo_id?` | 支持新增、列出、完成。待办写入数据库，执行结果包含稳定 ID 和状态。 |

Todo 的归属建议首版按 `session_id` 隔离，因为需求中不同窗口代表互不影响的对话。将来若希望“跨窗口共享待办”，可以增加明确的 `scope=user` 能力，不要依赖模型猜测归属范围。

可选增加 `read_docs`：参数为文档标识或查询词，只允许读取配置好的文档目录；不要接受任意文件系统路径。

## 6. LLM 响应与解析

优先使用模型 API 的原生工具调用能力，并把注册表生成的 Schema 传给模型。Runtime 仍需解析和验证返回值。若所选模型没有原生工具调用，可要求它输出严格 JSON，作为兼容路径。

建议 Runtime 内部只处理统一格式：

```json
{"kind":"tool_calls","tool_calls":[{"id":"call_1","name":"calculator","arguments":{"expression":"12 * (3 + 1)"}}]}
```

或：

```json
{"kind":"final","text":"结果是 48。"}
```

Parser 应处理：没有工具调用的最终文本、合法工具调用、多个工具调用、参数 JSON 解码失败、未知工具、空响应，以及供应商返回的拒绝/错误状态。遇到非法调用时，不执行工具；可以将结构化错误反馈给模型再尝试一次，但重试也必须计入轮次上限。

关于“思考过程”：不要要求或保存模型的私有逐步推理。很多模型不会提供这类内容，也不适合作为稳定协议。若调试或产品确实需要解释，使用模型明确输出的简短 `decision_summary`（例如“需要查询天气工具”），并将它视为普通说明文本；真正需要保存的是用户消息、assistant 最终回复、工具调用参数和工具结果。生产日志应避免记录密钥和不必要的个人数据。

## 7. Session 与持久化

一个 session 表示一条独立对话；窗口只是客户端展示，同一窗口继续聊天时复用 `session_id`，新窗口创建新的 `session_id`。不能用 `user_id` 单独作为会话键。

最小数据模型：

```text
sessions
  id, user_id, title, summary, created_at, updated_at

messages
  id, session_id, sequence, role, content, tool_call_json, tool_name, created_at

todos
  id, user_id, session_id, text, status, created_at, completed_at
```

所有读取和写入都应同时检查 `session_id` 与当前 `user_id`，防止用户访问他人的会话。每条消息用递增 `sequence` 或可靠时间戳排序。SQLite 足以支撑本地 MVP；多实例部署时再考虑 PostgreSQL 和并发锁。

建议 API：

```text
POST /sessions                  创建 session
GET  /sessions                  列出当前用户的 sessions
GET  /sessions/{session_id}     获取消息历史
POST /sessions/{session_id}/messages  继续对话
```

继续对话示例：

```json
{
  "session_id": "sess_123",
  "message": "刚才那个待办再加上周五前完成"
}
```

认证方式可先用开发环境固定 `user_id`，但所有数据层接口仍保留用户参数；接入真实产品时再换成登录态身份，不能让客户端随意传入并信任任意用户 ID。

## 8. Context 如何选取和压缩

每次模型调用建议包含：

- 稳定的系统规则：角色、工具使用要求、安全边界。
- 当前可用工具的名称、说明和 Schema。
- session 的 `summary`（没有则为空）。
- 最近若干条原始消息，包括工具调用与工具结果。
- 当前用户消息；若已写入消息表，避免重复拼接。

不应把数据库全量历史、所有用户所有 session 的消息、完整调试日志或不可用的隐藏推理塞进 context。Todo 等可查询的持久状态优先通过对应工具读取，不必永远复制到历史提示词。

基础压缩策略：当最近历史超过设定条数或估算 token 阈值时，取较早的一段对话生成短摘要，更新 `sessions.summary`，保留最近 8 至 12 条原始消息。摘要仅保留后续回答需要的事实、用户偏好、已完成/未完成事项和待确认点；原始消息仍留在数据库供查看。若摘要生成失败，保留旧摘要和消息，不要清空历史。

追问的关键是：`session_id` 稳定、历史消息按序保存、工具结果进入同一历史。纯对话追问由最近对话和摘要支撑；带工具追问则让模型先读取历史，再通过工具查询最新状态。工具结果不能只放在进程内变量中，否则跨请求追问会丢失。

## 9. 错误处理与安全要求

- 工具只从注册表调用；禁止模型输出任意函数名后动态导入执行。
- 不用 `eval` 算表达式；搜索 mock 与未来真实搜索都限制超时、结果数和文本长度。
- 工具异常转成统一结果，如 `{"ok": false, "error": {"code": "...", "message": "..."}}`，不返回堆栈、密钥或服务器路径。
- 对工具参数设置长度、数值范围、请求超时和调用次数上限。
- Todo 查询和变更均校验 session 所属用户；完成操作使用稳定 ID，不凭模糊文本猜测目标。
- 模型输出只作为不可信输入。即使 Schema 合法，也要验证业务权限和副作用。
- 记录 `request_id`、`session_id`、模型耗时、工具名、轮次和错误码；避免默认记录敏感原文。

## 10. 分阶段实现与验收

### 阶段 A：骨架与接口

先创建 API、SQLite 表、`LLMClient` 协议和假客户端。验收：无需真实 API Key 就能运行测试；创建两个 session 后数据互不串线。

### 阶段 B：工具系统

实现注册表、Schema 校验和 `calculator`、mock `search`、持久化 `todo`。验收：非法参数、未知工具、越权 Todo 都不会执行或泄漏数据。

### 阶段 C：Runtime loop

实现统一响应类型和最多 5 轮循环，使用可控的假 LLM 响应测试直接回答、一次工具调用、多次工具调用、工具失败、非法 JSON、超限退出。验收：所有路径终止，工具结果会传回下一次模型调用。

### 阶段 D：真实模型接入

按选定供应商实现 `LLMClient`，将 Tool Registry Schema 转换为供应商格式，再把返回格式转换成 Runtime 内部格式。验收：模型能在无需硬编码路由的情况下选择 calculator、search 和 todo。

### 阶段 E：追问与压缩

实现历史读取、上下文拼装、摘要压缩和两个窗口恢复测试。验收场景：窗口 1 创建待办并询问天气；窗口 2 写周报并创建另一个待办；回到窗口 1 追问天气或待办时，只使用窗口 1 的历史和默认范围内的数据。

### 阶段 F：端到端验收

- “计算 12 * (3 + 1)”得到 48，并有 calculator 工具记录。
- “搜一下示例主题”返回 mock search 结果。
- “记一个待办：周五前发周报”后，列出待办能看到它；完成后状态更新。
- 同一 session 的“刚才那个待办”能解析到正确上下文并调用 todo。
- 两个 session 的消息、摘要、待办隔离；重启服务后仍存在。
- 达到最大轮次时返回受控结果，不无限调用模型。
- 上下文压缩后保留摘要和最近消息，session 历史页面仍能查看完整原文。

## 11. 如何分阶段使用 Codex 协助实现

把本文件放在项目根目录，并在同一个项目聊天中按阶段推进。每次只让 Codex 完成一个可验证阶段；要求它先阅读现有代码和项目约定，再改动最少的文件、运行相关测试并报告结果。不要一次要求生成完整产品，这样出错时难以定位。

可直接使用以下提示词，按顺序发送：

**第一步：检查项目并制定落地切片**

```text
请阅读项目现状和开发文档（Agent_Runtime_开发实施指南.md），确认现有技术栈与约定。不要使用 LangGraph、OpenHands、OpenClaw 或其他 Agent Runtime 框架。先给出与现有项目匹配的最小实现切片、需要新增/修改的文件和测试命令；暂不改代码。若仓库已有实现或用户改动，先说明并沿用，不要覆盖。
```

**第二步：骨架和数据层**

```text
按开发文档阶段 A 实现 API 骨架、SQLite 初始化、session/messages/todos 数据访问层及可替换的 LLMClient 协议。先确保测试不依赖真实模型或 API Key。完成后运行测试，并报告文件、接口和结果。
```

**第三步：工具系统**

```text
按开发文档阶段 B 实现 Tool 定义、注册表、JSON Schema 参数校验，以及 calculator、mock search、持久化 todo（add/list/complete）。calculator 不得使用 eval。补充针对正常输入、非法参数、未知工具和 session 隔离的测试，然后运行测试。
```

**第四步：Agent Runtime**

```text
按开发文档阶段 C 实现 response parser 和自研 AgentRuntime 循环：模型可直接回答或提出工具调用；Runtime 校验、执行、保存工具结果并继续调用模型；轮次和单轮工具数有限。用脚本化 fake LLM 覆盖直接回答、工具调用、连续调用、工具错误、无效响应和达到上限。不要接入 Agent 框架。
```

**第五步：真实模型适配**

```text
阅读当前 LLMClient 和工具注册表，为项目配置的模型供应商增加适配器。供应商原生工具调用格式映射到 Runtime 内部统一类型；密钥从环境变量读取，不写入仓库。保留 fake LLM 测试，并给出本地配置方式。不要声称已完成真实模型验证，除非实际运行过。
```

**第六步：追问与压缩**

```text
按开发文档阶段 E 实现基于 session 的 context builder 和最基础摘要压缩：超过阈值时摘要较早对话、保留最近消息、历史原文仍持久化。确保模型请求不重复加入当前用户消息，摘要失败时不丢历史。补充纯对话追问、带工具追问、两个 session 隔离和重启恢复测试。
```

**第七步：最终审查**

```text
请对照 Agent_Runtime_开发实施指南.md 的验收清单检查实现，运行相关测试并做一次代码审查。优先报告会造成无限循环、跨 session 数据泄漏、任意代码执行、工具越权或历史丢失的问题；发现问题请直接修复并重新运行受影响测试。最后列出尚未实现的范围和真实模型验证所需条件。
```

Codex 每次完成后，先看它报告的改动和测试结果，再决定是否进入下一阶段。若结果不符合需求，直接指出具体场景、期望行为和失败输出，让它修复并补充回归测试。涉及真实 API Key 时只在本机环境变量或本地 `.env` 中配置，并确认 `.env` 已被 Git 忽略。

## 12. 最终交付应包含什么

- 可运行的服务和清晰的本地启动说明。
- 工具列表、参数 Schema、错误结果格式和 API 示例。
- 不依赖模型服务的自动化测试。
- `.env.example`（只放变量名和占位说明，不放真实密钥）。
- 两窗口/两个 session 的端到端验收说明。
- 已知限制，例如 mock search、模型供应商要求和摘要策略。

交付的核心证明不是“模型看起来会聊天”，而是 Runtime 能稳定地执行：模型决定、Schema 校验、受控工具执行、结果回填、继续或终止；并且不同 session 的上下文和状态可以独立恢复。
