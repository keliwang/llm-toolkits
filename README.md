# LLM toolkits

`check_prompt_cache.py` 探测 Anthropic Messages 格式接口的 prompt cache。通过
`ANTHROPIC_BASE_URL`、`ANTHROPIC_API_KEY` 或对应命令行参数配置接口。

## Claude Code 风格的 agent 场景

```sh
uv run check_prompt_cache.py YOUR_MODEL --probe-mode agent --timeout 120
```

默认执行所有七个场景，可重复指定 `--scenario`：

| 场景 | 实际请求流程 |
| --- | --- |
| `text_multiturn` | 三个用户轮次，完整回传回答，在末尾追加用户消息与 system-reminder |
| `tools_sequential` | Read manifest → 回传结果 → Read 结果中给出的 entrypoint → 回传结果 → 最终回答 |
| `tools_parallel` | 同一响应里两个 Read → 在同一 user 消息里回传全部 tool_result → 最终回答 |
| `thinking_tool` | 真实 thinking + Read main.py → 回传结果 → Glob `/probe/*.py` → 回传结果 → 最终回答 |
| `thinking_interleaved` | thinking + Read manifest → thinking + Read entrypoint → 最终回答 |
| `thinking_parallel` | thinking + 并行 Read → 在同一 user 消息回传全部结果 → Glob `/probe/*.py` → 回传结果 → 最终回答 |
| `thinking_followup` | 完成 thinking 工具链后，再追加普通用户追问，保留全部历史 |

工具返回虚拟 `/probe` 仓库的数据，不访问真实文件，也不执行命令。Thinking 和
tool_use 必须由目标模型实际生成；未出现预期行为时流程报告 `INCOMPLETE`，不会用
伪造 thinking 或普通续问补成成功。

thinking 块第一次作为输入发送时属于新增内容，要到再下一次请求才可能从缓存读到。
因此每个 thinking 场景都至少有三次推进请求：`thinking_tool` 和 `thinking_parallel`
在首轮工具后再做一次 Glob，保持纯工具循环；`thinking_followup` 用普通追问覆盖
“工具链后接普通 user 消息”的情况；`thinking_interleaved` 靠第二次依赖读取。
流程未走到“读取含 thinking 的前缀”那一步时，报告 `thinking_history_not_exercised`。流程结论和缓存结论相互独立：流程未完成时
仍对已发生的请求给出缓存结论。工具错误重试不在场景范围内。

### 请求策略

参考本机 Claude Code **2.1.295** 的完整历史请求路径及
[官方缓存说明](https://code.claude.com/docs/en/prompt-caching)：

- tools 的定义和顺序、system 和项目上下文在场景内保持稳定。
- 项目上下文（含探测文本）用会话开头 user 消息里的 `<system-reminder>` 表示；后续只在末尾追加内容。
- 每次发送完整历史，原样保留 assistant 块，包括 thinking 的 signature 和
  redacted_thinking 的 data。
- 默认设置 system 末尾和最后一个可缓存消息末尾的显式 cache_control。
  Thinking 块本身不设置标记。所有断点使用同一个 TTL。
- 保持默认 tool_choice，不强制工具、不在首轮后改变该参数。
- 默认使用 SSE；流式重建工具 JSON、thinking 文本、签名，并合并 usage。

Claude Code 的额外前一消息断点由服务端功能开关控制；可用
`--pin-previous-message` 模拟该选项，默认关闭。TTL 默认 `5m`，对应 API key
主会话策略；`--cache-ttl 1h` 可显式选择一小时。

这是请求结构和多轮行为的模拟，不是 Claude Code 的逐字节复制：不复制私有
system prompt、订阅鉴权、billing metadata、工具搜索、thread 增量协议和
服务端实验，也不模拟每个新模型的专有 system-message 扩展。

### Thinking 配置

默认所有 agent 场景使用 `thinking: {"type": "adaptive"}`，并携带
`output_config: {"effort": "high"}`。配置在整个场景内保持不变。网关上的模型
不一定都支持这些参数，可以调整：

| 参数 | 效果 |
| --- | --- |
| `--thinking adaptive` | 默认，发送 `{"type": "adaptive"}` |
| `--thinking enabled` | 发送 `{"type": "enabled", "budget_tokens": N}`，`N` 由 `--thinking-budget` 指定（默认 4096，需 ≥1024 且小于 `--max-tokens`），用于不支持 adaptive 的模型 |
| `--thinking off` | 不发送 `thinking` 字段；未指定 `--scenario` 时只跑非 thinking 场景，显式指定 thinking 场景会报错 |
| `--effort none` | 不发送 `output_config`；其他值照常发送对应档位 |

Adaptive 可能在简单请求中不产生 thinking，此时相应 thinking 专项流程报告未完成，
但缓存结论照常给出。普通对话和普通工具场景允许模型产生 thinking，只检查对应的
对话与工具流程；产生的 thinking 和签名同样完整回传。手动预算模式下交错 thinking
可能需要额外的 beta header，可用 `--header` 添加。

例如扫一个不支持 thinking/effort 的模型：

```sh
uv run check_prompt_cache.py YOUR_MODEL --probe-mode agent --thinking off --effort none
```

### 命中判定

每个场景只跑一遍真实 agent 循环，逐请求记录 usage，不再有静态校准和最终重放。
默认七个场景正常完成时共 20 次请求。

Claude Code 在最后一条消息上打缓存断点，所以第 n 个请求应至少读到第 n-1 个请求的
输入总量 `P = input + cache_read + cache_creation`（Anthropic Messages 口径，input
不含缓存读写）。规则与 `claude_code_probe.py` 相同：

| 标签 | 条件 |
| --- | --- |
| `FIRST` | 场景首个请求，没有期望值 |
| `HIT` | `read ≥ 阈值 × P` |
| `PARTIAL` | `0 < read < 阈值 × P` |
| `MISS` | `read = 0` |
| `UNKNOWN` | 本次或上次 usage 缺失 |

阈值是 `--min-prefix-reuse`（默认 0.95），用于容忍分块缓存造成的少量尾部差异；
需要零遗漏时可设为 `1`。期望前缀里已包含 thinking 块（上一请求已把它作为输入发出）
的请求标为 `HIT(t)` 等形式，`thinking_prefix_hits/thinking_prefix_checks` 给出计数；
本请求才首次发送的 thinking 不算。

为避免“只命中 tools+system 却因占比高被判 HIT”，大段项目上下文（`--repeat-count`
控制的探测文本）放在首条 user 消息的 `<system-reminder>` 里，与 Claude Code 放
CLAUDE.md 的位置一致；tools+system 本身短于最小缓存长度。因此读到上一请求的
完整输入只能来自消息断点。每个场景使用独立随机 system 前缀，互不预热。

场景结论：

| 结论 | 条件 |
| --- | --- |
| `PASS` | 首个请求之后全部 `HIT` |
| `PARTIAL REUSE` | 有 `HIT`/`PARTIAL`，但不全是 `HIT` |
| `CACHE NOT VERIFIED` | 全部 `MISS` |
| `REUSE UNKNOWN` | 存在 `UNKNOWN`，或只有一个请求 |

报告标题同时显示 `cache=<结论>` 和 `flow=COMPLETE|INCOMPLETE (原因)`。
JSON 里 `cache_requirement_met` 只看缓存，`passed` 要求两者都满足；
顶层对应 `all_cache_requirements_met` 和 `all_scenarios_passed`。
退出码按缓存结论：全部场景满足缓存要求时为 0，否则为 1。

`--max-agent-requests` 限制每个场景的请求数，默认 8；`--agent-turns` 控制普通对话的
用户轮数，默认 3。`--rounds` 仅用于原有快速探测。

### 预览和结果

只测 thinking 工具链：

```sh
uv run check_prompt_cache.py YOUR_MODEL --probe-mode agent \
  --scenario thinking_interleaved --scenario thinking_followup \
  --timeout 120 --json
```

不发请求，只查看各场景首轮请求体：

```sh
uv run check_prompt_cache.py YOUR_MODEL --probe-mode agent \
  --scenario thinking_interleaved --dry-run
```

### 读取报告

文本报告每个场景一张表，格式与真实会话探测相同：

```
[tools_sequential] cache=PARTIAL REUSE  requests=3  reuse=83.25%  overall_read=50.19%  missed=2035
    #  turn  after       resp             input    read   Δread  expect   miss  hit  tools
    1     1  prompt      think,tool×1      6025       0       -       -      -  FIRST    Read
    2     1  tool_result think,tool×1      6122    6016   +6016    6025      9  HIT      Read
    3     1  tool_result think,end_turn    8000    4096   ▼1920    6122   2026  PARTIAL(t)   ◀ read decreased
  timeline: FIRST → HIT → PARTIAL(t)
  first non-HIT: #3 turn 1 after tool_result: expect 6122, read 4096 (PARTIAL)
  thinking-prefix hits: 0/1
  flow=COMPLETE
```

| 列 | 含义 |
| --- | --- |
| `after` | 本请求跟在新的用户输入（`prompt`）还是 `tool_result` 之后 |
| `resp` | 本次响应形态：是否有 thinking、调用了几个工具或结束原因 |
| `input` | 本次输入总量 `input + cache_read + cache_creation` |
| `read` | 本次缓存读取量 |
| `Δread` | 与上一个请求的读取量之差；`▼` 表示下降 |
| `expect` | 上一个请求的输入总量 |
| `miss` | `max(expect - read, 0)` |

对话只追加时读取量不应下降；下降或跌到 0 的行标出 `◀`，JSON 中对应 `read_drops`
和每个请求的 `read_drop`。`reuse` 是所有推进请求 `Σmin(read, expect) / Σexpect`，
`missed` 是对应的未命中 token 合计。

JSON 报告包含每个请求的 usage、命中标签、耗时、stop_reason、thinking/工具数量、
缓存断点位置和请求哈希；不输出请求鉴权或 thinking 内容。有 `input_tokens` 时，
缺失的缓存计数按 0 处理（部分网关省略值为 0 的字段），常见 camelCase 别名会被
归一化；整个 usage 缺失时报告 `UNKNOWN`。不支持把缓存读取算进 `input_tokens`
的非标准网关。这些都是汇总 token 计数，不能按消息/块精确归因，也不等于实际费用。
`--no-stream` 可测试非流式接口。需要额外 beta 或网关鉴权时，可重复指定
`--header 'Name: Value'`。Agent 模式下 `--beta-mode auto` 不做旧缓存 beta 的
自动回退，以免改变场景内的请求配置；`on` 可显式添加旧 header。

### 录制请求与回复

排查问题时加 `--record DIR`，把每个请求和回复原样写到本地：

```sh
uv run check_prompt_cache.py YOUR_MODEL --probe-mode agent --record recordings
```

每次运行新建 `DIR/<时间>-<run_id 前 8 位>/`，其中每个场景一个 `<场景>.jsonl`，每个请求
一行，请求返回后立即写入，进程卡住或中断时已发生的请求也会保留；场景全部结束后再写入
`report.json`（即 `--json` 的报告，含每个请求的命中标签，按 `index` 与 jsonl 对应）。
每行字段：

| 字段 | 内容 |
| --- | --- |
| `index` / `turn` / `after` | 与报告表格相同的请求序号、用户轮次和前一条输入类型 |
| `started_at` / `elapsed_ms` | 发出时间（UTC）和耗时 |
| `request` | `url`、`headers`、完整请求体 `body`（含 `cache_control` 断点和完整历史） |
| `response.status_code` / `headers` | HTTP 状态码和响应头 |
| `response.body` | 原始响应文本；流式时是完整 SSE 流，解析失败或非 2xx 时同样保留 |
| `response.message` | 解析后的 Messages 对象（流式时为重组结果），失败时为 `null` |
| `usage` | 归一化的 `input/read/creation/total/output` |
| `error` | 请求或解析错误（如有） |

请求头和响应头里名字包含 key、token、auth、secret、cookie、password、signature 的值
替换为 `<redacted>`，所以 `x-api-key` 和 `--header 'Authorization: ...'` 不会落盘。
但请求和回复正文会完整保存，**包括 thinking 文本和签名**，以及你通过 `--header`
加入的其他非敏感头；分享录制文件前请自行检查。

常用查看方式：

```sh
R=recordings/<时间>-<run>
# 每个请求一行：usage、消息数、最后一条消息和回复的块类型
jq -c '{index, after, usage, msgs: (.request.body.messages|length),
        last: (.request.body.messages[-1].content|map(.type)),
        resp: (.response.message.content|map(.type))}' $R/tools_sequential.jsonl
# 第 3 个请求的完整请求体 / 断点位置
jq 'select(.index == 3) | .request.body' $R/tools_sequential.jsonl
jq -c '.request.body.messages | to_entries | map(select(any(.value.content[]; has("cache_control"))) | .key)' $R/tools_sequential.jsonl
# 原始 SSE 流
jq -r 'select(.index == 1) | .response.body' $R/thinking_tool.jsonl
```

请求体包含完整历史，因此相邻两行可以直接 diff 确认历史只是追加。

## 真实 Claude Code 会话探测

`claude_code_probe.py` 不再模拟请求：它在隔离的临时目录里生成一个小 Python
仓库，用 `claude -p --input-format stream-json` 驱动真实 Claude Code CLI 完成一个
多轮任务，结束后直接分析该会话的 session log（`projects/<cwd>/<session>.jsonl`
以及 `<session>/subagents/*.jsonl`）。

```sh
# 网关 / API key：配置目录完全隔离（新建空的 CLAUDE_CONFIG_DIR）
uv run claude_code_probe.py --model YOUR_MODEL --base-url https://gw.example.com --api-key sk-... --subagent

# 只分析已有 session log，例如自己日常的会话
uv run claude_code_probe.py --analyze ~/.claude/projects/<project>/<session>.jsonl
```

同一个 Claude Code 进程、同一个 session 内依次发送以下用户轮次，每轮等到
`result` 事件后再发下一轮：

| 轮次 | 任务 | 覆盖的环节 |
| --- | --- | --- |
| `fix` | 读 PROBE.md → 读 config/project.json 找入口 → 并行读两个配置 → Bash 跑测试 → thinking 后 Edit 修 bug → 再跑测试 | 顺序工具链、并行工具、Bash、Edit、thinking 与工具交错 |
| `explain` | 不用工具，解释修复 | 普通文本多轮；历史中已有 thinking |
| `extend` | 新增 `test_unknown_currency` 并运行测试 | 普通 user 消息之后再进入工具链 |
| `subagent`（`--subagent`） | 通过 Agent/Task 工具启动 general-purpose 子 agent 读文件 | 子 agent 的 sidechain 缓存 |

隔离方式：

- workspace 是新建临时目录（或 `--root` 指定的空目录），不是 git 仓库，没有 CLAUDE.md。
- 默认新建空的 `CLAUDE_CONFIG_DIR`：不加载本机用户 settings、CLAUDE.md、插件、hooks、
  MCP 和登录态，session log 也写在运行目录里。
- 真实运行必须显式传 `--api-key` 或 `--auth-token`（至少一个），否则直接报错退出。
  当前环境里的 `ANTHROPIC_BASE_URL`、`ANTHROPIC_API_KEY`、`ANTHROPIC_AUTH_TOKEN`、
  `CLAUDE_CODE_OAUTH_TOKEN` 一律不传给子进程，避免误用别的凭据或误打到别的接口；
  不指定 `--base-url` 时使用官方 API。`--analyze` 不需要凭据。
- 工具限定为 Read/Glob/Grep/Edit/Write/Bash（`--subagent` 时加 Agent），`acceptEdits`
  模式，Bash 只放行 `python3 -m unittest`；`--strict-mcp-config` 不加载 MCP。
- 剥离父 Claude Code 会话注入的环境变量（`CLAUDECODE`、`CLAUDE_CODE_SESSION_ID` 等），
  可以在 Claude Code 里运行本脚本。

运行目录保留 `workspace/`、`stream.jsonl`（CLI 输出事件）、`stderr.log` 和隔离配置，
便于复查。`--claude-arg` 原样透传额外 CLI 参数，例如 `--claude-arg=--max-budget-usd=1`。

### 分析方法

session log 里每个内容块一行；并行工具调用时 `tool_result` 会插在同一响应的块之间，
所以按 `message.id`（缺失时用 `requestId`）在整个文件内归并为一次 API 请求，跳过
`<synthetic>` 和 API 错误消息。每个链（主会话、每个子 agent）独立分析：

- Claude Code 在最后一条消息上打缓存断点，因此第 n 个请求应至少读到第 n-1 个请求的
  输入总量 `P = input + cache_read + cache_creation`。
- `read ≥ 阈值 × P` 为 `HIT`，`0 < read` 为 `PARTIAL`，`read = 0` 为 `MISS`；首个请求标为
  `FIRST`，没有期望值。`(t)`、`Δread`、`◀` 读取下降标记以及 `PASS`/`PARTIAL REUSE`/
  `CACHE NOT VERIFIED`/`REUSE UNKNOWN` 的含义与 agent 模拟场景一致（两者共用同一判定代码）。
- `after` 列区分请求是跟在 `tool_result` 还是新的用户输入之后，汇总
  `user-prompt boundary hits`，用于定位“换轮时丢缓存”。
- 正常情况下每步只差末尾少量未缓存 token（Claude Code 会话里通常是 2）。

流程检查（`flow=COMPLETE|INCOMPLETE`）确认任务真的走过了每个环节：多轮 Read、
单次响应多个工具、Bash、Edit、纯文本轮、普通追问后的工具调用、thinking 出现且
进入缓存前缀、子 agent 日志，并在结束后自己运行一次单元测试确认任务完成。模型
不产生 thinking 时可用 `--allow-no-thinking`。

退出码：主会话和所有子 agent 链都 `PASS` 时为 0，否则为 1；CLI 运行出错或找不到
session log 时为 2。真实运行会调用模型并产生费用；Claude Code 自身的标题生成等
辅助请求不写入 session log，不在分析范围内。

## 原有快速探测

```sh
uv run check_prompt_cache.py YOUR_MODEL
uv run check_prompt_cache.py YOUR_MODEL --probe-mode repeat --json
```

默认保留原有 `auto` 模式：工具多轮优先，然后按缓存证据尝试普通/beta header
和相同请求重放。快速探测的成功标准是观测到缓存写入或读取，适合检查接口
基础能力，不等同于 agent 场景套件通过。

## 本地验证

```sh
uv run python -m unittest -v test_agent_cache_probe test_claude_code_probe
```

测试使用合成 Messages 响应和本地 MockTransport，验证签名回传、并行工具
结果配对、流式事件重建、只命中 tools+system 不能通过、场景隔离、逐请求
命中标签与计数口径；不调用真实 API。真实缓存
能力需要用目标接口运行场景套件验证。
