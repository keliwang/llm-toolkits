# LLM toolkits

`check_prompt_cache.py` 探测 Anthropic Messages 格式接口的 prompt cache。通过
`ANTHROPIC_BASE_URL`、`ANTHROPIC_API_KEY` 或对应命令行参数配置接口。

## Claude Code 风格的 agent 场景

```sh
uv run check_prompt_cache.py YOUR_MODEL --probe-mode agent --timeout 120
```

默认执行所有七个场景，使用 `claude-code` 缓存策略；可重复指定 `--scenario`：

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
仍对已发生的请求给出缓存结论，并照常执行最终重放。工具错误重试不在场景范围内。

### 请求策略

参考本机 Claude Code **2.1.295** 的完整历史请求路径及
[官方缓存说明](https://code.claude.com/docs/en/prompt-caching)：

- tools 的定义和顺序、system 和项目上下文在场景内保持稳定。
- 项目上下文用会话开头的 `<system-reminder>` 文本表示；后续只在末尾追加内容。
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

### 静态命中与历史命中

每个场景/策略使用独立随机前缀，避免不同场景和运行之间互相预热。流程是：

1. 用相同 tools、system 和 thinking 配置发一个只标记 system 的短请求，校准
   静态缓存 token 数。这个请求会预热 tools+system，报告为 `static_calibration`。
   如果无法从写入量建立基线，会原样重放该短请求，报告为
   `static_calibration_replay`。当同一请求的 read 增加、input 等量减少且二者之和
   保持一致时，按隐式缓存的“命中 + 未命中输入”口径处理，并使用整个短请求的
   输入总量作为静态前缀的保守上界；不会用可能只命中部分前缀的 read 当作其
   完整长度。如果计数口径无法确认，历史复用仍报告未知。
2. 执行实际 agent 对话，每轮回传完整历史并记录 usage。
3. 场景完成后原样重放最后一次请求，验证最后一次写入的历史前缀。重放响应
   不加入主对话，报告为 `verification_replay`。

报告把 `progression`（实际 agent 推进）和 `verification_replay`（诊断重放）
分别汇总，校准和重放不计入实际推进的命中比例、重复输入未命中量或验收结果。
`history_read_beyond_static_observed` 和 `thinking_present_on_history_hit` 现在只
描述实际推进：前者表示至少一次 cache read 超过静态参考值；后者只在某个请求的
**期望缓存前缀里已经包含 thinking 块**（即上一请求已把它作为输入发出）且该请求
判定为 `HIT` 时为 true，本请求才首次发送的 thinking 不算。没有这样的请求时为
`null`。`thinking_prefix_hits/thinking_prefix_checks` 给出对应计数。它们仍不能
证明每个 thinking token 都命中。

### 逐请求命中判定

每个实际推进请求都对比“理论上应该读到多少”和“实际读到多少”：

- 首个请求：期望读到校准预热的静态前缀 `S`（tools + system）。
- 之后的请求：上一个请求在末尾打了断点，期望读到上一个请求的完整输入 `P`。

| 标签 | 条件 | 含义 |
| --- | --- | --- |
| `HIT` | `read ≥ 阈值 × 期望`，且（首个请求除外）`read > S` | 上一轮写入的前缀基本全部命中 |
| `PARTIAL` | 读到了超过 `S` 的部分，但未达阈值 | 历史部分命中，`missed` 为漏掉的 token |
| `STATIC` | `0 < read ≤ S` | 只命中 tools+system，历史完全没有被证明命中 |
| `MISS` | `read = 0` | 完全未命中 |
| `UNKNOWN` | 前缀/配置变化、usage 口径未知、静态参考缺失等 | 无法判定，附带原因 |

阈值就是 `--min-prefix-reuse`（默认 0.95）。报告中每个场景显示一条命中时间线，
例如 `HIT → HIT → STATIC → HIT`，以及首个非 `HIT` 请求的阶段、期望值和实际值，
可以直接定位哪一轮丢了缓存。期望前缀已包含 thinking 块的请求标记为 `HIT(t)`
等形式，每轮详情中的 `thinking_in_expected_prefix` 给出块数。JSON 中对应 `hit_timeline`、`hit_counts`、
`missed_tokens`、`first_non_hit` 和每轮的 `hit`。

### 逐轮复用与成本暴露

先确认前后请求的 tools、system、thinking/effort 等配置不变，历史只是追加；
比较时忽略移动的 `cache_control` 标记，但完整保留工具参数和 thinking 签名。
再将 usage 统一成输入总量 `T`、读取缓存量 `C` 和未读取缓存量 `T-C`。
设上一轮输入总量为 `P`，在服务端保持相同分词及连续前缀缓存的前提下：

| 指标 | 计算与意义 |
| --- | --- |
| `cache_ratio` | `C/T`，这一轮整体输入命中比例 |
| `old_input_reuse_ratio_estimate` | `min(C,P)/P`，已经作为输入发送过的旧前缀复用率 |
| `old_input_miss_tokens_estimate` | `max(P-C,0)`，重复发送但未读到缓存的旧输入 |
| `new_input_tokens_estimate` | `T-P`，本轮新增输入，包括上一轮刚生成、首次回传的 assistant 内容 |
| `old_history_reuse_ratio_lower_bound` | 静态参考值 `S` 为长度或保守上界时，`max(min(C,P)-S,0)/(P-S)`；只在 `P>S` 时报告 |

旧输入包括 tools、system 和旧消息，**旧输入复用率不是纯对话历史复用率**。
历史复用率下界用于避免静态上下文很长时掩盖历史未命中；下界为 0 表示没有
正下界证据，不表示历史一定完全没命中。前缀或配置改变、usage 口径不明、
输入长度异常（包括同一请求重放却改变长度）时，复用指标报告未知。
这些是基于汇总 token 计数的估计，无法校验网关内部改写或 thinking 过滤，
也不能按消息/块精确归因。首轮没有上一轮 agent 输入，因此没有旧输入指标。

例如你提供的普通对话：上一轮 `T=6075`，下一轮 `C=5120, T=6174`，则旧输入
约有 **955 tokens** 未命中，复用率约 **84.28%**，新增输入约 **99 tokens**。
即使最终重放命中率达到 99.51%，也不能掩盖推进时的这次未命中。

`not_read_tokens=T-C` 不等于按普通输入价格收费的 token 数：显式缓存接口
还可能把它拆成写入和普通输入，两者价格不同。保留原始 read/create/input
计数；缺失 create 始终是未知。脚本不猜测网关价格或把 token 比例当成实际
费用折扣。实际费用需要结合对应接口的缓存读取、写入和普通输入单价。

默认七个场景正常完成且校准提供写入量时共 34 次请求（包括校准和重放）；
需要校准重放时最多增加 7 次请求。实际未完成的场景
可能提前结束。`--max-agent-requests` 限制每个场景的主链路请求数，默认 8。
`--agent-turns` 控制普通对话的用户轮数，默认 3。`--rounds` 仅用于原有快速探测。

### 对照、预览和结果

只测 thinking 工具链，同时比较两种缓存策略：

```sh
uv run check_prompt_cache.py YOUR_MODEL --probe-mode agent \
  --scenario thinking_interleaved --scenario thinking_followup \
  --cache-strategy claude-code --cache-strategy system \
  --timeout 120 --json
```

`system` 对照要求实际推进观察到缓存读取，显示 `STATIC HIT`。
`claude-code` 的缓存结论由命中时间线得出：

| 结论 | 条件 |
| --- | --- |
| `PASS` | 至少有一次推进，且所有推进请求都是 `HIT` |
| `REUSE UNKNOWN` | 存在 `UNKNOWN`，或只有首个请求 |
| `PARTIAL REUSE` | 后续请求中有 `HIT`/`PARTIAL`，但不全是 `HIT` |
| `REPLAY ONLY` | 推进中从未读到超过 `S` 的内容，只有诊断重放读到了 |
| `STATIC ONLY` | 只有静态前缀读取 |
| `CACHE NOT VERIFIED` | 没有任何缓存读取 |

95% 是可调整的验收目标，不是任何提供方的缓存块大小、定价或完整历史命中
保证。需要零估计遗漏时可设为 `1`，但分词边界也可能造成小差异。

报告标题同时显示 `cache=<结论>` 和 `flow=COMPLETE|INCOMPLETE (原因)`。
JSON 里 `cache_requirement_met` 只看缓存，`passed` 要求两者都满足；
顶层对应 `all_cache_requirements_met` 和 `all_scenarios_passed`。
退出码按缓存结论：全部场景/策略满足缓存要求时为 0，否则为 1。

不发请求，只查看场景计划和首轮请求体：

```sh
uv run check_prompt_cache.py YOUR_MODEL --probe-mode agent \
  --scenario thinking_interleaved --dry-run
```

### 读取报告

默认文本报告每个场景一张表，只保留判断缓存所需的列：

```
[tools_sequential/claude-code] cache=PARTIAL REUSE  flow=COMPLETE  reuse=49.53% (min 0.00%)  static=5932 (implicit)
   #  stage       resp           input    read    Δread  expect  miss  hit
   1  initial     think,tool×1    6025    5888        -    5932    44  HIT
   2  tool_result think,tool×1    6122    6016     +128    6025     9  HIT
   3  tool_result think,end       8000       0    ▼6016    6122  6122  MISS(t)  ◀ read fell to 0
   r  replay      think,end       8000    7936    +7936    8000    64  HIT(t)
```

| 列 | 含义 |
| --- | --- |
| `resp` | 本次响应形态：是否有 thinking、调用了几个工具或正常结束 |
| `input` | 本次输入总量（按计数口径归一化） |
| `read` | 本次缓存读取量 |
| `Δread` | 与上一个请求的读取量之差；`▼` 表示下降 |
| `expect` | 上一个请求留下的可缓存量（首个请求为静态前缀） |
| `miss` | `expect - read` |
| `hit` | 命中标签，`(t)` 表示期望前缀里已有 thinking 块 |

对话只追加时，读取量不应下降。只要某个请求读得比上一个请求少（`decrease`），
或者从非零跌到 0（`zero`），该行会标出 `◀`，报告末尾汇总到
`⚠ Read drops`，JSON 中对应各阶段的 `read_drops` 和每轮的 `read_change`。
校准请求只在出错时显示；完整的逐轮 usage 和估算明细使用 `--verbose`。

JSON 报告包含逐轮 usage、命中比例、耗时、stop_reason、thinking/工具数量、
缓存断点位置以及请求/system/tools 的哈希；不输出请求鉴权或 thinking 内容。
`usage_accounting` 记录使用的计数口径，不按模型名做分支。
默认 `--usage-accounting auto` 通过校准建立口径；若第三方计数语义已确认，
可显式选择以下值，避免未公开字段使自动判断失败：

| 值 | 输入总量 |
| --- | --- |
| `anthropic` | `input + cache_read + cache_creation`，三项须提供 |
| `implicit` | `input + cache_read`，input 包括全部未命中部分 |
| `total` | `input` 已包含所有输入，cache_read 是其中的子集 |

这些选择只改变本地统计，不改变请求体。显式配置应依据对应网关的 usage
语义，不能只因为缺少 create 就选择 implicit。计数缺失或自相矛盾时仍报告未知。
`--no-stream` 可测试非流式接口。需要额外 beta 或网关鉴权时，可重复指定
`--header 'Name: Value'`。Agent 模式下 `--beta-mode auto` 不做旧缓存 beta 的
自动回退，以免改变场景内的请求配置；`on` 可显式添加旧 header。

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
uv run python -m unittest -v test_agent_cache_probe
```

测试使用合成 Messages 响应和本地 MockTransport，验证签名回传、并行工具
结果配对、流式事件重建、静态命中误判、场景隔离、重放掩盖推进未命中、
计数口径与前缀变化；不调用真实 API。真实缓存
能力需要用目标接口运行场景套件验证。
