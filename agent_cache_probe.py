"""Claude Code 风格的 Messages agent 缓存探测。

参考 Claude Code 2.1.295 的普通、完整历史请求路径：稳定的 tools/system，
会话开头的项目上下文，追加历史，最后一个可缓存消息上的显式断点。
前一消息断点是可选项（Claude Code 中由功能开关控制）。不复制其私有提示词、
订阅鉴权、服务端实验、thread 增量协议或工具搜索。所有工具均为本地虚拟数据。

每个场景只跑一遍 agent 循环并记录每个请求的 usage。由于断点打在最后一条消息上，
第 n 个请求应读到第 n-1 个请求的完整输入；判定规则与 claude_code_probe 相同。
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import fnmatch
import json
from pathlib import Path
import time
from typing import Any
import uuid

import httpx

from claude_code_probe import _usage_totals, classify_chain, print_chain

SCENARIOS = (
    "text_multiturn",
    "tools_sequential",
    "tools_parallel",
    "thinking_tool",
    "thinking_interleaved",
    "thinking_parallel",
    "thinking_followup",
)
REFERENCE = "Claude Code 2.1.295; full-history API-key/gateway request profile"
REFERENCE_URL = "https://code.claude.com/docs/en/prompt-caching"
THINKING_SCENARIOS = tuple(scenario for scenario in SCENARIOS if scenario.startswith("thinking"))
THINKING_TYPES = {"thinking", "redacted_thinking"}
USAGE_ALIASES = {
    "cache_creation_input_tokens": ("cache_creation_input_tokens", "cacheCreationInputTokens", "cache_creation_tokens", "cacheCreationTokens"),
    "cache_read_input_tokens": ("cache_read_input_tokens", "cacheReadInputTokens", "cache_read_tokens", "cacheReadTokens"),
    "input_tokens": ("input_tokens", "inputTokens"),
    "output_tokens": ("output_tokens", "outputTokens"),
}
# Header names containing these are redacted in recordings (credentials, cookies).
SENSITIVE_HEADER_PARTS = ("key", "token", "auth", "secret", "cookie", "password", "signature")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _cache_control(ttl: str) -> dict[str, str]:
    return {"type": "ephemeral", **({"ttl": "1h"} if ttl == "1h" else {})}


def _cacheable(block: dict[str, Any]) -> bool:
    if block.get("type") in THINKING_TYPES:
        return False
    return block.get("type") != "text" or bool(block.get("text", "").strip())


def build_request(
    model: str,
    system: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    *,
    ttl: str,
    thinking: dict[str, Any] | None,
    effort: str | None,
    max_tokens: int,
    stream: bool,
    pin_previous_message: bool,
) -> dict[str, Any]:
    # Mark only a copy of the wire request. Stored history, particularly signatures,
    # must survive unchanged as markers move forward on later requests.
    body = deepcopy({"model": model, "system": system, "tools": tools, "messages": messages})
    for block in body["system"]:
        block.pop("cache_control", None)
    for message in body["messages"]:
        if isinstance(message["content"], str):
            message["content"] = [{"type": "text", "text": message["content"]}]
        for block in message["content"]:
            block.pop("cache_control", None)
    body["system"][-1]["cache_control"] = _cache_control(ttl)
    # Ordinary Claude Code marks the tail message, skipping assistant messages
    # whose final block is thinking. Optional fork pin marks the prior message.
    eligible = [
        message for message in body["messages"]
        if message["content"] and _cacheable(message["content"][-1])
    ]
    for message in eligible[-(2 if pin_previous_message else 1):]:
        message["content"][-1]["cache_control"] = _cache_control(ttl)
    body.update(max_tokens=max_tokens, stream=stream)
    if thinking is not None:
        body["thinking"] = deepcopy(thinking)
    if effort is not None:
        body["output_config"] = {"effort": effort}
    # Like Claude Code's normal agent loop, leave tool_choice at the API default.
    # Changing tool_choice during the agent loop can invalidate message caches.
    return body


def _request_audit(body: dict[str, Any]) -> dict[str, Any]:
    markers = [f"system[{index}]" for index, block in enumerate(body["system"]) if "cache_control" in block]
    for index, message in enumerate(body["messages"]):
        for block_index, block in enumerate(message["content"]):
            if "cache_control" in block:
                markers.append(f"messages[{index}].content[{block_index}]")
    return {"request_sha256": _digest(body), "message_count": len(body["messages"]), "cache_breakpoints": markers}


def _read_sse(response: httpx.Response, raw_lines: list[str] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] | None = None
    blocks: dict[int, dict[str, Any]] = {}
    tool_json: dict[int, str] = {}
    stopped = False

    def consume(data: str) -> None:
        nonlocal payload, stopped
        event = json.loads(data)
        kind = event.get("type")
        if kind == "error":
            raise ValueError(f"SSE error: {json.dumps(event.get('error'), ensure_ascii=False)}")
        if kind == "message_start":
            payload = deepcopy(event["message"])
        elif kind == "content_block_start":
            index = event["index"]
            if index in blocks:
                raise ValueError("Duplicate SSE content block index")
            blocks[index] = deepcopy(event["content_block"])
        elif kind == "content_block_delta":
            index = event["index"]
            if index not in blocks:
                raise ValueError("SSE delta without content_block_start")
            delta = event["delta"]
            field = {"text_delta": "text", "thinking_delta": "thinking", "signature_delta": "signature"}.get(delta["type"])
            if field:
                blocks[index][field] = blocks[index].get(field, "") + delta[field]
            elif delta["type"] == "input_json_delta":
                tool_json[index] = tool_json.get(index, "") + delta["partial_json"]
            else:
                raise ValueError(f"Unsupported SSE delta: {delta['type']}")
        elif kind == "message_delta":
            if payload is None:
                raise ValueError("SSE message_delta without message_start")
            payload.update(event.get("delta", {}))
            # Input/cache counts usually arrive in message_start; output in delta.
            payload.setdefault("usage", {}).update(event.get("usage", {}))
        elif kind == "message_stop":
            stopped = True

    data_lines: list[str] = []
    for line in response.iter_lines():
        if raw_lines is not None:
            raw_lines.append(line)
        if not line:
            if data_lines:
                consume("\n".join(data_lines))
                data_lines = []
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        consume("\n".join(data_lines))
    if payload is None or not stopped:
        raise ValueError("Incomplete SSE message (missing message_start/message_stop)")
    for index, value in tool_json.items():
        blocks[index]["input"] = json.loads(value)
    payload["content"] = [blocks[index] for index in sorted(blocks)]
    return payload


def _usage(payload: dict[str, Any] | None) -> dict[str, int | None]:
    """Anthropic Messages usage: total = input + cache_read + cache_creation.

    Gateway aliases are normalized first. With input reported, an omitted cache
    counter counts as 0 (some gateways drop zero-valued fields).
    """
    usage = (payload or {}).get("usage")
    if not isinstance(usage, dict):
        usage = {}
    return _usage_totals({
        metric: next((usage[key] for key in keys if type(usage.get(key)) in (int, float)), None)
        for metric, keys in USAGE_ALIASES.items()
    })


def _redact_headers(headers: Any) -> dict[str, str]:
    return {
        name: "<redacted>" if any(part in name.lower() for part in SENSITIVE_HEADER_PARTS) else value
        for name, value in headers.items()
    }


def _send(
    client: httpx.Client, url: str, headers: dict[str, str], body: dict[str, Any],
    trace: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Send one request. With ``trace``, also capture the raw response for recording."""
    started = time.perf_counter()
    record: dict[str, Any] = _request_audit(body)
    payload = None
    raw_lines: list[str] = []
    try:
        with client.stream("POST", url, headers=headers, json=body) as response:
            record["status_code"] = response.status_code
            if trace is not None:
                trace["headers"] = _redact_headers(response.headers)
            if not response.is_success:
                response.read()
                record["error"] = response.text[:1000]
                raw_lines.append(response.text)
            else:
                if "text/event-stream" in response.headers.get("content-type", ""):
                    payload = _read_sse(response, raw_lines)
                else:
                    response.read()
                    raw_lines.append(response.text)
                    payload = response.json()
                if not isinstance(payload, dict) or not isinstance(payload.get("content"), list):
                    raise ValueError("Response must be a Messages object with content blocks")
                if not all(isinstance(block, dict) for block in payload["content"]):
                    raise ValueError("Invalid response content block")
    except (httpx.RequestError, ValueError, KeyError, TypeError) as exc:
        payload = None
        record.setdefault("status_code", 0)
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    if trace is not None:
        # Raw body as received (SSE lines included), kept even when parsing failed midway.
        trace["body"] = "\n".join(raw_lines)
        trace["message"] = payload
    content = payload["content"] if payload else []
    record.update(
        _usage(payload),
        stop_reason=payload.get("stop_reason") if payload else None,
        thinking_blocks=sum(block.get("type") in THINKING_TYPES for block in content),
        tools=[block.get("name") for block in content if block.get("type") == "tool_use"],
    )
    return record, payload


def _record_exchange(
    path: Path, record: dict[str, Any], started_at: str, url: str,
    headers: dict[str, str], body: dict[str, Any], trace: dict[str, Any],
) -> None:
    """Append one full request/response exchange; written immediately so a hang or crash keeps earlier lines."""
    line = {
        "index": record["index"], "turn": record["turn"], "after": record["after"],
        "started_at": started_at, "elapsed_ms": record["elapsed_ms"],
        "request": {"method": "POST", "url": url, "headers": _redact_headers(headers), "body": body},
        "response": {"status_code": record["status_code"], **trace},
        "usage": {key: record[key] for key in ("input", "read", "creation", "total", "output")},
        **({"error": record["error"]} if "error" in record else {}),
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(line, ensure_ascii=False) + "\n")


def _tools() -> list[dict[str, Any]]:
    # Stable ordering and definitions throughout every conversation.
    return [
        {
            "name": "Read", "description": "Read a UTF-8 file in the virtual /probe repository. Returns its contents.",
            "input_schema": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"], "additionalProperties": False},
        },
        {
            "name": "Glob", "description": "List paths in the virtual /probe repository matching a glob pattern.",
            "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"], "additionalProperties": False},
        },
    ]


def _initial_messages(scenario: str, probe_text: str) -> list[dict[str, Any]]:
    # The bulk context lives in the first user message, like Claude Code's CLAUDE.md
    # reminder. tools+system stay too short to cache alone, so a read reaching
    # the previous request's input can only come from the message breakpoint.
    project = (
        "<system-reminder>\n# CLAUDE.md\nThis is a virtual Python repository. "
        "Use the Read tool to inspect files; never invent their contents. "
        "Working directory: /probe. Environment and project instructions remain fixed for this session.\n\n"
        f"{probe_text}\n</system-reminder>"
    )
    if scenario == "text_multiturn":
        task = "Do not call tools. Remember these requirements: Python 3.11, no external dependencies, deterministic results. Summarize them briefly."
    elif scenario == "tools_parallel":
        task = (
            "Read /probe/alpha.py and /probe/beta.py using two Read calls in the SAME response, "
            "since the reads are independent. After both results arrive, compare their functions and explain their difference briefly."
        )
    elif scenario == "thinking_parallel":
        task = (
            "Read /probe/alpha.py and /probe/beta.py using two Read calls in the SAME response, "
            "since the reads are independent. After both results arrive, call Glob with pattern /probe/*.py "
            "in a separate response to check for other variants. Then compare the functions and explain their difference briefly."
        )
    elif scenario == "thinking_tool":
        # The second tool round makes a later request re-read the first thinking block.
        task = (
            "Read /probe/main.py once. After its result arrives, call Glob with pattern /probe/*.py "
            "in a separate response to check for related files. Then reason about the function's behavior and give a brief explanation."
        )
    elif scenario == "thinking_followup":
        task = "Read /probe/main.py once. After its result arrives, reason about the function's behavior and give a brief explanation."
    else:
        task = (
            "First Read /probe/manifest.json. Wait for its result before choosing the entrypoint. "
            "Then Read ONLY the entrypoint path named in that result in a separate response. "
            "After that result arrives, analyze the function and explain it briefly."
        )
    if scenario in THINKING_SCENARIOS:
        # Adaptive thinking often skips reasoning before tool calls; ask for it explicitly.
        task += (
            " This task requires careful reasoning. Before EVERY tool call, think step by step about "
            "what you expect to find and why the call is needed. After each tool result arrives, "
            "reflect on what it shows before deciding the next step."
        )
    return [{"role": "user", "content": [{"type": "text", "text": project}, {"type": "text", "text": task}]}]


def _followup(scenario: str, turn: int, turns: int) -> list[dict[str, Any]] | None:
    """The next plain user message after a completed turn, or None when the scenario ends."""
    if scenario == "text_multiturn" and turn < turns:
        return [
            {"type": "text", "text": f"<system-reminder>Continue the same project; keep the earlier requirements. Turn {turn + 1}.</system-reminder>"},
            {"type": "text", "text": "Without tools, propose the next implementation step consistent with those requirements. Keep it brief."},
        ]
    if scenario == "thinking_followup" and turn == 1:
        return [{"type": "text", "text": "Without any further tools, explain how the function handles negative values, using what you read."}]
    return None


def _tool_results(content: list[dict[str, Any]], padding: int) -> tuple[list[dict[str, Any]], list[str]]:
    paths = []
    results = []
    seen_ids = set()
    source = "def total(values):\n    return sum(value * value for value in values if value >= 0)\n"
    source += "".join(f"# fixture row {i}: stable repository context for reading and analysis\n" for i in range(padding))
    files = {
        "/probe/manifest.json": '{"entrypoint": "/probe/main.py"}',
        "/probe/main.py": source,
        "/probe/alpha.py": source,
        "/probe/beta.py": source.replace("value >= 0", "value < 0"),
    }
    for block in content:
        if block.get("type") != "tool_use":
            continue
        tool_id = block.get("id")
        args = block.get("input")
        if not isinstance(tool_id, str) or not tool_id or tool_id in seen_ids or not isinstance(args, dict):
            raise ValueError("Invalid/duplicate tool_use ID or input")
        seen_ids.add(tool_id)
        if block.get("name") == "Read" and isinstance(args.get("file_path"), str) and args["file_path"] in files:
            path = args["file_path"]
            paths.append(path)
            result = files[path]
        elif block.get("name") == "Glob" and isinstance(args.get("pattern"), str):
            result = "\n".join(path for path in files if fnmatch.fnmatch(path, args["pattern"]))
            paths.append(f"Glob:{args['pattern']}")
        else:
            raise ValueError("Unexpected tool or virtual file; requested scenario did not occur")
        results.append({"type": "tool_result", "tool_use_id": tool_id, "content": result})
    return results, paths


def _flow_reasons(scenario: str, requests: list[dict[str, Any]], tool_batches: list[list[str]], turns: int) -> list[str]:
    reasons = []
    if any("error" in request for request in requests):
        reasons.append("request_failed")
    if scenario == "text_multiturn":
        if tool_batches:
            reasons.append("unexpected_tool_use")
        if sum(request["stop_reason"] == "end_turn" for request in requests) < turns:
            reasons.append("missing_user_turns")
    elif scenario in {"tools_parallel", "thinking_parallel"}:
        if not tool_batches or sorted(tool_batches[0]) != ["/probe/alpha.py", "/probe/beta.py"]:
            reasons.append("parallel_reads_not_observed")
        if len(tool_batches) != (2 if scenario == "thinking_parallel" else 1):
            reasons.append("unexpected_tool_rounds")
    elif scenario == "thinking_followup":
        if tool_batches != [["/probe/main.py"]]:
            reasons.append("single_read_not_observed")
    elif scenario == "thinking_tool":
        if not tool_batches or tool_batches[0] != ["/probe/main.py"]:
            reasons.append("single_read_not_observed")
    elif tool_batches != [["/probe/manifest.json"], ["/probe/main.py"]]:
        reasons.append("dependent_reads_not_observed")
    if scenario in {"thinking_tool", "thinking_parallel"} and (
        len(tool_batches) != 2 or len(tool_batches[1]) != 1 or not tool_batches[1][0].startswith("Glob:")
    ):
        reasons.append("glob_after_reads_not_observed")
    if scenario in THINKING_SCENARIOS:
        # A thinking block is first sent as new input, then read from cache one
        # request later; the flow must reach that later request.
        if not any(request["thinking_in_expected_prefix"] and "error" not in request for request in requests):
            reasons.append("thinking_history_not_exercised")
        tool_rounds = [request for request in requests if request["tools"]]
        if not tool_rounds or tool_rounds[0]["thinking_blocks"] == 0:
            reasons.append("thinking_before_tool_not_observed")
        if scenario == "thinking_interleaved" and (len(tool_rounds) < 2 or tool_rounds[1]["thinking_blocks"] == 0):
            reasons.append("thinking_between_tools_not_observed")
        if scenario == "thinking_followup" and not any(
            request["turn"] > 1 and request["stop_reason"] == "end_turn" for request in requests
        ):
            reasons.append("new_user_turn_not_completed")
    if not requests or requests[-1]["stop_reason"] != "end_turn":
        reasons.append("agent_turn_not_completed")
    return reasons


def run_agent_suite(
    *, model: str, base_url: str, api_key: str, probe_text: str,
    scenarios: list[str], effort: str | None = "high",
    thinking: str = "adaptive", thinking_budget: int = 4096,
    max_tokens: int, ttl: str,
    stream: bool, pin_previous_message: bool, turns: int, max_requests: int,
    tool_output_lines: int, round_delay_ms: int, timeout: int,
    extra_headers: dict[str, str], dry_run: bool = False,
    client: httpx.Client | None = None,
    min_prefix_reuse: float = 0.95,
    record_dir: str | None = None,
) -> dict[str, Any]:
    if not 0 <= min_prefix_reuse <= 1:
        raise ValueError("min_prefix_reuse must be between 0 and 1")
    if thinking not in {"adaptive", "enabled", "off"}:
        raise ValueError("Unsupported thinking mode")
    if thinking == "off" and set(scenarios) & set(THINKING_SCENARIOS):
        raise ValueError("thinking scenarios require thinking to be enabled")
    if thinking == "enabled" and not 1024 <= thinking_budget < max_tokens:
        raise ValueError("thinking_budget must be >= 1024 and below max_tokens")
    # Keep the agent's thinking configuration stable across all flows.
    # Scenario names select behavioral checks, not the thinking configuration.
    thinking_config = {
        "adaptive": {"type": "adaptive"},
        "enabled": {"type": "enabled", "budget_tokens": thinking_budget},
        "off": None,
    }[thinking]
    results = {}
    plans = []
    headers = {"anthropic-version": "2023-06-01", "content-type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key
    headers.update(extra_headers)
    # No credential headers are included in reports or dry-run output.
    run_id = uuid.uuid4().hex
    url = f"{base_url.rstrip('/')}/v1/messages"
    owned_client = client is None and not dry_run
    session_dir = None
    if record_dir is not None and not dry_run:
        session_dir = Path(record_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}-{run_id[:8]}"
        session_dir.mkdir(parents=True, exist_ok=True)
    if owned_client:
        client = httpx.Client(timeout=timeout)
    try:
        for scenario in scenarios:
            # A run-unique system prefix keeps scenarios and runs from warming each other.
            system = [
                {"type": "text", "text": f"You are a coding assistant. Probe session: {run_id}/{scenario}."},
                {"type": "text", "text": "Use the virtual tools to complete the user's task. Follow dependencies and return a concise final answer."},
            ]
            messages = _initial_messages(scenario, probe_text)
            options = dict(
                model=model, system=system, tools=_tools(), ttl=ttl,
                thinking=thinking_config, effort=effort,
                max_tokens=max_tokens, stream=stream, pin_previous_message=pin_previous_message,
            )
            if dry_run:
                plans.append({"scenario": scenario, "initial_request": build_request(messages=messages, **options),
                              "anthropic_beta": headers.get("anthropic-beta")})
                continue
            assert client is not None
            requests: list[dict[str, Any]] = []
            tool_batches: list[list[str]] = []
            flow_errors: list[str] = []
            turn, after = 1, "prompt"
            for _ in range(max_requests):
                if requests and round_delay_ms > 0:
                    time.sleep(round_delay_ms / 1000)
                body = build_request(messages=messages, **options)
                trace = {} if session_dir is not None else None
                started_at = datetime.now(timezone.utc).isoformat()
                record, payload = _send(client, url, headers, body, trace)
                requests.append({"index": len(requests) + 1, "turn": turn, "after": after, **record})
                if session_dir is not None:
                    _record_exchange(session_dir / f"{scenario}.jsonl", requests[-1], started_at, url, headers, body, trace)
                if payload is None:
                    break
                # Preserve all blocks and opaque thinking signatures, unchanged.
                content = deepcopy(payload["content"])
                if not content:
                    flow_errors.append("empty_assistant_content")
                    break
                if record["stop_reason"] not in {"tool_use", "end_turn"}:
                    flow_errors.append(f"unexpected_stop_reason:{record['stop_reason']}")
                    break
                messages.append({"role": "assistant", "content": content})
                try:
                    tool_results, paths = _tool_results(content, tool_output_lines)
                except ValueError as exc:
                    flow_errors.append(str(exc))
                    break
                if bool(tool_results) != (record["stop_reason"] == "tool_use"):
                    flow_errors.append("tool_use_stop_reason_mismatch")
                    break
                if tool_results:
                    tool_batches.append(paths)
                    requests[-1]["tool_paths"] = paths
                    # Every parallel result goes in one user message, before text.
                    messages.append({"role": "user", "content": tool_results})
                    after = "tool_result"
                    continue
                followup = _followup(scenario, turn, turns)
                if followup is None:
                    break
                messages.append({"role": "user", "content": followup})
                turn, after = turn + 1, "prompt"
            chain = classify_chain(requests, min_prefix_reuse)
            reasons = list(dict.fromkeys(_flow_reasons(scenario, requests, tool_batches, turns) + flow_errors))
            results[scenario] = {
                "name": scenario, "tool_batches": tool_batches,
                "scenario_completed": not reasons, "incomplete_reasons": reasons,
                **chain,
                # Strict pass needs both; cache_requirement_met alone is the cache verdict.
                "passed": not reasons and chain["cache_requirement_met"],
            }
    finally:
        if owned_client and client is not None:
            client.close()
    report = {
        "probe_mode": "agent", "model": model, "base_url": base_url,
        "reference_profile": REFERENCE, "reference_url": REFERENCE_URL,
        "run_id": run_id, "stream": stream, "ttl": ttl,
        "thinking": thinking_config, "effort": effort,
        "pin_previous_message": pin_previous_message,
        "min_prefix_reuse_target": min_prefix_reuse,
        "dry_run": dry_run, "plans": plans,
        "all_scenarios_passed": bool(results) and all(result["passed"] for result in results.values()),
        "all_cache_requirements_met": bool(results) and all(result["cache_requirement_met"] for result in results.values()),
        "scenarios": results,
        "record_dir": str(session_dir) if session_dir is not None else None,
    }
    if session_dir is not None:
        # The report holds hit labels per request index, matching the recorded lines.
        (session_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def print_report(report: dict[str, Any]) -> None:
    print(
        f"Model: {report['model']}  stream={report['stream']} ttl={report['ttl']} "
        f"thinking={(report['thinking'] or {}).get('type', 'off')} effort={report['effort'] or 'none'}"
    )
    print(f"HIT = read >= {report['min_prefix_reuse_target']:.0%} of expect (previous request's input + cache_read + cache_creation)")
    for result in report["scenarios"].values():
        print()
        print_chain(result)
        for request in result["requests"]:
            if "error" in request:
                print(f"  #{request['index']} error (status {request['status_code']}): {request['error'][:200]}")
        print("  flow=" + ("COMPLETE" if result["scenario_completed"] else "INCOMPLETE (" + ", ".join(result["incomplete_reasons"]) + ")"))
    print("\nSummary")
    width = max([len(name) for name in report["scenarios"]] + [8])
    print(f"  {'scenario':<{width}}  {'cache':<18} {'flow':<10} {'reuse':>7}  timeline")
    for name, result in report["scenarios"].items():
        reuse = result["old_input_reuse_ratio"]
        print(f"  {name:<{width}}  {result['verdict']:<18} {'COMPLETE' if result['scenario_completed'] else 'INCOMPLETE':<10} "
              f"{(f'{reuse:.2%}' if reuse is not None else '-'):>7}  {result['hit_timeline']}")
    print("\n(t) = expected prefix already holds thinking blocks; ◀ = read fell vs previous request; --json for full per-request usage.")
    if report.get("record_dir"):
        print(f"Recorded requests/responses: {report['record_dir']}")
