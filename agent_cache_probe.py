"""Claude Code 风格的 Messages agent 缓存探测。

参考 Claude Code 2.1.295 的普通、完整历史请求路径：稳定的 tools/system，
会话开头的项目上下文，追加历史，最后一个可缓存消息上的显式断点。
前一消息断点是可选项（Claude Code 中由功能开关控制）。不复制其私有提示词、
订阅鉴权、服务端实验、thread 增量协议或工具搜索。所有工具均为本地虚拟数据。
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import fnmatch
import json
import time
from typing import Any
import uuid

import httpx

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
HIT_LABELS = ("HIT", "PARTIAL", "STATIC", "MISS", "UNKNOWN")
METRICS = (
    "cache_creation_input_tokens", "cache_read_input_tokens", "input_tokens", "output_tokens",
)
THINKING_TYPES = {"thinking", "redacted_thinking"}


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
    cache_strategy: str,
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
    if cache_strategy == "claude-code":
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
    markers = []
    for index, block in enumerate(body["system"]):
        if "cache_control" in block:
            markers.append(f"system[{index}]")
    for index, message in enumerate(body["messages"]):
        for block_index, block in enumerate(message["content"]):
            if "cache_control" in block:
                markers.append(f"messages[{index}].content[{block_index}]")
    return {
        "request_sha256": _digest(body),
        "system_sha256": _digest(body["system"]),
        "tools_sha256": _digest(body["tools"]),
        "message_count": len(body["messages"]),
        "input_thinking_blocks": sum(
            block.get("type") in THINKING_TYPES
            for message in body["messages"] for block in message["content"]
        ),
        "cache_breakpoints": markers,
    }


def _read_sse(response: httpx.Response) -> dict[str, Any]:
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


def _metrics(payload: dict[str, Any]) -> dict[str, int | float | None]:
    usage = payload.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    aliases = {
        "cache_creation_input_tokens": ("cache_creation_input_tokens", "cacheCreationInputTokens", "cache_creation_tokens", "cacheCreationTokens"),
        "cache_read_input_tokens": ("cache_read_input_tokens", "cacheReadInputTokens", "cache_read_tokens", "cacheReadTokens"),
        "input_tokens": ("input_tokens", "inputTokens"),
        "output_tokens": ("output_tokens", "outputTokens"),
    }
    return {
        metric: next((usage[key] for key in keys if type(usage.get(key)) in (int, float)), None)
        for metric, keys in aliases.items()
    }


def _input_usage(detail: dict[str, Any], accounting: str) -> dict[str, Any]:
    """Normalize documented counter semantics, never provider/model names."""
    creation, read, input_tokens = (detail.get(key) for key in METRICS[:3])
    valid = lambda value: type(value) in (int, float) and value >= 0
    total = None
    if valid(read) and valid(input_tokens):
        if accounting == "anthropic" and valid(creation):
            total = read + creation + input_tokens
        elif accounting == "implicit":
            total = read + input_tokens
        elif accounting == "total":
            total = input_tokens
    # Contradictory counters cannot establish token boundaries or cost exposure.
    if total is not None and (read > total or (valid(creation) and creation > total - read)):
        total = None
    return {
        "total_input_tokens": total,
        "cache_read_tokens": read if valid(read) else None,
        "not_read_tokens": total - read if total is not None else None,
        "cache_write_tokens": creation if valid(creation) else None,
    }


def _cache_ratio(detail: dict[str, Any], accounting: str = "anthropic") -> float | None:
    usage = _input_usage(detail, accounting)
    total = usage["total_input_tokens"]
    return round(usage["cache_read_tokens"] / total, 4) if total else None


def _prefix_snapshot(body: dict[str, Any]) -> dict[str, Any]:
    clean = deepcopy(body)
    # Strip only wire-level markers, never fields inside tool inputs or schemas.
    for block in clean["system"] + clean["tools"]:
        block.pop("cache_control", None)
    for message in clean["messages"]:
        for block in message["content"]:
            block.pop("cache_control", None)
    return clean


def _classify_hit(
    reuse: dict[str, Any], read: int | float | None,
    static_tokens: int | float | None, threshold: float,
) -> dict[str, Any]:
    """Compare what the previous breakpoint should make readable with what was read.

    The initial request can only read the static prefix warmed by calibration;
    later requests should read the whole previous input (its tail breakpoint).
    STATIC means reads never got past tools+system, so history was not proven.
    """
    initial = reuse["reason"] == "initial_request"
    expected = static_tokens if initial else reuse.get("previous_input_tokens")
    hit = {
        "label": "UNKNOWN", "expected_read_tokens": expected, "read_tokens": read, "missed_tokens": None,
        "expected_prefix_thinking_blocks": reuse.get("previous_input_thinking_blocks", 0),
    }
    if not initial and reuse["reason"] != "estimated":
        hit["reason"] = reuse["reason"]
    elif read is None or expected is None or static_tokens is None:
        hit["reason"] = "static_reference_or_usage_unavailable"
    else:
        hit["missed_tokens"] = max(expected - read, 0)
        if read <= 0:
            hit["label"] = "MISS"
        elif not initial and read <= static_tokens:
            hit["label"] = "STATIC"
        else:
            hit["label"] = "HIT" if read >= expected * threshold else "PARTIAL"
    return hit


def _annotate_reuse(
    rounds: list[dict[str, Any]], accounting: str, static_tokens: int | float | None,
    threshold: float = 0.95,
) -> None:
    previous = None
    for detail in rounds:
        detail["normalized_usage"] = _input_usage(detail, accounting)
        detail["cache_ratio"] = _cache_ratio(detail, accounting)
        if detail["stage"].startswith("static_calibration"):
            continue
        reuse = {"basis": "previous_request_input_prefix_estimate", "reason": "initial_request"}
        if previous is not None:
            reuse["previous_request_sha256"] = previous["request_sha256"]
            # Thinking blocks sent last request sit inside the prefix this one should read.
            reuse["previous_input_thinking_blocks"] = previous.get("input_thinking_blocks", 0)
            before, after = previous["_prefix_snapshot"], detail["_prefix_snapshot"]
            stable = {k: v for k, v in before.items() if k != "messages"} == {k: v for k, v in after.items() if k != "messages"}
            preserved = stable and after["messages"][:len(before["messages"])] == before["messages"]
            reuse["prefix_preserved"] = preserved
            old_total = previous["normalized_usage"]["total_input_tokens"]
            total = detail["normalized_usage"]["total_input_tokens"]
            read = detail["normalized_usage"]["cache_read_tokens"]
            if not preserved:
                reuse["reason"] = "request_configuration_or_history_changed"
            elif "error" in previous or "error" in detail:
                reuse["reason"] = "request_failed"
            elif old_total is None or total is None or old_total <= 0:
                reuse["reason"] = "input_accounting_unknown"
            elif total < old_total or (before == after and total != old_total):
                reuse["reason"] = "input_boundary_changed"
            else:
                reused = min(read, old_total)
                reuse.update(
                    reason="estimated", previous_input_tokens=old_total,
                    new_input_tokens_estimate=total - old_total,
                    old_input_read_tokens_estimate=reused,
                    old_input_miss_tokens_estimate=old_total - reused,
                    old_input_reuse_ratio_estimate=reused / old_total,
                )
                # Even with only an upper bound for static input, subtracting it
                # gives a conservative history coverage bound under prefix caching.
                reuse["old_history_reuse_ratio_lower_bound"] = (
                    max(0, reused - static_tokens) / (old_total - static_tokens)
                    if static_tokens is not None and old_total > static_tokens else None
                )
        detail["prefix_reuse"] = reuse
        # In a growing conversation reads should never shrink; flag any regression.
        read_now = None if "error" in detail else detail["normalized_usage"]["cache_read_tokens"]
        read_before = None if previous is None or "error" in previous else previous["normalized_usage"]["cache_read_tokens"]
        change = {"previous_read": read_before, "delta": None, "drop": None}
        if read_now is not None and read_before is not None:
            change["delta"] = read_now - read_before
            if read_now == 0 and read_before > 0:
                change["drop"] = "zero"
            elif read_now < read_before:
                change["drop"] = "decrease"
        detail["read_change"] = change
        detail["hit"] = _classify_hit(
            reuse, None if "error" in detail else detail["normalized_usage"]["cache_read_tokens"],
            static_tokens, threshold,
        )
        if detail["stage"] != "verification_replay":
            previous = detail


def _static_reference(
    calibration: dict[str, Any], replay: dict[str, Any] | None,
) -> tuple[int | float | None, str, str]:
    if "error" not in calibration and all(type(calibration.get(key)) in (int, float) for key in METRICS[:2]):
        cached = calibration[METRICS[0]] + calibration[METRICS[1]]
        if cached > 0:
            return cached, "explicit_static_prefix", "anthropic"
    if replay is not None and "error" not in calibration and "error" not in replay:
        cold_read, warm_read = (detail.get(METRICS[1]) for detail in (calibration, replay))
        cold_input, warm_input = (detail.get(METRICS[2]) for detail in (calibration, replay))
        counters = (cold_read, warm_read, cold_input, warm_input)
        if all(type(value) in (int, float) and value >= 0 for value in counters):
            cold_total, warm_total = cold_read + cold_input, warm_read + warm_input
            # Identical requests: more reads replace exactly as many input tokens.
            # This supports read+uncached-input accounting without assuming missing
            # creation=0. Use the WHOLE short request as an upper bound, rather than
            # a partially cached read (which could undercount the static prefix).
            if (
                warm_read > cold_read and cold_total == warm_total and cold_total > 0
                and all((detail.get(METRICS[0]) or 0) == 0 for detail in (calibration, replay))
            ):
                return cold_total, "calibration_total_input_upper_bound", "implicit"
    return None, "unavailable", "unknown"


def _send(
    client: httpx.Client, url: str, headers: dict[str, str], body: dict[str, Any], stage: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    detail: dict[str, Any] = {"stage": stage, "_prefix_snapshot": _prefix_snapshot(body), **_request_audit(body)}
    try:
        with client.stream("POST", url, headers=headers, json=body) as response:
            detail["status_code"] = response.status_code
            if not response.is_success:
                response.read()
                detail["error"] = response.text[:1000]
            else:
                if "text/event-stream" in response.headers.get("content-type", ""):
                    payload = _read_sse(response)
                else:
                    response.read()
                    payload = response.json()
                if not isinstance(payload, dict) or not isinstance(payload.get("content"), list):
                    raise ValueError("Response must be a Messages object with content blocks")
                if not all(isinstance(block, dict) for block in payload["content"]):
                    raise ValueError("Invalid response content block")
                detail.update(_metrics(payload))
                detail.update(
                    _payload=payload,
                    stop_reason=payload.get("stop_reason"),
                    response_block_types=[block.get("type") for block in payload["content"]],
                    thinking_blocks=sum(block.get("type") in THINKING_TYPES for block in payload["content"]),
                    tool_use_count=sum(block.get("type") == "tool_use" for block in payload["content"]),
                )
                detail["cache_ratio"] = _cache_ratio(detail)
    except (httpx.RequestError, ValueError, KeyError, TypeError) as exc:
        detail.setdefault("status_code", 0)
        detail["error"] = f"{type(exc).__name__}: {exc}"
    detail["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return detail


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


def _initial_messages(scenario: str) -> list[dict[str, Any]]:
    project = (
        "<system-reminder>\n# CLAUDE.md\nThis is a virtual Python repository. "
        "Use the Read tool to inspect files; never invent their contents. "
        "Working directory: /probe. Environment and project instructions remain fixed for this session.\n"
        "</system-reminder>"
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
    return [{"role": "user", "content": [{"type": "text", "text": project}, {"type": "text", "text": task}]}]


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


def _flow_reasons(scenario: str, rounds: list[dict[str, Any]], tool_batches: list[list[str]], turns: int) -> list[str]:
    reasons = []
    if any("error" in detail for detail in rounds):
        reasons.append("request_failed")
    if scenario == "text_multiturn":
        if tool_batches:
            reasons.append("unexpected_tool_use")
        if sum(detail.get("stop_reason") == "end_turn" for detail in rounds) < turns:
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
    if scenario.startswith("thinking"):
        # A thinking block is first sent as new input, then read from cache one
        # request later; the flow must reach that later request.
        if not any(
            previous.get("input_thinking_blocks", 0) > 0 and "error" not in current
            for previous, current in zip(rounds, rounds[1:])
        ):
            reasons.append("thinking_history_not_exercised")
        tool_rounds = [detail for detail in rounds if detail.get("tool_use_count", 0) > 0]
        if not tool_rounds or tool_rounds[0].get("thinking_blocks", 0) == 0:
            reasons.append("thinking_before_tool_not_observed")
        if scenario == "thinking_interleaved" and (
            len(tool_rounds) < 2 or tool_rounds[1].get("thinking_blocks", 0) == 0
        ):
            reasons.append("thinking_between_tools_not_observed")
        if scenario == "thinking_followup" and not any(
            detail["stage"] == "new_user_turn" and detail.get("stop_reason") == "end_turn"
            for detail in rounds
        ):
            reasons.append("new_user_turn_not_completed")
    if not rounds or rounds[-1].get("stop_reason") != "end_turn":
        reasons.append("agent_turn_not_completed")
    return reasons


def _phase_summary(evidence: list[dict[str, Any]], static_tokens: int | float | None, accounting: str) -> dict[str, Any]:
    read = any((detail.get("cache_read_input_tokens") or 0) > 0 for detail in evidence)
    write = any((detail.get("cache_creation_input_tokens") or 0) > 0 for detail in evidence)
    known_reads = [detail for detail in evidence if type(detail.get("cache_read_input_tokens")) in (int, float)]
    beyond_static = None if static_tokens is None or not known_reads else any(
        (detail.get("cache_read_input_tokens") or 0) > static_tokens for detail in evidence
    )
    if beyond_static is False and len(known_reads) != len(evidence):
        beyond_static = None
    # Only a HIT whose expected prefix already held thinking proves thinking was read.
    thinking_checks = [detail["hit"] for detail in evidence if detail["hit"]["expected_prefix_thinking_blocks"] > 0]
    if any(hit["label"] == "HIT" for hit in thinking_checks):
        thinking_hit = True
    elif not thinking_checks or any(hit["label"] == "UNKNOWN" for hit in thinking_checks):
        thinking_hit = None
    else:
        thinking_hit = False
    if read:
        state = "cache_read_observed"
    elif write:
        state = "cache_write_observed"
    elif any(detail.get("cache_read_input_tokens") is not None and detail.get("cache_creation_input_tokens") is not None for detail in evidence):
        state = "cache_miss_observed"
    elif evidence:
        state = "usage_unavailable"
    else:
        state = "request_failed"
    complete_usage = bool(evidence) and all("error" not in detail and _cache_ratio(detail, accounting) is not None for detail in evidence)
    totals = {
        key: sum(detail[key] for detail in evidence) if all(type(detail.get(key)) in (int, float) for detail in evidence) else None
        for key in METRICS
    }
    transitions = [detail["prefix_reuse"] for detail in evidence if detail["prefix_reuse"]["reason"] != "initial_request"]
    measured = [reuse for reuse in transitions if reuse["reason"] == "estimated"]
    complete_reuse = bool(transitions) and len(measured) == len(transitions)
    old_total = sum(reuse["previous_input_tokens"] for reuse in measured)
    old_read = sum(reuse["old_input_read_tokens_estimate"] for reuse in measured)
    hits = [detail["hit"] for detail in evidence]
    return {
        "read_drops": [
            {"request": index, "stage": detail["stage"], "kind": detail["read_change"]["drop"],
             "previous_read": detail["read_change"]["previous_read"], "read": detail["hit"]["read_tokens"],
             "delta": detail["read_change"]["delta"]}
            for index, detail in enumerate(evidence, start=1) if detail["read_change"]["drop"]
        ],
        "hit_timeline": [hit["label"] for hit in hits],
        "thinking_prefix_checks": len(thinking_checks),
        "thinking_prefix_hits": sum(hit["label"] == "HIT" for hit in thinking_checks),
        "hit_counts": {label: count for label in HIT_LABELS if (count := sum(hit["label"] == label for hit in hits))},
        "missed_tokens": sum(hit["missed_tokens"] for hit in hits) if hits and all(hit["missed_tokens"] is not None for hit in hits) else None,
        "first_non_hit": next(
            ({"request": index, "stage": detail["stage"], **detail["hit"]}
             for index, detail in enumerate(evidence, start=1) if detail["hit"]["label"] != "HIT"),
            None,
        ),
        "detection_state": state, "cache_read_detected": read, "cache_creation_detected": write,
        "static_cached_tokens": static_tokens,
        "history_read_beyond_static_observed": beyond_static,
        "thinking_present_on_history_hit": thinking_hit,
        "aggregate_cache_ratio": _cache_ratio(totals, accounting) if complete_usage else None,
        "usage_totals": totals if complete_usage else None,
        "reuse_transition_count": len(transitions), "measured_reuse_transition_count": len(measured),
        "old_input_reuse_ratio_estimate": old_read / old_total if complete_reuse else None,
        "min_old_input_reuse_ratio_estimate": min(reuse["old_input_reuse_ratio_estimate"] for reuse in measured) if complete_reuse else None,
        "old_input_miss_tokens_estimate": old_total - old_read if complete_reuse else None,
        "not_read_tokens": sum(detail["normalized_usage"]["not_read_tokens"] for detail in evidence) if complete_usage else None,
    }


def _summarize(
    rounds: list[dict[str, Any]], static_tokens: int | float | None,
    cache_strategy: str, accounting: str = "anthropic", min_prefix_reuse: float = 0.95,
) -> dict[str, Any]:
    progression = _phase_summary([
        detail for detail in rounds
        if not detail["stage"].startswith("static_calibration") and detail["stage"] != "verification_replay"
    ], static_tokens, accounting)
    replay = _phase_summary([detail for detail in rounds if detail["stage"] == "verification_replay"], static_tokens, accounting)
    if cache_strategy == "system":
        met = progression["cache_read_detected"]
        verdict = "STATIC HIT" if met else "CACHE NOT VERIFIED"
    else:
        # Every progression request must read what the previous breakpoint wrote.
        timeline = progression["hit_timeline"]
        transitions = timeline[1:]
        met = bool(transitions) and all(label == "HIT" for label in timeline)
        if met:
            verdict = "PASS"
        elif not transitions or "UNKNOWN" in timeline:
            verdict = "REUSE UNKNOWN"
        elif any(label in {"HIT", "PARTIAL"} for label in transitions):
            verdict = "PARTIAL REUSE"
        elif replay["history_read_beyond_static_observed"] is True:
            verdict = "REPLAY ONLY"
        else:
            verdict = "STATIC ONLY" if progression["cache_read_detected"] else "CACHE NOT VERIFIED"
    return {
        **progression, "progression": progression, "verification_replay": replay,
        "cache_requirement_met": met, "reuse_verdict": verdict,
        "usage_accounting": accounting, "min_prefix_reuse_target": min_prefix_reuse,
        "cache_ratio_basis": {
            "anthropic": "read / (read + creation + uncached input)",
            "implicit": "read / (read + input)", "total": "read / input", "unknown": "unknown",
        }[accounting],
        "hit_rule": f"HIT: read >= {min_prefix_reuse:g} x expected (initial: static prefix; later: previous request's whole input) and, after the initial request, read > static; STATIC: read <= static; MISS: read = 0",
        "thinking_hit_rule": "thinking_present_on_history_hit is True only when a request whose expected prefix already contains thinking blocks (sent as input by the previous request) is a HIT; None when no such request exists",
        "history_evidence_note": "Read beyond the static reference is some-history evidence. Old-input reuse assumes server-side contiguous prefix caching and unchanged tokenization. Previous assistant output first sent this round is NEW input. Aggregate usage cannot measure individual thinking/history blocks or actual billing.",
    }


def run_agent_suite(
    *, model: str, base_url: str, api_key: str, probe_text: str,
    scenarios: list[str], cache_strategies: list[str], effort: str | None = "high",
    thinking: str = "adaptive", thinking_budget: int = 4096,
    max_tokens: int, ttl: str,
    stream: bool, pin_previous_message: bool, turns: int, max_requests: int,
    tool_output_lines: int, round_delay_ms: int, timeout: int,
    extra_headers: dict[str, str], dry_run: bool = False,
    client: httpx.Client | None = None,
    usage_accounting: str = "auto", min_prefix_reuse: float = 0.95,
) -> dict[str, Any]:
    if usage_accounting not in {"auto", "anthropic", "implicit", "total"}:
        raise ValueError("Unsupported usage accounting")
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
    if owned_client:
        client = httpx.Client(timeout=timeout)
    try:
        for scenario in scenarios:
            for strategy in cache_strategies:
                request_headers = dict(headers)
                # Isolate cases and strategies before every cache breakpoint.
                system = [
                    {"type": "text", "text": f"You are a coding assistant. Probe session: {run_id}/{scenario}/{strategy}."},
                    {"type": "text", "text": "Use the virtual tools to complete the user's task. Follow dependencies and return a concise final answer.\n\n" + probe_text},
                ]
                tools = _tools()
                messages = _initial_messages(scenario)
                options = dict(
                    model=model, system=system, tools=tools, ttl=ttl,
                    thinking=thinking_config, effort=effort,
                    max_tokens=max_tokens, stream=stream, pin_previous_message=pin_previous_message,
                )
                first = build_request(messages=messages, cache_strategy=strategy, **options)
                if dry_run:
                    plans.append({"scenario": scenario, "cache_strategy": strategy, "initial_request": first,
                                  "anthropic_beta": request_headers.get("anthropic-beta"),
                                  "flow": "static calibration (replay if creation usage is absent) -> live agent loop -> exact final-request replay"})
                    continue
                assert client is not None
                details = []

                def send(body: dict[str, Any], stage: str) -> dict[str, Any]:
                    if details and round_delay_ms > 0:
                        time.sleep(round_delay_ms / 1000)
                    detail = _send(client, url, request_headers, body, stage)
                    details.append(detail)
                    return detail

                # Use a separate short exchange to measure only the static prefix.
                calibration_body = build_request(
                    messages=[{"role": "user", "content": "Do not call tools. Reply exactly ok."}],
                    cache_strategy="system", **options,
                )
                calibration = send(calibration_body, "static_calibration")
                static_tokens, reference_source, accounting = _static_reference(calibration, None)
                if static_tokens is None and "error" not in calibration:
                    calibration_replay = send(deepcopy(calibration_body), "static_calibration_replay")
                    static_tokens, reference_source, accounting = _static_reference(calibration, calibration_replay)
                if usage_accounting != "auto":
                    accounting = usage_accounting
                    if reference_source != "explicit_static_prefix":
                        total = _input_usage(calibration, accounting)["total_input_tokens"]
                        static_tokens = total
                        reference_source = "calibration_total_input_upper_bound" if total is not None else "unavailable"
                workflow_start = len(details)
                tool_batches = []
                completed_turns = 0
                stage = "initial_user_turn"
                last_body = None
                for _ in range(max_requests):
                    if "error" in calibration:
                        break
                    last_body = build_request(messages=messages, cache_strategy=strategy, **options)
                    detail = send(last_body, stage)
                    if "error" in detail:
                        break
                    payload = detail["_payload"]
                    # Preserve all blocks and opaque thinking signatures, unchanged.
                    content = deepcopy(payload["content"])
                    if not content:
                        detail["flow_error"] = "empty_assistant_content"
                        break
                    if detail["stop_reason"] not in {"tool_use", "end_turn"}:
                        detail["flow_error"] = f"unexpected_stop_reason:{detail['stop_reason']}"
                        break
                    messages.append({"role": "assistant", "content": content})
                    try:
                        tool_results, paths = _tool_results(content, tool_output_lines)
                    except ValueError as exc:
                        detail["flow_error"] = str(exc)
                        break
                    if bool(tool_results) != (detail["stop_reason"] == "tool_use"):
                        detail["flow_error"] = "tool_use_stop_reason_mismatch"
                        break
                    if tool_results:
                        tool_batches.append(paths)
                        detail["tool_paths"] = paths
                        # Every parallel result goes in one user message, before text.
                        messages.append({"role": "user", "content": tool_results})
                        stage = "tool_result_followup"
                    else:
                        completed_turns += 1
                        if scenario == "text_multiturn" and completed_turns < turns:
                            messages.append({"role": "user", "content": [
                                {"type": "text", "text": f"<system-reminder>Continue the same project; keep the earlier requirements. Turn {completed_turns + 1}.</system-reminder>"},
                                {"type": "text", "text": "Without tools, propose the next implementation step consistent with those requirements. Keep it brief."},
                            ]})
                            stage = "new_user_turn"
                        elif scenario == "thinking_followup" and completed_turns == 1:
                            messages.append({"role": "user", "content": [{"type": "text", "text": "Without any further tools, explain how the function handles negative values, using what you read."}]})
                            stage = "new_user_turn"
                        else:
                            break
                workflow = details[workflow_start:]
                reasons = _flow_reasons(scenario, workflow, tool_batches, turns)
                reasons.extend(detail["flow_error"] for detail in workflow if "flow_error" in detail)
                if "error" in calibration:
                    reasons.append("static_calibration_failed")
                # A replay verifies history that only became cacheable in the last
                # request. Its response never gets spliced into the primary history.
                # Flow problems do not block it: cache evidence stays independent.
                if last_body is not None and "error" not in details[-1]:
                    replay = send(deepcopy(last_body), "verification_replay")
                    if "error" in replay:
                        reasons.append("verification_replay_failed")
                _annotate_reuse(details, accounting, static_tokens, min_prefix_reuse)
                summary = _summarize(details, static_tokens, strategy, accounting, min_prefix_reuse)
                results[f"{scenario}/{strategy}"] = {
                    "scenario": scenario, "cache_strategy": strategy,
                    "anthropic_beta": request_headers.get("anthropic-beta"),
                    "thinking": thinking_config, "effort": effort, "tool_batches": tool_batches,
                    "static_reference_source": reference_source,
                    "scenario_completed": not reasons, "incomplete_reasons": list(dict.fromkeys(reasons)),
                    **summary,
                    # Strict pass needs both; cache_requirement_met alone is the cache verdict.
                    "passed": not reasons and summary["cache_requirement_met"],
                    "rounds": [{key: value for key, value in detail.items() if not key.startswith("_")} for detail in details],
                }
    finally:
        if owned_client and client is not None:
            client.close()
    return {
        "probe_mode": "agent", "model": model, "base_url": base_url,
        "reference_profile": REFERENCE, "reference_url": REFERENCE_URL,
        "run_id": run_id, "stream": stream, "ttl": ttl,
        "thinking": thinking_config, "effort": effort,
        "pin_previous_message": pin_previous_message,
        "usage_accounting_requested": usage_accounting, "min_prefix_reuse_target": min_prefix_reuse,
        "dry_run": dry_run, "plans": plans,
        "all_scenarios_passed": bool(results) and all(result["passed"] for result in results.values()),
        "all_cache_requirements_met": bool(results) and all(result["cache_requirement_met"] for result in results.values()),
        "scenarios": results,
    }


STAGE_LABELS = {
    "initial_user_turn": "initial", "tool_result_followup": "tool_result",
    "new_user_turn": "user_turn", "verification_replay": "replay",
}


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{value:g}" if isinstance(value, float) else str(value)


def _response_shape(detail: dict[str, Any]) -> str:
    parts = ["think"] if detail.get("thinking_blocks") else []
    tools = detail.get("tool_use_count", 0)
    parts.append(f"tool×{tools}" if tools else "end" if detail.get("stop_reason") == "end_turn" else _fmt(detail.get("stop_reason")))
    return ",".join(parts)


def print_report(report: dict[str, Any], verbose: bool = False) -> None:
    if verbose:
        _print_verbose_report(report)
        return
    print(
        f"Model: {report['model']}  stream={report['stream']} ttl={report['ttl']} "
        f"thinking={(report['thinking'] or {}).get('type', 'off')} effort={report['effort'] or 'none'}"
    )
    print(f"HIT = read >= {report['min_prefix_reuse_target']:.0%} of expect (tokens the previous request made cacheable)")
    alerts = []
    for name, result in report["scenarios"].items():
        flow = "COMPLETE" if result["scenario_completed"] else "INCOMPLETE(" + ",".join(result["incomplete_reasons"]) + ")"
        reuse, worst = result["old_input_reuse_ratio_estimate"], result["min_old_input_reuse_ratio_estimate"]
        reuse_text = f"{reuse:.2%} (min {worst:.2%})" if reuse is not None else "unknown"
        print(f"\n[{name}] cache={result['reuse_verdict']}  flow={flow}  reuse={reuse_text}  "
              f"static={_fmt(result['static_cached_tokens'])} ({result['usage_accounting']})")
        print(f"  {'#':>2}  {'stage':<11} {'resp':<12} {'input':>7} {'read':>7} {'Δread':>8} {'expect':>7} {'miss':>5}  hit")
        index = 0
        for detail in result["rounds"]:
            if detail["stage"].startswith("static_calibration"):
                if "error" in detail:
                    print(f"   c  {'calibration':<11} error: {detail['error'][:200]}")
                continue
            replay = detail["stage"] == "verification_replay"
            if not replay:
                index += 1
            number = "r" if replay else str(index)
            if "error" in detail:
                print(f"  {number:>2}  {STAGE_LABELS.get(detail['stage'], detail['stage']):<11} error: {detail['error'][:200]}")
                continue
            hit, change = detail["hit"], detail["read_change"]
            total = detail["normalized_usage"]["total_input_tokens"]
            delta = change["delta"]
            delta_text = "-" if delta is None else f"{'▼' if delta < 0 else '+'}{abs(delta)}" if delta else "0"
            label = hit["label"] + ("(t)" if hit["expected_prefix_thinking_blocks"] else "")
            note = {"zero": "  ◀ read fell to 0", "decrease": "  ◀ read fell"}.get(change["drop"], "")
            print(
                f"  {number:>2}  {STAGE_LABELS.get(detail['stage'], detail['stage']):<11} {_response_shape(detail):<12} "
                f"{_fmt(total if total is not None else detail.get('input_tokens')):>7} {_fmt(hit['read_tokens']):>7} "
                f"{delta_text:>8} {_fmt(hit['expected_read_tokens']):>7} {_fmt(hit['missed_tokens']):>5}  {label}{note}"
            )
            if change["drop"]:
                alerts.append(f"  {name} #{number} {STAGE_LABELS.get(detail['stage'], detail['stage'])}: "
                              f"{_fmt(change['previous_read'])} → {_fmt(hit['read_tokens'])}"
                              + (" (fell to 0)" if change["drop"] == "zero" else f" ({_fmt(delta)})"))
    print("\nSummary")
    width = max([len(name) for name in report["scenarios"]] + [8])
    print(f"  {'scenario':<{width}}  {'cache':<18} {'flow':<10} {'min reuse':>9}  hits")
    for name, result in report["scenarios"].items():
        worst = result["min_old_input_reuse_ratio_estimate"]
        print(f"  {name:<{width}}  {result['reuse_verdict']:<18} {'COMPLETE' if result['scenario_completed'] else 'INCOMPLETE':<10} "
              f"{(f'{worst:.2%}' if worst is not None else '-'):>9}  {' '.join(result['hit_timeline']) or '-'}")
    if alerts:
        print("\n⚠ Read drops (a later request read fewer cached tokens than the one before):")
        print("\n".join(alerts))
    else:
        print("\nRead drops: none")
    print("\n(t) = expected prefix already holds thinking blocks; ▼ = read fell vs previous request; --verbose for full usage detail.")


def _print_verbose_report(report: dict[str, Any]) -> None:
    def percent(value: float | None) -> str:
        return f"{value:.2%}" if value is not None else "unknown"

    print(f"Model: {report['model']}\nProfile: {report['reference_profile']}")
    print(f"Stream: {report['stream']}  TTL: {report['ttl']}  Previous message pin: {report['pin_previous_message']}")
    print(f"Thinking: {json.dumps(report['thinking'])}  Effort: {report['effort']}")
    print(f"HIT target per progression request: read >= {report['min_prefix_reuse_target']:.0%} of expected")
    for name, result in report["scenarios"].items():
        flow = "COMPLETE" if result["scenario_completed"] else "INCOMPLETE (" + ",".join(result["incomplete_reasons"]) + ")"
        print(f"\n[{name}] cache={result['reuse_verdict']}  flow={flow}")
        # (t) marks requests whose expected cached prefix already contains thinking blocks.
        timeline = " → ".join(
            detail["hit"]["label"] + ("(t)" if detail["hit"]["expected_prefix_thinking_blocks"] else "")
            for detail in result["rounds"]
            if "hit" in detail and detail["stage"] != "verification_replay"
        ) or "-"
        print(f"  hits: {timeline}  missed_tokens={result['missed_tokens']}  thinking_prefix_hits={result['thinking_prefix_hits']}/{result['thinking_prefix_checks']}")
        first = result["first_non_hit"]
        if first and result["cache_strategy"] != "system":
            print(f"  first non-HIT: request #{first['request']} {first['stage']} {first['label']} expected={first['expected_read_tokens']} read={first['read_tokens']}" + (f" ({first['reason']})" if "reason" in first else ""))
        print(f"  state={result['detection_state']} static_tokens={result['static_cached_tokens']} history_hit={result['history_read_beyond_static_observed']} thinking_on_history_hit={result['thinking_present_on_history_hit']}")
        print(f"  static_reference={result['static_reference_source']} usage_accounting={result['usage_accounting']}")
        progression, replay = result["progression"], result["verification_replay"]
        print(f"  progression: cache_ratio={percent(progression['aggregate_cache_ratio'])} old_input_reuse={percent(progression['old_input_reuse_ratio_estimate'])} worst={percent(progression['min_old_input_reuse_ratio_estimate'])} old_input_miss={progression['old_input_miss_tokens_estimate']} transitions={progression['measured_reuse_transition_count']}/{progression['reuse_transition_count']}")
        print(f"  replay (diagnostic): cache_ratio={percent(replay['aggregate_cache_ratio'])} old_input_reuse={percent(replay['old_input_reuse_ratio_estimate'])} history_hit={replay['history_read_beyond_static_observed']}")
        for index, detail in enumerate(result["rounds"], start=1):
            print(f"  request{index} {detail['stage']}: status={detail['status_code']} elapsed_ms={detail['elapsed_ms']} stop={detail.get('stop_reason')}")
            print(f"    read={detail.get(METRICS[1])} create={detail.get(METRICS[0])} input={detail.get(METRICS[2])} output={detail.get(METRICS[3])} ratio={detail.get('cache_ratio')} thinking_in={detail.get('input_thinking_blocks')} thinking_out={detail.get('thinking_blocks')} tools={detail.get('tool_use_count')}")
            normalized, reuse = detail["normalized_usage"], detail.get("prefix_reuse")
            print(f"    total_input={normalized['total_input_tokens']} not_read={normalized['not_read_tokens']}")
            if "hit" in detail:
                hit = detail["hit"]
                print(f"    hit={hit['label']} expected_read={hit['expected_read_tokens']} read={hit['read_tokens']} missed={hit['missed_tokens']} thinking_in_expected_prefix={hit['expected_prefix_thinking_blocks']}")
            if reuse:
                if reuse["reason"] == "estimated":
                    print(f"    old_input={reuse['previous_input_tokens']} old_input_reuse~={percent(reuse['old_input_reuse_ratio_estimate'])} old_input_miss~={reuse['old_input_miss_tokens_estimate']} new_input~={reuse['new_input_tokens_estimate']} history_reuse_lower_bound~={percent(reuse['old_history_reuse_ratio_lower_bound'])}")
                else:
                    print(f"    reuse={reuse['reason']}")
            if "error" in detail:
                print(f"    error={detail['error']}")
    print("\n(t) = the expected cached prefix already contains thinking blocks; only such HITs prove thinking was read from cache.")
    print("~ values assume unchanged server tokenization and contiguous prefix caching. Old input includes static context; new input includes the previous assistant output first sent this round.")
    print("Not-read tokens include possible cache writes; usage cannot isolate thinking-token hits or determine billing without provider prices. Calibration/replay are excluded from progression totals.")
