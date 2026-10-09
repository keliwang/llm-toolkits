#!/usr/bin/env python3
"""检测 Anthropic 格式 API 的 prompt cache 运行时行为。

脚本面向官方接口和第三方兼容接口，核心目标是观测两类证据：
1. usage 中出现 cache_creation_input_tokens，说明观测到缓存写入
2. usage 中出现 cache_read_input_tokens，说明观测到缓存命中

默认使用真实 tool_use/tool_result 多轮请求探测，脚本同时报告请求接受度，
帮助区分“接口接受了 cache_control”与“本次已观测到缓存生效”。
--probe-mode agent 则逐场景模拟 Claude Code 风格的完整历史、thinking 和工具循环。
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import sys
import time
from typing import Any

SOURCE_PARAGRAPH = (
    "The Anthropic API supports prompt caching, which allows you to mark "
    "portions of the prompt for reuse. Cached content is stored ephemerally "
    "and can dramatically reduce latency and cost for long system prompts "
    "or repeated context. To use caching, add cache_control breakpoints to "
    "your content blocks. The minimum cacheable size varies by model and "
    "provider implementation. This paragraph is repeated to build a long, "
    "stable cacheable prefix for runtime probing. "
)
TOOL_NAME = "prompt_cache_probe_lookup"
TOOL_RESULT = {
    "source": "local_probe_tool",
    "value": "ok",
    "confidence": 1.0,
}


def _build_probe_text(repeat_count: int) -> str:
    return (SOURCE_PARAGRAPH * repeat_count).strip()


def _parse_header_values(values: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for raw_value in values:
        if ":" in raw_value:
            key, value = raw_value.split(":", 1)
        elif "=" in raw_value:
            key, value = raw_value.split("=", 1)
        else:
            raise ValueError(
                f"Invalid header format: {raw_value!r}. Use 'Name: Value' or 'Name=Value'."
            )
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid header key in: {raw_value!r}")
        headers[key] = value
    return headers


def _extract_numeric_value(payload: Any, candidate_keys: tuple[str, ...]) -> int | float | None:
    if isinstance(payload, dict):
        for key in candidate_keys:
            value = payload.get(key)
            if isinstance(value, (int, float)):
                return value
        for value in payload.values():
            found = _extract_numeric_value(value, candidate_keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _extract_numeric_value(item, candidate_keys)
            if found is not None:
                return found
    return None


def _extract_usage_metrics(body: dict[str, Any]) -> dict[str, int | float | None]:
    return {
        "cache_creation_input_tokens": _extract_numeric_value(
            body,
            (
                "cache_creation_input_tokens",
                "cacheCreationInputTokens",
                "cache_creation_tokens",
                "cacheCreationTokens",
            ),
        ),
        "cache_read_input_tokens": _extract_numeric_value(
            body,
            (
                "cache_read_input_tokens",
                "cacheReadInputTokens",
                "cache_read_tokens",
                "cacheReadTokens",
            ),
        ),
        "input_tokens": _extract_numeric_value(
            body,
            ("input_tokens", "inputTokens"),
        ),
        "output_tokens": _extract_numeric_value(
            body,
            ("output_tokens", "outputTokens"),
        ),
    }


def _build_system_blocks(probe_text: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "text",
            "text": probe_text,
            "cache_control": {"type": "ephemeral"},
        },
    ]


def _build_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": TOOL_NAME,
            "description": (
                "Return deterministic probe data for prompt cache runtime checks. "
                "Use this tool whenever the user asks for cache probe lookup data."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "probe_key": {
                        "type": "string",
                        "description": "Stable probe key supplied by the caller.",
                    },
                },
                "required": ["probe_key"],
                "additionalProperties": False,
            },
        },
    ]


def _build_repeat_body(model: str, probe_text: str) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": 64,
        "system": _build_system_blocks(probe_text),
        "messages": [{"role": "user", "content": "reply just 'ok'"}],
    }


def _build_tool_body(
    model: str,
    probe_text: str,
    messages: list[dict[str, Any]],
    forced_tool_name: str | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": 128,
        "system": _build_system_blocks(probe_text),
        "tools": _build_tools(),
        "messages": messages,
    }
    if forced_tool_name is not None:
        body["tool_choice"] = {"type": "tool", "name": forced_tool_name}
    return body


def _normalize_assistant_content(payload: dict[str, Any]) -> list[dict[str, Any]]:
    content = payload.get("content")
    if not isinstance(content, list):
        return []

    # Thinking signatures and redacted data are opaque and must round-trip intact.
    return deepcopy([block for block in content if isinstance(block, dict)])


def _find_tool_use(content: list[dict[str, Any]]) -> dict[str, Any] | None:
    for block in content:
        if block.get("type") == "tool_use" and block.get("name") == TOOL_NAME:
            return block
    return None


def _compute_cache_ratio(round_detail: dict[str, Any]) -> float | None:
    cache_read = round_detail.get("cache_read_input_tokens")
    cache_create = round_detail.get("cache_creation_input_tokens")
    input_tokens = round_detail.get("input_tokens")

    if not isinstance(cache_read, (int, float)) or cache_read <= 0:
        return None

    total_input = 0.0
    for value in (cache_read, cache_create, input_tokens):
        if isinstance(value, (int, float)) and value > 0:
            total_input += float(value)

    if total_input <= 0:
        return None

    return cache_read / total_input


def _pick_best_round(rounds: list[dict[str, Any]]) -> dict[str, Any] | None:
    ranked_rounds = sorted(
        rounds,
        key=lambda item: (
            item.get("cache_read_input_tokens") or 0,
            item.get("cache_creation_input_tokens") or 0,
            1 if 200 <= item.get("status_code", 0) < 300 else 0,
        ),
        reverse=True,
    )
    return ranked_rounds[0] if ranked_rounds else None


def _classify_strategy(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    accepted_rounds = sum(
        1 for round_detail in rounds if 200 <= round_detail.get("status_code", 0) < 300
    )
    cache_creation_detected = any(
        isinstance(round_detail.get("cache_creation_input_tokens"), (int, float))
        and round_detail["cache_creation_input_tokens"] > 0
        for round_detail in rounds
    )
    cache_read_detected = any(
        isinstance(round_detail.get("cache_read_input_tokens"), (int, float))
        and round_detail["cache_read_input_tokens"] > 0
        for round_detail in rounds[1:]
    )

    if cache_read_detected:
        detection_state = "cache_read_observed"
    elif cache_creation_detected:
        detection_state = "cache_write_observed"
    elif accepted_rounds > 0:
        detection_state = "cache_control_accepted"
    else:
        detection_state = "request_rejected"

    best_round = _pick_best_round(rounds)
    cache_ratio = _compute_cache_ratio(best_round) if best_round else None

    return {
        "accepted_rounds": accepted_rounds,
        "cache_creation_detected": cache_creation_detected,
        "cache_read_detected": cache_read_detected,
        "detection_state": detection_state,
        "cache_ratio": round(cache_ratio, 4) if cache_ratio is not None else None,
    }


def _send_probe_round(
    client: httpx.Client,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    response = client.post(url, headers=headers, json=body)
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)

    if response.status_code < 200 or response.status_code >= 300:
        return {
            "stage": stage,
            "status_code": response.status_code,
            "elapsed_ms": elapsed_ms,
            "error": response.text[:1000],
        }

    try:
        payload = response.json()
    except ValueError:
        payload = {}

    metrics = _extract_usage_metrics(payload)
    return {
        "stage": stage,
        "status_code": response.status_code,
        "elapsed_ms": elapsed_ms,
        **metrics,
        "_payload": payload,
    }


def _public_round_detail(round_detail: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in round_detail.items() if key != "_payload"}


def _run_repeat_strategy(
    client: httpx.Client,
    base_url: str,
    model: str,
    probe_text: str,
    base_headers: dict[str, str],
    extra_headers: dict[str, str],
    rounds: int,
    round_delay_ms: int,
) -> dict[str, Any]:
    body = _build_repeat_body(model=model, probe_text=probe_text)
    request_headers = {**base_headers, **extra_headers}
    request_url = f"{base_url.rstrip('/')}/v1/messages"

    round_details: list[dict[str, Any]] = []
    for round_index in range(rounds):
        round_details.append(
            _send_probe_round(
                client=client,
                url=request_url,
                headers=request_headers,
                body=body,
                stage="repeat",
            )
        )
        if round_index < rounds - 1 and round_delay_ms > 0:
            time.sleep(round_delay_ms / 1000)

    public_rounds = [_public_round_detail(round_detail) for round_detail in round_details]
    summary = _classify_strategy(public_rounds)
    return {
        "probe_mode": "repeat",
        "headers": extra_headers,
        "rounds": public_rounds,
        **summary,
    }


def _run_tool_strategy(
    client: httpx.Client,
    base_url: str,
    model: str,
    probe_text: str,
    base_headers: dict[str, str],
    extra_headers: dict[str, str],
    rounds: int,
    round_delay_ms: int,
    force_tool_choice: bool,
) -> dict[str, Any]:
    request_headers = {**base_headers, **extra_headers}
    request_url = f"{base_url.rstrip('/')}/v1/messages"
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"Call {TOOL_NAME} with probe_key='alpha'. "
                "After the tool result, answer exactly ok."
            ),
        }
    ]

    round_details: list[dict[str, Any]] = []
    expect_tool = True
    for round_index in range(rounds):
        stage = "tool_request" if expect_tool else "tool_result_followup"
        body = _build_tool_body(
            model=model,
            probe_text=probe_text,
            messages=messages,
            forced_tool_name=TOOL_NAME if expect_tool and force_tool_choice else None,
        )
        round_detail = _send_probe_round(
            client=client,
            url=request_url,
            headers=request_headers,
            body=body,
            stage=stage,
        )
        round_details.append(round_detail)

        if "error" in round_detail:
            break

        payload = round_detail.get("_payload")
        assistant_content = (
            _normalize_assistant_content(payload) if isinstance(payload, dict) else []
        )
        tool_use = _find_tool_use(assistant_content)

        public_detail = round_details[-1]
        public_detail["tool_use_observed"] = tool_use is not None

        if tool_use is not None:
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use["id"],
                            "content": json.dumps(TOOL_RESULT, ensure_ascii=False),
                        }
                    ],
                }
            )
            expect_tool = False
        else:
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})
            messages.append(
                {
                    "role": "user",
                    "content": "Continue the cache probe and answer exactly ok.",
                }
            )
            expect_tool = False

        if round_index < rounds - 1 and round_delay_ms > 0:
            time.sleep(round_delay_ms / 1000)

    public_rounds = [_public_round_detail(round_detail) for round_detail in round_details]
    summary = _classify_strategy(public_rounds)
    return {
        "probe_mode": "tool",
        "headers": extra_headers,
        "rounds": public_rounds,
        **summary,
    }


def _strategy_order(beta_mode: str) -> list[tuple[str, dict[str, str]]]:
    beta_headers = {"anthropic-beta": "prompt-caching-2024-07-31"}
    if beta_mode == "off":
        return [("standard", {})]
    if beta_mode == "on":
        return [("beta", beta_headers)]
    return [("standard", {}), ("beta", beta_headers)]


def _strategy_rank(strategy_result: dict[str, Any]) -> tuple[int, int, int]:
    return (
        2 if strategy_result.get("cache_read_detected") else 0,
        1 if strategy_result.get("cache_creation_detected") else 0,
        strategy_result.get("accepted_rounds", 0),
    )


def _probe_mode_order(probe_mode: str) -> list[str]:
    if probe_mode == "auto":
        return ["tool", "repeat"]
    return [probe_mode]


def _run_probe_strategy(
    probe_mode: str,
    client: httpx.Client,
    base_url: str,
    model: str,
    probe_text: str,
    base_headers: dict[str, str],
    strategy_headers: dict[str, str],
    rounds: int,
    round_delay_ms: int,
    force_tool_choice: bool,
) -> dict[str, Any]:
    if probe_mode == "tool":
        return _run_tool_strategy(
            client=client,
            base_url=base_url,
            model=model,
            probe_text=probe_text,
            base_headers=base_headers,
            extra_headers=strategy_headers,
            rounds=rounds,
            round_delay_ms=round_delay_ms,
            force_tool_choice=force_tool_choice,
        )
    return _run_repeat_strategy(
        client=client,
        base_url=base_url,
        model=model,
        probe_text=probe_text,
        base_headers=base_headers,
        extra_headers=strategy_headers,
        rounds=rounds,
        round_delay_ms=round_delay_ms,
    )


def check_cache_support(
    model: str,
    base_url: str,
    api_key: str,
    timeout: int = 60,
    beta_mode: str = "auto",
    repeat_count: int = 64,
    rounds: int = 2,
    round_delay_ms: int = 250,
    probe_mode: str = "auto",
    force_tool_choice: bool = False,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    if probe_mode not in {"auto", "tool", "repeat"}:
        raise ValueError("Use agent_cache_probe.run_agent_suite for the agent probe mode")
    import httpx

    base_headers = {
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    if api_key:
        base_headers["x-api-key"] = api_key
    if extra_headers:
        base_headers.update(extra_headers)

    probe_text = _build_probe_text(repeat_count)
    strategies: dict[str, Any] = {}

    with httpx.Client(timeout=timeout) as client:
        for current_probe_mode in _probe_mode_order(probe_mode):
            for strategy_name, strategy_headers in _strategy_order(beta_mode):
                result = _run_probe_strategy(
                    probe_mode=current_probe_mode,
                    client=client,
                    base_url=base_url,
                    model=model,
                    probe_text=probe_text,
                    base_headers=base_headers,
                    strategy_headers=strategy_headers,
                    rounds=rounds,
                    round_delay_ms=round_delay_ms,
                    force_tool_choice=force_tool_choice,
                )
                result_name = f"{current_probe_mode}/{strategy_name}"
                strategies[result_name] = result
                if strategy_name == "standard" and beta_mode == "auto":
                    if result["cache_read_detected"] or result["cache_creation_detected"]:
                        break
            if probe_mode == "auto" and any(
                strategy["cache_read_detected"] or strategy["cache_creation_detected"]
                for strategy in strategies.values()
            ):
                break

    selected_strategy_name, selected_strategy = max(
        strategies.items(),
        key=lambda item: _strategy_rank(item[1]),
    )

    prompt_cache_supported = (
        selected_strategy["cache_read_detected"] or selected_strategy["cache_creation_detected"]
    )
    cache_control_accepted = selected_strategy["accepted_rounds"] > 0

    if selected_strategy["cache_read_detected"]:
        detection_state = "cache_read_observed"
    elif selected_strategy["cache_creation_detected"]:
        detection_state = "cache_write_observed"
    elif cache_control_accepted:
        detection_state = "cache_control_accepted"
    else:
        detection_state = "request_rejected"

    return {
        "model": model,
        "base_url": base_url,
        "probe_text_characters": len(probe_text),
        "probe_repeat_count": repeat_count,
        "rounds": rounds,
        "round_delay_ms": round_delay_ms,
        "probe_mode": probe_mode,
        "force_tool_choice": force_tool_choice,
        "beta_mode": beta_mode,
        "selected_strategy": selected_strategy_name,
        "prompt_cache_supported": prompt_cache_supported,
        "cache_control_accepted": cache_control_accepted,
        "cache_creation_detected": selected_strategy["cache_creation_detected"],
        "cache_read_detected": selected_strategy["cache_read_detected"],
        "cache_ratio": selected_strategy["cache_ratio"],
        "detection_state": detection_state,
        "strategies": strategies,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="检测 Anthropic 格式 API 的 prompt cache 运行时行为"
    )
    parser.add_argument("model", help="模型 ID")
    parser.add_argument(
        "--base-url",
        default=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        help="API base URL，也可用 ANTHROPIC_BASE_URL 环境变量",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("ANTHROPIC_API_KEY", ""),
        help="API key，也可用 ANTHROPIC_API_KEY 环境变量",
    )
    parser.add_argument(
        "--beta-mode",
        choices=("auto", "off", "on"),
        default="auto",
        help="prompt caching beta header 策略: auto/off/on",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=64,
        help="探测文本重复次数，默认 64",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="每个策略发送的轮次，默认 2",
    )
    parser.add_argument(
        "--round-delay-ms",
        type=int,
        default=250,
        help="轮次之间的等待时间，单位毫秒，默认 250",
    )
    parser.add_argument(
        "--probe-mode",
        choices=("auto", "tool", "repeat", "agent"),
        default="auto",
        help="探测流程: auto/tool/repeat/agent；agent 逐场景模拟 Claude Code 多轮请求",
    )
    agent = parser.add_argument_group("Claude Code 风格 agent 场景 (--probe-mode agent)")
    from agent_cache_probe import SCENARIOS, THINKING_SCENARIOS

    agent.add_argument("--scenario", choices=SCENARIOS, action="append", help="选择场景，可重复；默认全部场景")
    agent.add_argument("--cache-strategy", choices=("claude-code", "system"), action="append", help="缓存策略，可重复对比；默认 claude-code（system + 消息末尾断点）")
    agent.add_argument("--effort", choices=("none", "low", "medium", "high", "xhigh", "max"), default="high", help="agent 请求的 output_config.effort，默认 high，整个场景保持不变；none 不发送 output_config")
    agent.add_argument("--thinking", choices=("adaptive", "enabled", "off"), default="adaptive", help="thinking 配置：adaptive（默认）、enabled（手动预算，适合不支持 adaptive 的模型）、off（不发送 thinking，默认跳过 thinking 场景）")
    agent.add_argument("--thinking-budget", type=int, default=4096, help="--thinking enabled 的 budget_tokens，默认 4096，需 >=1024 且小于 --max-tokens")
    agent.add_argument("--usage-accounting", choices=("auto", "anthropic", "implicit", "total"), default="auto", help="输入计数口径：auto 校准；anthropic=input+read+create；implicit=input+read；total=input 已包含全部输入")
    agent.add_argument("--min-prefix-reuse", type=float, default=0.95, help="每次实际推进的旧输入复用率验收目标，0~1，默认 0.95；旧输入包含静态上下文")
    agent.add_argument("--max-tokens", type=int, default=8192, help="agent 单请求输出上限，默认 8192")
    agent.add_argument("--cache-ttl", choices=("5m", "1h"), default="5m", help="统一断点 TTL，默认 5m（API key 模式）")
    agent.add_argument("--pin-previous-message", action="store_true", help="额外标记前一可缓存消息，模拟 Claude Code 可选的 fork cache pin")
    agent.add_argument("--agent-turns", type=int, default=3, help="普通对话用户轮数，默认 3")
    agent.add_argument("--max-agent-requests", type=int, default=8, help="每个场景主链路请求上限，不含校准和重放，默认 8")
    agent.add_argument("--tool-output-lines", type=int, default=128, help="虚拟源码工具结果的上下文行数，默认 128")
    agent.add_argument("--no-stream", action="store_true", help="关闭 agent 默认的 SSE 流式请求")
    agent.add_argument("--dry-run", action="store_true", help="打印场景计划和首轮请求体，不调用接口")
    agent.add_argument("--verbose", action="store_true", help="agent 报告输出每轮完整 usage 和估算明细")
    parser.add_argument(
        "--force-tool-choice",
        action="store_true",
        help="发送 tool_choice 强制模型调用探测工具，适用于支持该参数的接口",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="附加请求头，格式: 'Name: Value' 或 'Name=Value'，可重复传入",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="输出原始 JSON",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="请求超时秒数，默认 60",
    )
    args = parser.parse_args()

    if args.repeat_count < 1:
        parser.error("--repeat-count 需要大于等于 1")
    if args.rounds < 2:
        parser.error("--rounds 需要大于等于 2")
    if args.round_delay_ms < 0:
        parser.error("--round-delay-ms 需要大于等于 0")

    try:
        extra_headers = _parse_header_values(args.header)
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run and args.probe_mode != "agent":
        parser.error("--dry-run 需要 --probe-mode agent")
    if args.probe_mode == "agent":
        from agent_cache_probe import run_agent_suite, print_report

        if args.force_tool_choice:
            parser.error("agent 模式使用默认 tool_choice，不能搭配 --force-tool-choice")
        if args.agent_turns < 3 or args.max_agent_requests < args.agent_turns:
            parser.error("--agent-turns 至少为 3，--max-agent-requests 不能小于它")
        if args.max_tokens < 1:
            parser.error("--max-tokens 必须为正数")
        if args.tool_output_lines < 0:
            parser.error("--tool-output-lines 不能为负数")
        if not 0 <= args.min_prefix_reuse <= 1:
            parser.error("--min-prefix-reuse 必须在 0~1 之间")
        if args.thinking == "enabled" and not 1024 <= args.thinking_budget < args.max_tokens:
            parser.error("--thinking-budget 需 >=1024 且小于 --max-tokens")
        if args.thinking == "off" and set(args.scenario or ()) & set(THINKING_SCENARIOS):
            parser.error("--thinking off 不能搭配 thinking_* 场景")
        scenarios = list(dict.fromkeys(args.scenario or (
            [scenario for scenario in SCENARIOS if scenario not in THINKING_SCENARIOS]
            if args.thinking == "off" else SCENARIOS
        )))
        if args.beta_mode == "on":
            extra_headers["anthropic-beta"] = ",".join(filter(None, (extra_headers.get("anthropic-beta"), "prompt-caching-2024-07-31")))
        result = run_agent_suite(
            model=args.model, base_url=args.base_url, api_key=args.api_key,
            probe_text=_build_probe_text(args.repeat_count),
            scenarios=scenarios,
            cache_strategies=list(dict.fromkeys(args.cache_strategy or ["claude-code"])),
            effort=None if args.effort == "none" else args.effort,
            thinking=args.thinking, thinking_budget=args.thinking_budget,
            max_tokens=args.max_tokens, ttl=args.cache_ttl,
            stream=not args.no_stream, pin_previous_message=args.pin_previous_message,
            turns=args.agent_turns, max_requests=args.max_agent_requests,
            tool_output_lines=args.tool_output_lines, round_delay_ms=args.round_delay_ms,
            timeout=args.timeout, extra_headers=extra_headers, dry_run=args.dry_run,
            usage_accounting=args.usage_accounting, min_prefix_reuse=args.min_prefix_reuse,
        )
        if args.json_output or args.dry_run:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print_report(result, verbose=args.verbose)
        # Exit on cache verdicts; flow completeness is reported separately.
        sys.exit(0 if args.dry_run or result["all_cache_requirements_met"] else 1)

    result = check_cache_support(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        beta_mode=args.beta_mode,
        repeat_count=args.repeat_count,
        rounds=args.rounds,
        round_delay_ms=args.round_delay_ms,
        probe_mode=args.probe_mode,
        force_tool_choice=args.force_tool_choice,
        extra_headers=extra_headers,
    )

    if args.json_output:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        support_label = "✓ 观测到支持" if result["prompt_cache_supported"] else "· 本次未观测到支持"
        accepted_label = "✓ 已接受" if result["cache_control_accepted"] else "· 未接受"
        print(f"Model:                {result['model']}")
        print(f"Base URL:             {result['base_url']}")
        print(f"Detection State:      {result['detection_state']}")
        print(f"Prompt Cache:         {support_label}")
        print(f"Cache Control:        {accepted_label}")
        print(f"Selected Strategy:    {result['selected_strategy']}")
        print(f"Probe Text Chars:     {result['probe_text_characters']}")
        print(f"Probe Repeat Count:   {result['probe_repeat_count']}")
        print(f"Rounds Per Strategy:  {result['rounds']}")
        print(f"Round Delay Ms:       {result['round_delay_ms']}")
        print(f"Probe Mode:           {result['probe_mode']}")
        print(f"Force Tool Choice:    {result['force_tool_choice']}")
        print(f"Beta Mode:            {result['beta_mode']}")
        print(f"Cache Creation:       {'✓' if result['cache_creation_detected'] else '·'}")
        print(f"Cache Read:           {'✓' if result['cache_read_detected'] else '·'}")
        if result["cache_ratio"] is not None:
            print(f"Cache Ratio:          {result['cache_ratio']:.2%}")

        for strategy_name, strategy in result["strategies"].items():
            print("")
            print(f"[{strategy_name}] {strategy['detection_state']}")
            if strategy["headers"]:
                print(f"headers: {json.dumps(strategy['headers'], ensure_ascii=False)}")
            for index, round_detail in enumerate(strategy["rounds"], start=1):
                print(
                    f"round{index}: status={round_detail.get('status_code')} "
                    f"elapsed_ms={round_detail.get('elapsed_ms')} "
                    f"stage={round_detail.get('stage')}"
                )
                if "error" in round_detail:
                    print(f"  error={round_detail['error']}")
                    continue
                if "tool_use_observed" in round_detail:
                    print(f"  tool_use_observed={round_detail['tool_use_observed']}")
                print(
                    "  usage="
                    f"create={round_detail.get('cache_creation_input_tokens')} "
                    f"read={round_detail.get('cache_read_input_tokens')} "
                    f"input={round_detail.get('input_tokens')} "
                    f"output={round_detail.get('output_tokens')}"
                )

    sys.exit(0 if result["prompt_cache_supported"] else 1)


if __name__ == "__main__":
    main()
