#!/usr/bin/env python3
"""检测 Anthropic 格式 API 的 prompt cache 运行时行为。

脚本面向官方接口和第三方兼容接口，核心目标是观测两类证据：
1. usage 中出现 cache_creation_input_tokens，说明观测到缓存写入
2. usage 中出现 cache_read_input_tokens，说明观测到缓存命中

脚本同时报告请求接受度，帮助区分“接口接受了 cache_control”与“本次已观测到缓存生效”。
"""

from __future__ import annotations

import argparse
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
) -> dict[str, Any]:
    started_at = time.perf_counter()
    response = client.post(url, headers=headers, json=body)
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 1)

    if response.status_code < 200 or response.status_code >= 300:
        return {
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
        "status_code": response.status_code,
        "elapsed_ms": elapsed_ms,
        **metrics,
    }


def _run_strategy(
    client: httpx.Client,
    base_url: str,
    model: str,
    probe_text: str,
    base_headers: dict[str, str],
    extra_headers: dict[str, str],
    rounds: int,
    round_delay_ms: int,
) -> dict[str, Any]:
    body = {
        "model": model,
        "max_tokens": 64,
        "system": [
            {
                "type": "text",
                "text": probe_text,
                "cache_control": {"type": "ephemeral"},
            },
        ],
        "messages": [{"role": "user", "content": "reply just 'ok'"}],
    }
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
            )
        )
        if round_index < rounds - 1 and round_delay_ms > 0:
            time.sleep(round_delay_ms / 1000)

    summary = _classify_strategy(round_details)
    return {
        "headers": extra_headers,
        "rounds": round_details,
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


def check_cache_support(
    model: str,
    base_url: str,
    api_key: str,
    timeout: int = 60,
    beta_mode: str = "auto",
    repeat_count: int = 64,
    rounds: int = 2,
    round_delay_ms: int = 250,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
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
        for strategy_name, strategy_headers in _strategy_order(beta_mode):
            result = _run_strategy(
                client=client,
                base_url=base_url,
                model=model,
                probe_text=probe_text,
                base_headers=base_headers,
                extra_headers=strategy_headers,
                rounds=rounds,
                round_delay_ms=round_delay_ms,
            )
            strategies[strategy_name] = result
            if strategy_name == "standard" and beta_mode == "auto":
                if result["cache_read_detected"] or result["cache_creation_detected"]:
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

    result = check_cache_support(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
        beta_mode=args.beta_mode,
        repeat_count=args.repeat_count,
        rounds=args.rounds,
        round_delay_ms=args.round_delay_ms,
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
                    f"elapsed_ms={round_detail.get('elapsed_ms')}"
                )
                if "error" in round_detail:
                    print(f"  error={round_detail['error']}")
                    continue
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
