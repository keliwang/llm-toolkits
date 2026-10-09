"""Drive a real Claude Code CLI session in an isolated workspace and analyze its session log for prompt-cache reuse."""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

THINKING_TYPES = {"thinking", "redacted_thinking"}
EDIT_TOOLS = {"Edit", "MultiEdit", "Write"}
SUBAGENT_TOOLS = {"Agent", "Task"}
UNIT_TEST_COMMAND = "python3 -m unittest discover -s tests"

# Session-linkage variables a parent Claude Code exports to its child processes; they must not leak into the probe.
PARENT_SESSION_ENV = (
    "CLAUDECODE",
    "CLAUDE_PID",
    "CLAUDE_EFFORT",
    "CLAUDE_CONFIG_DIR",
    "AI_AGENT",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_SESSION_ATTENDED",
    "CLAUDE_CODE_MESSAGING_SOCKET",
    "CLAUDE_CODE_MESSAGING_TOKEN",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_EXECPATH",
    "CLAUDE_CODE_SSE_PORT",
)

# Credentials are never inherited: only the explicit --base-url/--api-key/--auth-token reach the CLI.
CREDENTIAL_ENV = ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")

FIXTURE_FILES = {
    "PROBE.md": """# Ledger probe project

The entrypoint module is declared in `config/project.json` under the key `entrypoint`.
Tests live in `tests/` and run with `python3 -m unittest discover -s tests`.
""",
    "config/project.json": json.dumps({"name": "ledger", "entrypoint": "src/ledger/core.py", "tests": "tests"}, indent=2) + "\n",
    "config/rates.json": json.dumps({"USD": "1", "EUR": "0.925", "CHF": "0.883", "JPY": "1.4937"}, indent=2) + "\n",
    "config/rounding.json": json.dumps(
        {
            "mode": "half_up",
            "comment": "Round the converted minor-unit amount half-up to the nearest multiple of the currency step.",
            "steps": {"CHF": 5},
            "default_step": 1,
        },
        indent=2,
    )
    + "\n",
    "src/ledger/__init__.py": "",
    "src/ledger/core.py": '''import json
from decimal import Decimal
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load_rates() -> dict[str, Decimal]:
    raw = json.loads((CONFIG_DIR / "rates.json").read_text())
    return {currency: Decimal(rate) for currency, rate in raw.items()}


def load_rounding() -> dict:
    return json.loads((CONFIG_DIR / "rounding.json").read_text())


def convert(amount_minor: int, currency: str) -> int:
    """Convert USD minor units to `currency` minor units using config/rates.json and config/rounding.json."""
    rates = load_rates()
    rounding = load_rounding()
    value = Decimal(amount_minor) * rates[currency]
    step = rounding["steps"].get(currency, rounding["default_step"])
    return int(value) // step * step
''',
    "tests/test_core.py": '''import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ledger.core import convert


class ConvertTest(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(convert(1000, "EUR"), 925)

    def test_half_up(self):
        self.assertEqual(convert(1001, "EUR"), 926)

    def test_currency_step(self):
        self.assertEqual(convert(1000, "CHF"), 885)


if __name__ == "__main__":
    unittest.main()
''',
}

TURN_FIX = f"""You are working in a small Python repository. Do the following, in order:

1. Read PROBE.md first. It tells you where the entrypoint module is declared; read that declaration in a separate step before opening the module (do not guess the path).
2. The entrypoint depends on config/rates.json and config/rounding.json. They are independent: read both of them in parallel, in a single response.
3. Run `{UNIT_TEST_COMMAND}` to see the failures.
4. Think carefully about the rounding rules in config/rounding.json, then fix src/ledger/core.py with the Edit tool. Do not modify tests or config files.
5. Re-run the tests and finish with a one-paragraph summary."""

TURN_EXPLAIN = (
    "Without using any tools, explain in at most three sentences why the original rounding was wrong "
    "and how your fix handles the CHF step."
)

TURN_EXTEND = f"""Add a test named test_unknown_currency to tests/test_core.py asserting that convert(100, "XYZ") raises KeyError (change convert only if it does not already do so). Then run `{UNIT_TEST_COMMAND}` again and report the result."""

TURN_SUBAGENT = """Use your subagent tool (Agent/Task) with subagent_type "general-purpose" to delegate this to a subagent: read every file under src/ and config/ and report each file's path and line count. Relay the subagent's answer as a short list."""


def write_fixture(workspace: Path) -> None:
    for relative, content in FIXTURE_FILES.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def turn_plan(subagent: bool) -> list[dict[str, str]]:
    turns = [
        {"name": "fix", "prompt": TURN_FIX},
        {"name": "explain", "prompt": TURN_EXPLAIN},
        {"name": "extend", "prompt": TURN_EXTEND},
    ]
    if subagent:
        turns.append({"name": "subagent", "prompt": TURN_SUBAGENT})
    return turns


def build_command(args: argparse.Namespace, session_id: str) -> list[str]:
    tools = ["Read", "Glob", "Grep", "Edit", "Write", "Bash"]
    if args.subagent:
        tools.append("Agent")
    command = [
        args.claude_bin,
        "-p",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--verbose",
        "--session-id",
        session_id,
        "--tools",
        ",".join(tools),
        "--permission-mode",
        "acceptEdits",
        "--allowedTools",
        "Read,Glob,Grep,Edit,Write,Agent,Bash(python3 -m unittest:*)",
        "--strict-mcp-config",
    ]
    if args.model:
        command += ["--model", args.model]
    if args.effort != "none":
        command += ["--effort", args.effort]
    for extra in args.claude_arg or []:
        command.append(extra)
    return command


def build_env(args: argparse.Namespace, config_dir: Path) -> dict[str, str]:
    dropped = set(PARENT_SESSION_ENV) | set(CREDENTIAL_ENV)
    env = {key: value for key, value in os.environ.items() if key not in dropped}
    env["CLAUDE_CONFIG_DIR"] = str(config_dir)
    env["DISABLE_AUTOUPDATER"] = "1"
    env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    if args.base_url:
        env["ANTHROPIC_BASE_URL"] = args.base_url
    if args.api_key:
        env["ANTHROPIC_API_KEY"] = args.api_key
    if args.auth_token:
        env["ANTHROPIC_AUTH_TOKEN"] = args.auth_token
    return env


def _pump(stream: Any, sink: queue.Queue, log_file: Any) -> None:
    for line in stream:
        log_file.write(line)
        log_file.flush()
        sink.put(line)
    sink.put(None)


def run_session(
    command: list[str],
    env: dict[str, str],
    workspace: Path,
    turns: list[dict[str, str]],
    stream_log: Path,
    stderr_log: Path,
    turn_timeout: float,
) -> dict[str, Any]:
    """Send each turn over stream-json stdin and wait for its result event before sending the next one."""
    results: list[dict[str, Any]] = []
    init: dict[str, Any] | None = None
    error: str | None = None
    with stream_log.open("w") as out_log, stderr_log.open("w") as err_log:
        process = subprocess.Popen(
            command,
            cwd=workspace,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=err_log,
            text=True,
            bufsize=1,
        )
        lines: queue.Queue = queue.Queue()
        reader = threading.Thread(target=_pump, args=(process.stdout, lines, out_log), daemon=True)
        reader.start()
        try:
            for turn in turns:
                message = {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": turn["prompt"]}]}}
                process.stdin.write(json.dumps(message) + "\n")
                process.stdin.flush()
                started = time.monotonic()
                result = None
                while result is None:
                    remaining = turn_timeout - (time.monotonic() - started)
                    if remaining <= 0:
                        error = f"turn {turn['name']!r} timed out after {turn_timeout:.0f}s"
                        break
                    try:
                        line = lines.get(timeout=remaining)
                    except queue.Empty:
                        continue
                    if line is None:
                        error = f"claude exited during turn {turn['name']!r}"
                        break
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "system" and event.get("subtype") == "init":
                        init = event
                    elif event.get("type") == "result":
                        result = event
                if result is None:
                    break
                results.append(
                    {
                        "turn": turn["name"],
                        "subtype": result.get("subtype"),
                        "is_error": bool(result.get("is_error")),
                        "num_turns": result.get("num_turns"),
                        "duration_ms": result.get("duration_ms"),
                        "total_cost_usd": result.get("total_cost_usd"),
                    }
                )
                if result.get("is_error"):
                    detail = str(result.get("result") or result.get("subtype"))[:300]
                    error = f"turn {turn['name']!r} failed: {detail}"
                    if "login" in detail.lower():
                        error += " (isolated config has no login: pass --api-key or --auth-token)"
                    break
        except BrokenPipeError:
            error = "claude closed stdin unexpectedly"
        finally:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
            try:
                returncode = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait()
            reader.join(timeout=5)
    return {
        "returncode": returncode,
        "error": error,
        "turn_results": results,
        "tools": (init or {}).get("tools"),
        "model": (init or {}).get("model"),
        "claude_code_version": (init or {}).get("claude_code_version"),
    }


def find_session_logs(config_dir: Path, session_id: str) -> tuple[Path | None, list[Path]]:
    main = next(iter(sorted(config_dir.glob(f"projects/*/{session_id}.jsonl"))), None)
    if main is None:
        return None, []
    return main, sorted(main.parent.glob(f"{session_id}/subagents/*.jsonl"))


# ---------------------------------------------------------------------------
# Session log analysis


def _is_prompt(entry: dict[str, Any]) -> bool:
    """A user entry that starts a new user turn (not a tool result, injected reminder, or local command)."""
    if entry.get("type") != "user" or entry.get("isMeta") or entry.get("isCompactSummary"):
        return False
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, str):
        return not content.lstrip().startswith(("<local-command", "<command-name>"))
    if isinstance(content, list):
        return not any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)
    return False


def _usage_totals(usage: dict[str, Any] | None) -> dict[str, int | None]:
    if not isinstance(usage, dict) or not isinstance(usage.get("input_tokens"), (int, float)):
        return {"input": None, "read": None, "creation": None, "total": None, "output": None}
    uncached = int(usage["input_tokens"])
    read = int(usage.get("cache_read_input_tokens") or 0)
    creation = int(usage.get("cache_creation_input_tokens") or 0)
    output = usage.get("output_tokens")
    return {
        "input": uncached,
        "read": read,
        "creation": creation,
        "total": uncached + read + creation,
        "output": int(output) if isinstance(output, (int, float)) else None,
    }


def parse_requests(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse per-content-block assistant lines into one record per API response, in order.

    Claude Code writes one line per content block, and with parallel tool calls it interleaves the tool_result
    entries between blocks of the same response, so lines are grouped by message id across the whole log.
    """
    requests: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    turn = 0
    prompt_since_last = False
    previous_kind = None
    current: dict[str, Any] | None = None
    for index, entry in enumerate(entries):
        kind = entry.get("type")
        if kind == "user" and _is_prompt(entry):
            turn += 1
            prompt_since_last = True
        if kind in {"user", "assistant"}:
            previous_kind, kind_before = kind, previous_kind
        if kind != "assistant" or entry.get("isApiErrorMessage"):
            continue
        message = entry.get("message") or {}
        if message.get("model") == "<synthetic>":
            continue
        key = message.get("id") or entry.get("requestId")
        if key is None:
            # No id from the gateway: only consecutive assistant lines can be told apart as one response.
            key = current["key"] if current is not None and kind_before == "assistant" else f"line-{index}"
        current = by_key.get(key)
        if current is None:
            current = by_key[key] = {
                "key": key,
                "request_id": entry.get("requestId"),
                "model": message.get("model"),
                "turn": max(turn, 1),
                "after": "prompt" if prompt_since_last or not requests else "tool_result",
                "blocks": [],
                "tools": [],
                "stop_reason": None,
                "usage": None,
            }
            requests.append(current)
            prompt_since_last = False
        for block in message.get("content") or []:
            if not isinstance(block, dict):
                continue
            current["blocks"].append(block.get("type"))
            if block.get("type") == "tool_use":
                current["tools"].append(block.get("name"))
        if message.get("usage"):
            current["usage"] = message["usage"]
        if message.get("stop_reason"):
            current["stop_reason"] = message["stop_reason"]
    records = []
    for number, request in enumerate(requests, 1):
        record = {key: value for key, value in request.items() if key not in {"key", "blocks", "usage"}}
        record["index"] = number
        record["thinking_blocks"] = sum(1 for block in request["blocks"] if block in THINKING_TYPES)
        record.update(_usage_totals(request["usage"]))
        records.append(record)
    return records


def classify_chain(requests: list[dict[str, Any]], min_reuse: float) -> dict[str, Any]:
    """Expect each request to read at least the full input of the previous one (Claude Code marks the latest message)."""
    thinking_sent = 0  # thinking blocks already sent as input by the previous request
    previous: dict[str, Any] | None = None
    for request in requests:
        if previous is None:
            request.update(expect=None, miss=None, delta_read=None, hit="FIRST", read_drop=None)
        elif request["total"] is None or previous["total"] is None:
            request.update(expect=previous["total"], miss=None, delta_read=None, hit="UNKNOWN", read_drop=None)
        else:
            expect = previous["total"]
            read = request["read"]
            request["expect"] = expect
            request["miss"] = max(expect - read, 0)
            request["delta_read"] = None if previous["read"] is None else read - previous["read"]
            if read >= min_reuse * expect:
                hit = "HIT"
            elif read > 0:
                hit = "PARTIAL"
            else:
                hit = "MISS"
            request["hit"] = hit
            if previous["read"] and read == 0:
                request["read_drop"] = "zero"
            elif previous["read"] is not None and read < previous["read"]:
                request["read_drop"] = "decrease"
            else:
                request["read_drop"] = None
        request["thinking_in_expected_prefix"] = thinking_sent if previous is not None else 0
        if previous is not None:
            thinking_sent += previous["thinking_blocks"]
        previous = request

    progression = requests[1:]
    labels = [request["hit"] for request in progression]
    counts = {label: labels.count(label) for label in ("HIT", "PARTIAL", "MISS", "UNKNOWN")}
    known = [request for request in progression if request["hit"] != "UNKNOWN"]
    expected_sum = sum(request["expect"] for request in known)
    reused_sum = sum(min(request["read"], request["expect"]) for request in known)
    total_input = sum(request["total"] or 0 for request in requests)
    total_read = sum(request["read"] or 0 for request in requests)
    if not progression or counts["UNKNOWN"]:
        verdict = "REUSE UNKNOWN"
    elif counts["HIT"] == len(progression):
        verdict = "PASS"
    elif counts["HIT"] or counts["PARTIAL"]:
        verdict = "PARTIAL REUSE"
    else:
        verdict = "CACHE NOT VERIFIED"
    first_non_hit = next((request for request in progression if request["hit"] != "HIT"), None)
    thinking_checks = [request for request in progression if request["thinking_in_expected_prefix"]]
    return {
        "verdict": verdict,
        "cache_requirement_met": verdict == "PASS",
        "requests": requests,
        "hit_timeline": " → ".join(_hit_tag(request) for request in requests),
        "hit_counts": counts,
        "old_input_reuse_ratio": reused_sum / expected_sum if expected_sum else None,
        "missed_tokens": expected_sum - reused_sum,
        "total_input_tokens": total_input,
        "total_cache_read_tokens": total_read,
        "cache_ratio": total_read / total_input if total_input else None,
        "first_non_hit": None
        if first_non_hit is None
        else {key: first_non_hit[key] for key in ("index", "turn", "after", "expect", "read", "hit")},
        "thinking_prefix_hits": sum(1 for request in thinking_checks if request["hit"] == "HIT"),
        "thinking_prefix_checks": len(thinking_checks),
        "prompt_boundary_hits": sum(1 for request in progression if request["after"] == "prompt" and request["hit"] == "HIT"),
        "prompt_boundary_checks": sum(1 for request in progression if request["after"] == "prompt"),
        "read_drops": [{"index": request["index"], "kind": request["read_drop"]} for request in requests if request.get("read_drop")],
    }


def _hit_tag(request: dict[str, Any]) -> str:
    return request["hit"] + ("(t)" if request.get("thinking_in_expected_prefix") else "")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    entries = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def analyze_session(main_log: Path, subagent_logs: list[Path], min_reuse: float) -> dict[str, Any]:
    chains = [{"name": "main", "log": str(main_log), **classify_chain(parse_requests(load_jsonl(main_log)), min_reuse)}]
    for path in subagent_logs:
        chains.append({"name": f"subagent:{path.stem}", "log": str(path), **classify_chain(parse_requests(load_jsonl(path)), min_reuse)})
    return {"chains": chains, "all_cache_requirements_met": all(chain["cache_requirement_met"] for chain in chains)}


def flow_checks(
    main_requests: list[dict[str, Any]],
    subagent_chains: list[dict[str, Any]],
    workspace: Path,
    subagent: bool,
    require_thinking: bool,
) -> list[dict[str, Any]]:
    """Confirm the session actually exercised each pattern the probe is meant to measure."""

    def turn(number: int) -> list[dict[str, Any]]:
        return [request for request in main_requests if request["turn"] == number]

    fix, explain, extend = turn(1), turn(2), turn(3)
    read_rounds = [request for request in fix if "Read" in request["tools"]]
    progression = main_requests[1:]
    checks = [
        ("sequential_tool_rounds", len(read_rounds) >= 2, f"{len(read_rounds)} responses with Read in turn 1"),
        ("parallel_tool_calls", any(len(request["tools"]) >= 2 for request in fix), "a turn-1 response with ≥2 tool_use blocks"),
        ("bash_tool", any("Bash" in request["tools"] for request in fix), "unit tests run through Bash in turn 1"),
        ("edit_tool", any(EDIT_TOOLS & set(request["tools"]) for request in fix), "core.py edited in turn 1"),
        ("text_only_followup", bool(explain) and not any(request["tools"] for request in explain), f"{len(explain)} responses in turn 2, no tools"),
        ("tools_after_followup", any(request["tools"] for request in extend), "tool calls after a plain user turn (turn 3)"),
        ("prompt_boundary", any(request["after"] == "prompt" for request in progression), "a request that follows a new user prompt"),
    ]
    if require_thinking:
        checks.append(("thinking_observed", any(request["thinking_blocks"] for request in main_requests), "at least one thinking block"))
        checks.append(
            (
                "thinking_history_exercised",
                any(request["thinking_in_expected_prefix"] for request in progression),
                "a request whose cached prefix already contains thinking",
            )
        )
    if subagent:
        used = any(SUBAGENT_TOOLS & set(request["tools"]) for request in turn(4))
        logged = any(len(chain["requests"]) >= 2 for chain in subagent_chains)
        checks.append(("subagent", used and logged, "Agent tool used and a subagent log with ≥2 requests"))
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=60,
    )
    test_tail = (tests.stderr.strip().splitlines() or ["(no output)"])[-1]
    checks.append(("tests_pass_after_run", tests.returncode == 0, test_tail))
    checks.append(("new_test_added", "test_unknown_currency" in (workspace / "tests/test_core.py").read_text(), "test_unknown_currency present"))
    return [{"name": name, "ok": ok, "detail": detail} for name, ok, detail in checks]


# ---------------------------------------------------------------------------
# Reporting


def _fmt(value: Any) -> str:
    return "-" if value is None else str(value)


def _shape(request: dict[str, Any]) -> str:
    parts = ["think"] if request["thinking_blocks"] else []
    if request["tools"]:
        parts.append(f"tool×{len(request['tools'])}")
    else:
        parts.append(request["stop_reason"] or "?")
    return ",".join(parts)


def print_chain(chain: dict[str, Any]) -> None:
    ratio = chain["old_input_reuse_ratio"]
    reuse = "-" if ratio is None else f"{ratio:.2%}"
    overall = "-" if chain["cache_ratio"] is None else f"{chain['cache_ratio']:.2%}"
    print(f"[{chain['name']}] cache={chain['verdict']}  requests={len(chain['requests'])}  reuse={reuse}  overall_read={overall}  missed={chain['missed_tokens']}")
    print(f"  {'#':>3}  {'turn':>4}  {'after':<11} {'resp':<14} {'input':>7} {'read':>7} {'Δread':>7} {'expect':>7} {'miss':>6}  hit  tools")
    for request in chain["requests"]:
        delta = request["delta_read"]
        delta_text = "-" if delta is None else (f"▼{-delta}" if delta < 0 else f"+{delta}")
        drop = {"zero": "  ◀ read fell to 0", "decrease": "  ◀ read decreased"}.get(request.get("read_drop") or "", "")
        print(
            f"  {request['index']:>3}  {request['turn']:>4}  {request['after']:<11} {_shape(request):<14} "
            f"{_fmt(request['total']):>7} {_fmt(request['read']):>7} {delta_text:>7} {_fmt(request['expect']):>7} "
            f"{_fmt(request['miss']):>6}  {_hit_tag(request):<8} {','.join(request['tools'])}{drop}"
        )
    print(f"  timeline: {chain['hit_timeline']}")
    if chain["first_non_hit"]:
        first = chain["first_non_hit"]
        print(f"  first non-HIT: #{first['index']} turn {first['turn']} after {first['after']}: expect {first['expect']}, read {first['read']} ({first['hit']})")
    if chain["prompt_boundary_checks"]:
        print(f"  user-prompt boundary hits: {chain['prompt_boundary_hits']}/{chain['prompt_boundary_checks']}")
    if chain["thinking_prefix_checks"]:
        print(f"  thinking-prefix hits: {chain['thinking_prefix_hits']}/{chain['thinking_prefix_checks']}")


def print_report(report: dict[str, Any]) -> None:
    run = report.get("run")
    if run:
        print(f"Claude Code {_fmt(run.get('claude_code_version'))}  model={_fmt(run.get('model'))}  session={report['session_id']}")
        print(f"workspace: {report['workspace']}")
        print(f"session log: {report.get('session_log') or '(not found)'}")
        for result in run["turn_results"]:
            cost = result.get("total_cost_usd")
            cost_text = "" if cost is None else f"  cumulative_cost=${cost:.4f}"
            print(f"  turn {result['turn']:<9} {result['subtype']}  api_turns={_fmt(result['num_turns'])}  {_fmt(result['duration_ms'])}ms{cost_text}")
        if run.get("error"):
            print(f"  ✗ run error: {run['error']} (exit {run['returncode']}; see {report['stderr_log']})")
    if report.get("flow_checks") is not None:
        status = "COMPLETE" if report["flow_complete"] else "INCOMPLETE"
        print(f"flow={status}")
        for check in report["flow_checks"]:
            print(f"  {'✓' if check['ok'] else '✗'} {check['name']}: {check['detail']}")
    print()
    for chain in (report.get("analysis") or {}).get("chains", []):
        print_chain(chain)
        print()


# ---------------------------------------------------------------------------
# CLI


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve() if args.root else Path(tempfile.mkdtemp(prefix="cc-cache-probe-")).resolve()
    workspace = root / "workspace"
    if workspace.exists() and any(workspace.iterdir()):
        raise SystemExit(f"workspace {workspace} is not empty")
    workspace.mkdir(parents=True, exist_ok=True)
    write_fixture(workspace)
    config_dir = Path(args.config_dir).expanduser().resolve() if args.config_dir else root / "claude-config"
    config_dir.mkdir(parents=True, exist_ok=True)
    session_id = str(uuid.uuid4())
    command = build_command(args, session_id)
    env = build_env(args, config_dir)
    report: dict[str, Any] = {
        "session_id": session_id,
        "root": str(root),
        "workspace": str(workspace),
        "config_dir": str(config_dir),
        "stream_log": str(root / "stream.jsonl"),
        "stderr_log": str(root / "stderr.log"),
        "command": command,
    }
    print(f"running Claude Code in {workspace} (session {session_id}) ...", file=sys.stderr)
    run = run_session(
        command,
        env,
        workspace,
        turn_plan(args.subagent),
        root / "stream.jsonl",
        root / "stderr.log",
        args.turn_timeout,
    )
    report["run"] = run
    main_log, subagent_logs = find_session_logs(config_dir, session_id)
    report["session_log"] = str(main_log) if main_log else None
    if main_log is None:
        report["analysis"] = None
        report["flow_checks"] = None
        report["flow_complete"] = False
        return report
    analysis = analyze_session(main_log, subagent_logs, args.min_prefix_reuse)
    report["analysis"] = analysis
    checks = flow_checks(analysis["chains"][0]["requests"], analysis["chains"][1:], workspace, args.subagent, not args.allow_no_thinking)
    report["flow_checks"] = checks
    report["flow_complete"] = run["error"] is None and all(check["ok"] for check in checks)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="在隔离 workspace 中驱动真实 Claude Code CLI 完成一个多轮任务，再分析其 session log 的缓存命中"
    )
    parser.add_argument("--model", help="传给 claude --model；不指定时使用 Claude Code 默认模型")
    parser.add_argument("--base-url", default=None, help="子进程 ANTHROPIC_BASE_URL；不继承当前环境，不指定时使用官方 API")
    parser.add_argument("--api-key", default=None, help="子进程 ANTHROPIC_API_KEY；真实运行时与 --auth-token 至少指定一个，不继承当前环境")
    parser.add_argument("--auth-token", default=None, help="子进程 ANTHROPIC_AUTH_TOKEN（Bearer 鉴权网关）；真实运行时与 --api-key 至少指定一个，不继承当前环境")
    parser.add_argument("--effort", choices=("none", "low", "medium", "high", "xhigh", "max"), default="high", help="claude --effort，默认 high；none 不传")
    parser.add_argument("--subagent", action="store_true", help="追加第 4 轮：通过 Agent 工具启动 general-purpose 子 agent，并分析其 sidechain 日志")
    parser.add_argument("--allow-no-thinking", action="store_true", help="模型不产生 thinking 时不视为流程未完成")
    parser.add_argument("--min-prefix-reuse", type=float, default=0.95, help="read ≥ 阈值 × 上一请求输入总量 记为 HIT，默认 0.95")
    parser.add_argument("--turn-timeout", type=float, default=600, help="每轮等待 result 事件的秒数，默认 600")
    parser.add_argument("--root", help="运行目录（含 workspace/、claude-config/、stream.jsonl），默认新建临时目录")
    parser.add_argument("--config-dir", help="CLAUDE_CONFIG_DIR；默认在运行目录下新建空配置，与本机 ~/.claude 隔离")
    parser.add_argument("--claude-bin", default=shutil.which("claude") or "claude", help="claude 可执行文件路径")
    parser.add_argument("--claude-arg", action="append", help="原样追加给 claude 的参数，可重复，例如 --claude-arg=--max-budget-usd=1")
    parser.add_argument("--analyze", metavar="SESSION_JSONL", help="只分析已有 session log（同目录下 <session>/subagents/*.jsonl 一并分析），不启动 Claude Code")
    parser.add_argument("--json", action="store_true", help="输出 JSON 报告")
    args = parser.parse_args()
    if not 0 < args.min_prefix_reuse <= 1:
        parser.error("--min-prefix-reuse must be in (0, 1]")
    if not args.analyze and not (args.api_key or args.auth_token):
        parser.error("real runs require an explicit --api-key or --auth-token (credentials are not inherited from the environment)")

    if args.analyze:
        main_log = Path(args.analyze).expanduser()
        subagent_logs = sorted(main_log.parent.glob(f"{main_log.stem}/subagents/*.jsonl"))
        report: dict[str, Any] = {"session_log": str(main_log), "analysis": analyze_session(main_log, subagent_logs, args.min_prefix_reuse)}
    else:
        report = run_probe(args)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)
    analysis = report.get("analysis")
    if analysis is None or (report.get("run") or {}).get("error"):
        sys.exit(2)
    sys.exit(0 if analysis["all_cache_requirements_met"] else 1)


if __name__ == "__main__":
    main()
