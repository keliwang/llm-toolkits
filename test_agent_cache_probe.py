"""Protocol regression checks with a synthetic local Messages endpoint (no API calls)."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import httpx

from agent_cache_probe import SCENARIOS, _read_sse, run_agent_suite


class MessagesFixture:
    """Return controlled tool/thinking turns and emulate exact prefix cache entries."""

    def __init__(self, *, thinking=True, static_only=False, usage=True, premature=False):
        self.requests = []
        self.cache = set()
        self.thinking = thinking
        self.static_only = static_only
        self.usage = usage
        self.premature = premature

    def __call__(self, request):
        body = json.loads(request.content)
        self.requests.append(deepcopy(body))
        scenario = body["system"][0]["text"].rstrip(".").split("/")[-1]
        content = []
        reads = sum(block.get("type") == "tool_use" for message in body["messages"] for block in message["content"])
        followup = body["messages"][-1]["content"][0].get("type") == "tool_result"
        paths = []
        if not self.premature and scenario != "text_multiturn":
            if reads == 0:
                if scenario in {"tools_parallel", "thinking_parallel"}:
                    paths = ["/probe/alpha.py", "/probe/beta.py"]
                elif scenario in {"thinking_tool", "thinking_followup"}:
                    paths = ["/probe/main.py"]
                else:
                    paths = ["/probe/manifest.json"]
            elif reads == 1 and followup and scenario in {"tools_sequential", "thinking_interleaved"}:
                paths = ["/probe/main.py"]
            elif followup and (scenario, reads) in {("thinking_tool", 1), ("thinking_parallel", 2)}:
                paths = ["Glob:/probe/*.py"]
        if self.thinking and body.get("thinking", {"type": "disabled"})["type"] != "disabled":
            content.extend([
                {"type": "thinking", "thinking": "Synthetic fixture, not model-generated reasoning.", "signature": f"fixture-signature-{len(self.requests)}"},
                {"type": "redacted_thinking", "data": f"fixture-data-{len(self.requests)}"},
            ])
        content.extend(
            {"type": "tool_use", "id": f"tool-{reads + index}", "name": "Glob", "input": {"pattern": path[5:]}}
            if path.startswith("Glob:") else
            {"type": "tool_use", "id": f"tool-{reads + index}", "name": "Read", "input": {"file_path": path}}
            for index, path in enumerate(paths)
        )
        if not paths:
            content.append({"type": "text", "text": "ok"})
        payload = {"type": "message", "role": "assistant", "content": content, "stop_reason": "tool_use" if paths else "end_turn"}
        if self.usage:
            prefix = json.dumps({"tools": body["tools"], "thinking": body.get("thinking")}, sort_keys=True)
            marked = []
            static_prefix = ""
            for block in body["system"]:
                clean = {key: value for key, value in block.items() if key != "cache_control"}
                prefix += json.dumps(clean, sort_keys=True)
                if "cache_control" in block:
                    marked.append(prefix)
                    static_prefix = prefix
            for message in body["messages"]:
                prefix += message["role"]
                for block in message["content"]:
                    clean = {key: value for key, value in block.items() if key != "cache_control"}
                    prefix += json.dumps(clean, sort_keys=True)
                    if "cache_control" in block:
                        marked.append(prefix)
            matches = [entry for entry in self.cache if prefix.startswith(entry)]
            cached = max(matches, key=len, default="")
            if self.static_only:
                cached = static_prefix if static_prefix in self.cache else ""
            read = len(cached) // 4
            written = len(marked[-1]) // 4 if marked else 0
            self.cache.update(marked)
            payload["usage"] = {
                "cache_read_input_tokens": read,
                "cache_creation_input_tokens": max(0, written - read),
                "input_tokens": max(0, len(prefix) // 4 - max(written, read)),
                "output_tokens": 32,
            }
        return httpx.Response(200, json=payload)


def run_fixture(fixture, scenarios=None, **overrides):
    options = dict(
        model="fixture-model", base_url="https://fixture.invalid", api_key="fixture-key",
        probe_text="Stable instructions. " * 400, scenarios=list(scenarios or SCENARIOS),
        max_tokens=8192, ttl="5m", stream=False, pin_previous_message=False,
        turns=3, max_requests=8, tool_output_lines=8, round_delay_ms=0, timeout=10,
        extra_headers={},
    )
    options.update(overrides)
    with httpx.Client(transport=httpx.MockTransport(fixture)) as client:
        return run_agent_suite(client=client, **options)


class AgentProtocolTests(unittest.TestCase):
    def run_token_trace(self, scenario, trace, **options):
        fixture = MessagesFixture()
        counters = iter(trace)

        def handler(request):
            payload = fixture(request).json()
            read, input_tokens = next(counters)
            payload["usage"] = {"cache_read_input_tokens": read, "input_tokens": input_tokens, "output_tokens": 32}
            return httpx.Response(200, json=payload)

        return run_fixture(handler, [scenario], **options)["scenarios"][scenario]

    def test_all_scenarios_run_once_and_every_request_hits(self):
        fixture = MessagesFixture()
        report = run_fixture(fixture)
        self.assertTrue(report["all_scenarios_passed"])
        self.assertEqual(len(report["scenarios"]), len(SCENARIOS))
        # One pass per scenario: no calibration or replay requests.
        self.assertEqual(len(fixture.requests), 20)
        for result in report["scenarios"].values():
            self.assertEqual(result["requests"][0]["hit"], "FIRST")
            self.assertEqual({request["hit"] for request in result["requests"][1:]}, {"HIT"})
            self.assertEqual(result["missed_tokens"], 0)
            self.assertEqual(result["verdict"], "PASS")
        prefixes = {body["system"][0]["text"] for body in fixture.requests}
        self.assertEqual(len(prefixes), len(SCENARIOS))

    def test_hit_labels_compare_with_previous_request_usage(self):
        result = self.run_token_trace("tools_sequential", [(0, 6000), (5990, 200), (3000, 3500)])
        requests = result["requests"]
        self.assertEqual([request["hit"] for request in requests], ["FIRST", "HIT", "PARTIAL"])
        self.assertEqual([request["expect"] for request in requests], [None, 6000, 6190])
        self.assertEqual(requests[2]["miss"], 3190)
        self.assertEqual(requests[2]["read_drop"], "decrease")
        self.assertEqual(result["verdict"], "PARTIAL REUSE")
        self.assertFalse(result["passed"])
        result = self.run_token_trace("tools_sequential", [(0, 6000), (0, 6200), (0, 6400)])
        self.assertEqual(result["verdict"], "CACHE NOT VERIFIED")
        self.assertEqual(result["hit_timeline"], "FIRST → MISS → MISS(t)")

    def test_read_drop_to_zero_is_flagged(self):
        result = self.run_token_trace("tools_sequential", [(0, 6000), (6000, 200), (0, 8000)])
        self.assertEqual(result["read_drops"], [{"index": 3, "kind": "zero"}])
        self.assertEqual(result["first_non_hit"]["index"], 3)

    def test_system_only_hit_cannot_pass(self):
        # Project context sits in the first user message, so tools+system alone
        # are far below the previous request's input.
        result = run_fixture(MessagesFixture(static_only=True), ["thinking_tool"])["scenarios"]["thinking_tool"]
        self.assertTrue(result["scenario_completed"])
        self.assertGreater(result["requests"][1]["read"], 0)
        self.assertEqual(result["requests"][1]["hit"], "PARTIAL")
        self.assertFalse(result["cache_requirement_met"])

    def test_flow_incomplete_still_reports_cache_verdict(self):
        report = run_fixture(MessagesFixture(thinking=False), ["thinking_interleaved"])
        result = report["scenarios"]["thinking_interleaved"]
        self.assertFalse(result["scenario_completed"])
        self.assertEqual(result["verdict"], "PASS")
        self.assertFalse(result["passed"])
        self.assertTrue(report["all_cache_requirements_met"])
        self.assertFalse(report["all_scenarios_passed"])

    def test_thinking_and_effort_can_be_omitted_or_budgeted(self):
        fixture = MessagesFixture()
        report = run_fixture(fixture, ["tools_sequential"], thinking="off", effort=None)
        self.assertTrue(report["all_scenarios_passed"])
        self.assertTrue(all("thinking" not in body and "output_config" not in body for body in fixture.requests))
        fixture = MessagesFixture()
        run_fixture(fixture, ["thinking_tool"], thinking="enabled", thinking_budget=2048)
        self.assertEqual(fixture.requests[-1]["thinking"], {"type": "enabled", "budget_tokens": 2048})
        with self.assertRaises(ValueError):
            run_fixture(MessagesFixture(), ["thinking_tool"], thinking="off")
        with self.assertRaises(ValueError):
            run_fixture(MessagesFixture(), ["thinking_tool"], thinking="enabled", thinking_budget=8192)

    def test_thinking_hit_requires_thinking_inside_expected_prefix(self):
        report = run_fixture(MessagesFixture(), ["thinking_tool", "thinking_parallel", "thinking_interleaved", "thinking_followup"])
        for result in report["scenarios"].values():
            self.assertTrue(result["scenario_completed"], result["incomplete_reasons"])
            # Thinking first sent in request 2 can only be read by request 3.
            self.assertEqual([request["thinking_in_expected_prefix"] for request in result["requests"]], [0, 0, 2])
            self.assertEqual((result["thinking_prefix_hits"], result["thinking_prefix_checks"]), (1, 1))
        self.assertEqual(report["scenarios"]["thinking_tool"]["tool_batches"], [["/probe/main.py"], ["Glob:/probe/*.py"]])

    def test_thinking_only_in_new_input_is_not_a_thinking_hit(self):
        fixture = MessagesFixture()

        def handler(request):
            body = json.loads(request.content)
            if body["messages"][-1]["content"][0].get("type") == "tool_result":
                payload = fixture(request).json()
                payload["content"] = [block for block in payload["content"] if block["type"] != "tool_use"] + [{"type": "text", "text": "ok"}]
                payload["stop_reason"] = "end_turn"
                return httpx.Response(200, json=payload)
            return fixture(request)

        result = run_fixture(handler, ["thinking_tool"])["scenarios"]["thinking_tool"]
        self.assertEqual(result["thinking_prefix_checks"], 0)
        self.assertIn("thinking_history_not_exercised", result["incomplete_reasons"])
        self.assertIn("glob_after_reads_not_observed", result["incomplete_reasons"])

    def test_thinking_round_trip_and_parallel_results(self):
        fixture = MessagesFixture()
        report = run_fixture(fixture, ["thinking_parallel"], pin_previous_message=True)
        self.assertTrue(report["all_scenarios_passed"])
        initial, followup, glob_followup = fixture.requests
        self.assertNotIn("tool_choice", initial)
        self.assertEqual(initial["tools"], glob_followup["tools"])
        self.assertEqual(initial["system"], followup["system"])
        self.assertEqual(followup["messages"][1]["content"][:2], [
            {"type": "thinking", "thinking": "Synthetic fixture, not model-generated reasoning.", "signature": "fixture-signature-1"},
            {"type": "redacted_thinking", "data": "fixture-data-1"},
        ])
        self.assertEqual([block["tool_use_id"] for block in followup["messages"][-1]["content"]], ["tool-0", "tool-1"])
        # Both results precede any optional text; only the last result is marked.
        self.assertNotIn("cache_control", followup["messages"][-1]["content"][0])
        self.assertIn("cache_control", followup["messages"][-1]["content"][1])

    def test_new_user_turn_is_a_prompt_boundary(self):
        result = run_fixture(MessagesFixture(), ["thinking_followup"])["scenarios"]["thinking_followup"]
        self.assertTrue(result["passed"])
        self.assertEqual([(request["turn"], request["after"]) for request in result["requests"]],
                         [(1, "prompt"), (1, "tool_result"), (2, "prompt")])
        self.assertEqual((result["prompt_boundary_hits"], result["prompt_boundary_checks"]), (1, 1))

    def test_absent_thinking_or_tools_cannot_pass_on_cache_hit(self):
        result = run_fixture(MessagesFixture(thinking=False), ["thinking_interleaved"])["scenarios"]["thinking_interleaved"]
        self.assertFalse(result["passed"])
        self.assertIn("thinking_between_tools_not_observed", result["incomplete_reasons"])
        result = run_fixture(MessagesFixture(premature=True), ["tools_parallel"])["scenarios"]["tools_parallel"]
        self.assertFalse(result["scenario_completed"])
        self.assertIn("parallel_reads_not_observed", result["incomplete_reasons"])

    def test_missing_usage_is_unknown_and_never_passes(self):
        result = run_fixture(MessagesFixture(usage=False), ["thinking_tool"])["scenarios"]["thinking_tool"]
        self.assertTrue(result["scenario_completed"])
        self.assertEqual(result["verdict"], "REUSE UNKNOWN")
        self.assertFalse(result["passed"])

    def test_implicit_cache_without_creation_metric_can_pass(self):
        fixture = MessagesFixture()

        def handler(request):
            payload = fixture(request).json()
            usage = payload["usage"]
            # Third-party implicit cache: all misses are input, no creation counter.
            usage["input_tokens"] += usage.pop("cache_creation_input_tokens")
            return httpx.Response(200, json=payload)

        result = run_fixture(handler, ["thinking_interleaved"])["scenarios"]["thinking_interleaved"]
        self.assertTrue(result["passed"])
        self.assertEqual(result["thinking_prefix_hits"], 1)

    def test_omitted_zero_and_aliased_usage_counters(self):
        fixture = MessagesFixture()

        def handler(request):
            payload = fixture(request).json()
            usage = payload["usage"]
            # Gateway reports only non-zero cache counters, in camelCase.
            payload["usage"] = {
                {"cache_read_input_tokens": "cacheReadInputTokens"}.get(key, key): value
                for key, value in usage.items() if value or not key.startswith("cache_")
            }
            return httpx.Response(200, json=payload)

        result = run_fixture(handler, ["thinking_interleaved"])["scenarios"]["thinking_interleaved"]
        self.assertTrue(result["passed"])
        self.assertNotIn("UNKNOWN", result["hit_timeline"])

    def test_failed_case_does_not_skip_later_cases(self):
        fixture = MessagesFixture()

        def handler(request):
            if "/text_multiturn." in json.loads(request.content)["system"][0]["text"]:
                raise httpx.ConnectError("Synthetic connection failure", request=request)
            return fixture(request)

        report = run_fixture(handler, ["text_multiturn", "tools_sequential"])
        failed = report["scenarios"]["text_multiturn"]
        self.assertFalse(report["all_scenarios_passed"])
        self.assertIn("request_failed", failed["incomplete_reasons"])
        self.assertIn("ConnectError", failed["requests"][0]["error"])
        self.assertTrue(report["scenarios"]["tools_sequential"]["passed"])

    def test_dry_run_makes_no_requests(self):
        fixture = MessagesFixture()
        report = run_fixture(fixture, ["thinking_tool"], dry_run=True)
        self.assertEqual(fixture.requests, [])
        body = report["plans"][0]["initial_request"]
        self.assertEqual(body["thinking"], {"type": "adaptive"})
        self.assertIn("cache_control", body["messages"][-1]["content"][-1])
        self.assertIn("Stable instructions.", body["messages"][0]["content"][0]["text"])
        self.assertNotIn("fixture-key", json.dumps(report))


    def test_record_writes_every_exchange_with_redacted_credentials(self):
        fixture = MessagesFixture()
        with tempfile.TemporaryDirectory() as directory:
            report = run_fixture(fixture, ["tools_parallel", "text_multiturn"], record_dir=directory,
                                 extra_headers={"Authorization": "Bearer secret-token", "X-Trace": "visible"})
            session = Path(report["record_dir"])
            self.assertEqual(session.parent, Path(directory))
            lines = [json.loads(line) for line in (session / "tools_parallel.jsonl").read_text().splitlines()]
            self.assertEqual(len(lines), len(report["scenarios"]["tools_parallel"]["requests"]))
            self.assertEqual([line["request"]["body"] for line in lines], fixture.requests[:len(lines)])
            headers = lines[0]["request"]["headers"]
            self.assertEqual((headers["x-api-key"], headers["Authorization"], headers["X-Trace"]), ("<redacted>", "<redacted>", "visible"))
            self.assertNotIn("secret-token", (session / "tools_parallel.jsonl").read_text())
            self.assertNotIn("fixture-key", (session / "tools_parallel.jsonl").read_text())
            first = lines[0]["response"]
            self.assertEqual(first["status_code"], 200)
            self.assertEqual(first["message"], json.loads(first["body"]))
            self.assertEqual([block["type"] for block in first["message"]["content"]][-2:], ["tool_use", "tool_use"])
            self.assertEqual(lines[1]["usage"]["read"], lines[0]["usage"]["total"])
            self.assertEqual(len((session / "text_multiturn.jsonl").read_text().splitlines()), 3)
            saved = json.loads((session / "report.json").read_text())
            self.assertEqual(saved["scenarios"]["tools_parallel"]["hit_timeline"], report["scenarios"]["tools_parallel"]["hit_timeline"])

    def test_record_keeps_failed_requests(self):
        def handler(request):
            if "/text_multiturn." in json.loads(request.content)["system"][0]["text"]:
                raise httpx.ConnectError("Synthetic connection failure", request=request)
            return httpx.Response(529, json={"type": "error", "error": {"type": "overloaded_error"}})

        with tempfile.TemporaryDirectory() as directory:
            report = run_fixture(handler, ["text_multiturn", "tools_sequential"], record_dir=directory)
            session = Path(report["record_dir"])
            failed = json.loads((session / "text_multiturn.jsonl").read_text())
            self.assertIn("ConnectError", failed["error"])
            self.assertEqual(failed["response"], {"status_code": 0, "body": "", "message": None})
            overloaded = json.loads((session / "tools_sequential.jsonl").read_text())
            self.assertEqual(overloaded["response"]["status_code"], 529)
            self.assertIn("overloaded_error", overloaded["response"]["body"])

    def test_without_record_nothing_is_written(self):
        report = run_fixture(MessagesFixture(), ["tools_sequential"])
        self.assertIsNone(report["record_dir"])

class StreamingProtocolTests(unittest.TestCase):
    @staticmethod
    def response(events):
        stream = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        return httpx.Response(200, text=stream, headers={"content-type": "text/event-stream"})

    def test_stream_reassembles_signatures_tool_input_and_usage(self):
        payload = _read_sse(self.response([
            {"type": "message_start", "message": {"content": [], "usage": {"cache_read_input_tokens": 100, "input_tokens": 20}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "fixture"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "opaque-"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "signature"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t1", "name": "Read", "input": {}}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"file_path":'}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '"/probe/main.py"}'}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 40}},
            {"type": "message_stop"},
        ]))
        self.assertEqual(payload["content"][0]["signature"], "opaque-signature")
        self.assertEqual(payload["content"][1]["input"], {"file_path": "/probe/main.py"})
        self.assertEqual(payload["usage"], {"cache_read_input_tokens": 100, "input_tokens": 20, "output_tokens": 40})

    def test_entire_suite_uses_streamed_history_without_losing_blocks(self):
        fixture = MessagesFixture()

        def handler(request):
            self.assertTrue(json.loads(request.content)["stream"])
            payload = fixture(request).json()
            usage = dict(payload["usage"])
            output_tokens = usage.pop("output_tokens")
            events = [{"type": "message_start", "message": {"content": [], "usage": usage}}]
            for index, original in enumerate(payload["content"]):
                block = deepcopy(original)
                if block["type"] == "tool_use":
                    block["input"] = {}
                elif block["type"] == "thinking":
                    block["thinking"] = ""
                    block["signature"] = ""
                events.append({"type": "content_block_start", "index": index, "content_block": block})
                if block["type"] == "tool_use":
                    events.append({"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": json.dumps(original["input"])}})
                elif block["type"] == "thinking":
                    for kind, field in [("thinking_delta", "thinking"), ("signature_delta", "signature")]:
                        events.append({"type": "content_block_delta", "index": index, "delta": {"type": kind, field: original[field]}})
                events.append({"type": "content_block_stop", "index": index})
            events.extend([
                {"type": "message_delta", "delta": {"stop_reason": payload["stop_reason"]}, "usage": {"output_tokens": output_tokens}},
                {"type": "message_stop"},
            ])
            return self.response(events)

        with tempfile.TemporaryDirectory() as directory:
            report = run_fixture(handler, stream=True, record_dir=directory)
            line = json.loads((Path(report["record_dir"]) / "thinking_tool.jsonl").read_text().splitlines()[0])
            # The raw SSE stream is kept next to the reassembled message.
            self.assertTrue(line["response"]["body"].startswith("data: "))
            self.assertIn("signature_delta", line["response"]["body"])
            self.assertRegex(line["response"]["message"]["content"][0]["signature"], r"^fixture-signature-\d+$")
        self.assertTrue(report["all_scenarios_passed"])
        self.assertEqual(report["scenarios"]["thinking_interleaved"]["thinking_prefix_hits"], 1)

    def test_incomplete_stream_and_in_band_error_are_not_success(self):
        with self.assertRaisesRegex(ValueError, "Incomplete SSE"):
            _read_sse(self.response([{"type": "message_start", "message": {"content": []}}]))
        with self.assertRaisesRegex(ValueError, "SSE error"):
            _read_sse(self.response([{"type": "error", "error": {"message": "fixture failure"}}]))


if __name__ == "__main__":
    unittest.main()
