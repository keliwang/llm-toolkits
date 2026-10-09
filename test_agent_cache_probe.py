"""Protocol regression checks with a synthetic local Messages endpoint (no API calls)."""

from copy import deepcopy
import json
import unittest

import httpx

from agent_cache_probe import SCENARIOS, _annotate_reuse, _input_usage, _read_sse, _static_reference, run_agent_suite


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
        scenario = body["system"][0]["text"].split("/")[-2]
        content = []
        calibration = len(body["messages"]) == 1 and body["messages"][0]["content"][0]["text"] == "Do not call tools. Reply exactly ok."
        reads = sum(block.get("type") == "tool_use" for message in body["messages"] for block in message["content"])
        followup = body["messages"][-1]["content"][0].get("type") == "tool_result"
        paths = []
        if not calibration and not self.premature and scenario != "text_multiturn":
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
        if not calibration and self.thinking and body.get("thinking", {"type": "disabled"})["type"] != "disabled":
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
        cache_strategies=["claude-code"],
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

        return run_fixture(handler, [scenario], **options)["scenarios"][f"{scenario}/claude-code"]

    def test_real_progression_misses_are_not_hidden_by_good_replay(self):
        result = self.run_token_trace("text_multiturn", [
            (0, 5907), (5120, 787), (5120, 859), (5888, 187), (5120, 1054), (6144, 30),
        ])
        self.assertFalse(result["passed"])
        # Progression reads never exceed the static reference; only replay does.
        self.assertEqual(result["reuse_verdict"], "REPLAY ONLY")
        self.assertEqual(result["hit_timeline"], ["PARTIAL", "STATIC", "STATIC"])
        self.assertEqual(result["first_non_hit"]["request"], 1)
        self.assertEqual(result["progression"]["old_input_miss_tokens_estimate"], 91 + 955)
        self.assertAlmostEqual(result["progression"]["min_old_input_reuse_ratio_estimate"], 5120 / 6075)
        self.assertAlmostEqual(result["verification_replay"]["aggregate_cache_ratio"], 6144 / 6174, places=4)
        followup = result["rounds"][3]["prefix_reuse"]
        self.assertEqual(followup["new_input_tokens_estimate"], 96)
        self.assertEqual(followup["old_history_reuse_ratio_lower_bound"], 0)
        self.assertNotIn("_prefix_snapshot", json.dumps(result))

    def test_history_hit_only_on_replay_cannot_pass(self):
        result = self.run_token_trace("tools_parallel", [
            (0, 5932), (5888, 44), (5888, 114), (5888, 1998), (7168, 718),
        ])
        self.assertFalse(result["passed"])
        self.assertFalse(result["history_read_beyond_static_observed"])
        self.assertTrue(result["verification_replay"]["history_read_beyond_static_observed"])
        self.assertEqual(result["reuse_verdict"], "REPLAY ONLY")
        # Most old input is static: high old-input reuse alone proves no history hit.
        self.assertGreater(result["old_input_reuse_ratio_estimate"], 0.95)
        self.assertEqual(result["rounds"][3]["prefix_reuse"]["old_history_reuse_ratio_lower_bound"], 0)

    def test_total_accounting_override_does_not_double_count_reads(self):
        trace = [(0, 5907), (5120, 5907), (5120, 5979), (5888, 6075), (5120, 6174), (6144, 6174)]
        unknown = self.run_token_trace("text_multiturn", trace)
        self.assertEqual(unknown["reuse_verdict"], "REUSE UNKNOWN")
        known = self.run_token_trace("text_multiturn", trace, usage_accounting="total")
        self.assertEqual(known["rounds"][3]["normalized_usage"]["total_input_tokens"], 6075)
        self.assertEqual(known["old_input_miss_tokens_estimate"], 1046)
        self.assertEqual(known["reuse_verdict"], "REPLAY ONLY")
        self.assertEqual(known["hit_timeline"], ["PARTIAL", "STATIC", "STATIC"])

    def test_rewritten_history_and_changed_config_cannot_estimate_reuse(self):
        fixture = MessagesFixture()
        run_fixture(fixture, ["thinking_tool"])
        before, after = fixture.requests[1:3]
        for change_config in (False, True):
            changed = deepcopy(after)
            if change_config:
                changed["output_config"]["effort"] = "low"
            else:
                changed["messages"][0]["content"][0]["text"] += " changed"
            from agent_cache_probe import _prefix_snapshot
            details = [
                {"stage": stage, "request_sha256": str(i), "_prefix_snapshot": _prefix_snapshot(body),
                 "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1000, "input_tokens": 100}
                for i, (stage, body) in enumerate((("initial_user_turn", before), ("tool_result_followup", changed)))
            ]
            _annotate_reuse(details, "anthropic", 900)
            reuse = details[1]["prefix_reuse"]
            self.assertFalse(reuse["prefix_preserved"])
            self.assertEqual(reuse["reason"], "request_configuration_or_history_changed")
            self.assertNotIn("old_input_reuse_ratio_estimate", reuse)

    def test_hit_labels_compare_expected_and_actual_reads(self):
        # Static 5907 is exact; each request should read the previous whole input.
        def detail(stage, read, creation, input_tokens, messages):
            body = {"system": [], "tools": [], "messages": [{"role": "user", "content": [{"type": "text", "text": m}]} for m in messages]}
            from agent_cache_probe import _prefix_snapshot
            return {"stage": stage, "request_sha256": stage + str(len(messages)), "_prefix_snapshot": _prefix_snapshot(body),
                    "cache_read_input_tokens": read, "cache_creation_input_tokens": creation, "input_tokens": input_tokens}
        details = [
            detail("initial_user_turn", 5907, 100, 3, ["a"]),
            detail("tool_result_followup", 6010, 500, 3, ["a", "b"]),
            detail("tool_result_followup", 5907, 1000, 3, ["a", "b", "c"]),
            detail("tool_result_followup", 6500, 1000, 3, ["a", "b", "c", "d"]),
            detail("tool_result_followup", 0, 9000, 3, ["a", "b", "c", "d", "e"]),
        ]
        _annotate_reuse(details, "anthropic", 5907)
        self.assertEqual([d["hit"]["label"] for d in details], ["HIT", "HIT", "STATIC", "PARTIAL", "MISS"])
        self.assertEqual(details[2]["hit"]["expected_read_tokens"], 6513)
        self.assertEqual(details[2]["hit"]["missed_tokens"], 606)

    def test_flow_incomplete_still_reports_cache_verdict_and_replays(self):
        report = run_fixture(MessagesFixture(thinking=False), ["thinking_interleaved"])
        result = report["scenarios"]["thinking_interleaved/claude-code"]
        self.assertFalse(result["scenario_completed"])
        self.assertEqual(result["reuse_verdict"], "PASS")
        self.assertTrue(result["cache_requirement_met"])
        self.assertFalse(result["passed"])
        self.assertTrue(report["all_cache_requirements_met"])
        self.assertFalse(report["all_scenarios_passed"])
        self.assertEqual(result["rounds"][-1]["stage"], "verification_replay")

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
            progression = [d for d in result["rounds"] if "hit" in d and d["stage"] != "verification_replay"]
            self.assertEqual([d["hit"]["expected_prefix_thinking_blocks"] for d in progression], [0, 0, 2])
            self.assertTrue(result["thinking_present_on_history_hit"])
            self.assertEqual(result["thinking_prefix_hits"], 1)
        self.assertEqual(report["scenarios"]["thinking_tool/claude-code"]["tool_batches"], [["/probe/main.py"], ["Glob:/probe/*.py"]])

    def test_thinking_only_in_new_input_is_not_a_thinking_hit(self):
        # Old two-request shape: thinking appears only as new input, never in a read prefix.
        fixture = MessagesFixture()

        def handler(request):
            body = json.loads(request.content)
            if body["messages"][-1]["content"][0].get("type") == "tool_result" and "/thinking_tool/" in body["system"][0]["text"]:
                payload = fixture(request).json()
                payload["content"] = [block for block in payload["content"] if block["type"] != "tool_use"] + [{"type": "text", "text": "ok"}]
                payload["stop_reason"] = "end_turn"
                return httpx.Response(200, json=payload)
            return fixture(request)

        result = run_fixture(handler, ["thinking_tool"])["scenarios"]["thinking_tool/claude-code"]
        self.assertIsNone(result["thinking_present_on_history_hit"])
        self.assertEqual(result["thinking_prefix_checks"], 0)
        self.assertIn("thinking_history_not_exercised", result["incomplete_reasons"])
        self.assertIn("glob_after_reads_not_observed", result["incomplete_reasons"])

    def test_read_drops_are_flagged(self):
        result = self.run_token_trace("tools_sequential", [
            (0, 5932), (5888, 44), (5888, 137), (6016, 106), (0, 8000), (7936, 64),
        ])
        self.assertEqual(result["read_drops"], [
            {"request": 3, "stage": "tool_result_followup", "kind": "zero", "previous_read": 6016, "read": 0, "delta": -6016},
        ])
        self.assertEqual(result["verification_replay"]["read_drops"], [])
        result = self.run_token_trace("tools_sequential", [
            (0, 5932), (5888, 44), (5888, 137), (6016, 106), (5888, 2112), (7936, 64),
        ])
        self.assertEqual(result["read_drops"][0]["kind"], "decrease")
        self.assertEqual(result["read_drops"][0]["delta"], -128)

    def test_contradictory_counters_cannot_establish_total_input(self):
        detail = {"cache_read_input_tokens": 200, "cache_creation_input_tokens": None, "input_tokens": 100}
        self.assertIsNone(_input_usage(detail, "total")["total_input_tokens"])
        self.assertEqual(_input_usage(detail, "implicit")["total_input_tokens"], 300)

    def test_all_scenarios_complete_and_reuse_history(self):
        fixture = MessagesFixture()
        report = run_fixture(fixture)
        self.assertTrue(report["all_scenarios_passed"])
        self.assertEqual(len(report["scenarios"]), len(SCENARIOS))
        for result in report["scenarios"].values():
            self.assertTrue(result["history_read_beyond_static_observed"])
            self.assertEqual(set(result["hit_timeline"]), {"HIT"})
            self.assertEqual(result["missed_tokens"], 0)
            self.assertEqual(result["rounds"][-1]["stage"], "verification_replay")
        prefixes = {body["system"][0]["text"] for body in fixture.requests}
        self.assertEqual(len(prefixes), len(SCENARIOS))

    def test_thinking_round_trip_parallel_results_and_replay(self):
        fixture = MessagesFixture()
        report = run_fixture(fixture, ["thinking_parallel"], pin_previous_message=True)
        self.assertTrue(report["all_scenarios_passed"])
        calibration, initial, followup, glob_followup, replay = fixture.requests
        self.assertEqual(glob_followup, replay)
        self.assertNotIn("tool_choice", initial)
        self.assertEqual(initial["tools"], followup["tools"])
        self.assertEqual(initial["system"], followup["system"])
        self.assertEqual(followup["messages"][1]["content"][:2], [
            {"type": "thinking", "thinking": "Synthetic fixture, not model-generated reasoning.", "signature": "fixture-signature-2"},
            {"type": "redacted_thinking", "data": "fixture-data-2"},
        ])
        self.assertEqual([block["tool_use_id"] for block in followup["messages"][-1]["content"]], ["tool-0", "tool-1"])
        # Both results precede any optional text; only the last result is marked.
        self.assertNotIn("cache_control", followup["messages"][-1]["content"][0])
        self.assertIn("cache_control", followup["messages"][-1]["content"][1])

    def test_interleaved_history_and_new_user_turn(self):
        fixture = MessagesFixture()
        report = run_fixture(fixture, ["thinking_interleaved", "thinking_followup"])
        interleaved = report["scenarios"]["thinking_interleaved/claude-code"]
        self.assertTrue(interleaved["passed"])
        self.assertEqual(interleaved["rounds"][3]["input_thinking_blocks"], 4)
        followup = report["scenarios"]["thinking_followup/claude-code"]
        self.assertTrue(followup["passed"])
        self.assertEqual(followup["rounds"][3]["stage"], "new_user_turn")
        self.assertEqual(followup["rounds"][3]["input_thinking_blocks"], 4)

    def test_system_only_hit_cannot_pass_history_requirement(self):
        report = run_fixture(MessagesFixture(static_only=True), ["thinking_tool"])
        result = report["scenarios"]["thinking_tool/claude-code"]
        self.assertTrue(result["scenario_completed"])
        self.assertTrue(result["cache_read_detected"])
        self.assertFalse(result["history_read_beyond_static_observed"])
        self.assertFalse(result["passed"])

    def test_system_control_does_not_claim_history_reuse(self):
        report = run_fixture(MessagesFixture(), ["tools_sequential"], cache_strategies=["system"])
        result = report["scenarios"]["tools_sequential/system"]
        self.assertTrue(result["passed"])
        self.assertFalse(result["history_read_beyond_static_observed"])

    def test_absent_thinking_or_tools_cannot_pass_on_cache_hit(self):
        report = run_fixture(MessagesFixture(thinking=False), ["thinking_interleaved"])
        result = report["scenarios"]["thinking_interleaved/claude-code"]
        self.assertTrue(result["cache_read_detected"])
        self.assertFalse(result["passed"])
        self.assertIn("thinking_between_tools_not_observed", result["incomplete_reasons"])
        report = run_fixture(MessagesFixture(premature=True), ["tools_parallel"])
        result = report["scenarios"]["tools_parallel/claude-code"]
        self.assertFalse(result["scenario_completed"])
        self.assertIn("parallel_reads_not_observed", result["incomplete_reasons"])

    def test_missing_usage_is_unknown_and_never_passes(self):
        report = run_fixture(MessagesFixture(usage=False), ["thinking_tool"])
        result = report["scenarios"]["thinking_tool/claude-code"]
        self.assertTrue(result["scenario_completed"])
        self.assertEqual(result["detection_state"], "usage_unavailable")
        self.assertIsNone(result["history_read_beyond_static_observed"])
        self.assertIsNone(result["aggregate_cache_ratio"])
        self.assertFalse(result["passed"])

    def test_implicit_cache_without_creation_metric_can_verify_history(self):
        fixture = MessagesFixture()

        def handler(request):
            payload = fixture(request).json()
            usage = payload["usage"]
            # Third-party implicit cache: all misses are input, no creation counter.
            usage["input_tokens"] += usage.pop("cache_creation_input_tokens")
            return httpx.Response(200, json=payload)

        report = run_fixture(handler, ["thinking_interleaved"])
        result = report["scenarios"]["thinking_interleaved/claude-code"]
        self.assertTrue(result["passed"])
        self.assertEqual(result["usage_accounting"], "implicit")
        self.assertEqual(result["static_reference_source"], "calibration_total_input_upper_bound")
        self.assertEqual(fixture.requests[0], fixture.requests[1])
        self.assertEqual(result["rounds"][1]["stage"], "static_calibration_replay")
        self.assertTrue(result["thinking_present_on_history_hit"])
        self.assertIsNone(result["usage_totals"]["cache_creation_input_tokens"])
        self.assertGreater(result["aggregate_cache_ratio"], 0)

    def test_partial_static_read_is_not_a_safe_history_baseline(self):
        cold = {"cache_read_input_tokens": 0, "cache_creation_input_tokens": None, "input_tokens": 5904}
        warm = {"cache_read_input_tokens": 5120, "cache_creation_input_tokens": None, "input_tokens": 784}
        reference, source, accounting = _static_reference(cold, warm)
        # 5888 > 5120 could still be entirely static: use the full 5904 bound.
        self.assertEqual(reference, 5904)
        self.assertEqual(accounting, "implicit")
        self.assertEqual(source, "calibration_total_input_upper_bound")

    def test_missing_creation_with_ambiguous_input_accounting_stays_unknown(self):
        # Input could exclude undisclosed creation, or include cache reads already.
        cold = {"cache_read_input_tokens": 0, "cache_creation_input_tokens": None, "input_tokens": 20}
        warm = {"cache_read_input_tokens": 5120, "cache_creation_input_tokens": None, "input_tokens": 20}
        self.assertEqual(_static_reference(cold, warm), (None, "unavailable", "unknown"))
        cold["input_tokens"] = warm["input_tokens"] = 5904
        self.assertEqual(_static_reference(cold, warm), (None, "unavailable", "unknown"))

    def test_failed_case_does_not_skip_later_cases(self):
        fixture = MessagesFixture()

        def handler(request):
            if "/text_multiturn/" in json.loads(request.content)["system"][0]["text"]:
                raise httpx.ConnectError("Synthetic connection failure", request=request)
            return fixture(request)

        report = run_fixture(handler, ["text_multiturn", "tools_sequential"])
        self.assertFalse(report["all_scenarios_passed"])
        self.assertIn("static_calibration_failed", report["scenarios"]["text_multiturn/claude-code"]["incomplete_reasons"])
        self.assertTrue(report["scenarios"]["tools_sequential/claude-code"]["passed"])

    def test_dry_run_makes_no_requests_and_shows_strategy(self):
        fixture = MessagesFixture()
        report = run_fixture(fixture, ["thinking_tool"], dry_run=True)
        self.assertEqual(fixture.requests, [])
        body = report["plans"][0]["initial_request"]
        self.assertEqual(body["thinking"], {"type": "adaptive"})
        self.assertIn("cache_control", body["messages"][-1]["content"][-1])
        self.assertNotIn("fixture-key", json.dumps(report))


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

        report = run_fixture(handler, stream=True)
        self.assertTrue(report["all_scenarios_passed"])
        self.assertTrue(report["scenarios"]["thinking_interleaved/claude-code"]["thinking_present_on_history_hit"])

    def test_incomplete_stream_and_in_band_error_are_not_success(self):
        with self.assertRaisesRegex(ValueError, "Incomplete SSE"):
            _read_sse(self.response([{"type": "message_start", "message": {"content": []}}]))
        with self.assertRaisesRegex(ValueError, "SSE error"):
            _read_sse(self.response([{"type": "error", "error": {"message": "fixture failure"}}]))


if __name__ == "__main__":
    unittest.main()
