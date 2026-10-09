import json
import tempfile
import unittest
from pathlib import Path

import claude_code_probe as probe


def prompt(text):
    return {"type": "user", "message": {"role": "user", "content": text}}


def tool_result(tool_id):
    return {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "ok"}]}}


def meta(text):
    return {"type": "user", "isMeta": True, "message": {"role": "user", "content": text}}


def assistant(message_id, block, uncached, read, creation, stop_reason=None):
    usage = {"input_tokens": uncached, "cache_read_input_tokens": read, "cache_creation_input_tokens": creation, "output_tokens": 10}
    return {
        "type": "assistant",
        "requestId": f"req_{message_id}",
        "message": {"id": message_id, "model": "m", "content": [block], "usage": usage, "stop_reason": stop_reason},
    }


def thinking():
    return {"type": "thinking", "thinking": "...", "signature": "sig"}


def tool_use(tool_id, name="Read"):
    return {"type": "tool_use", "id": tool_id, "name": name, "input": {}}


def text():
    return {"type": "text", "text": "done"}


class ParseRequestsTest(unittest.TestCase):
    def test_parallel_blocks_interleaved_with_results_form_one_request(self):
        entries = [
            prompt("go"),
            assistant("m1", thinking(), 2, 0, 1000),
            assistant("m1", tool_use("a"), 2, 0, 1000),
            tool_result("a"),
            assistant("m1", tool_use("b"), 2, 0, 1000, "tool_use"),
            tool_result("b"),
            assistant("m2", text(), 2, 1002, 300, "end_turn"),
        ]
        requests = probe.parse_requests(entries)
        self.assertEqual([request["tools"] for request in requests], [["Read", "Read"], []])
        self.assertEqual(requests[0]["thinking_blocks"], 1)
        self.assertEqual(requests[0]["stop_reason"], "tool_use")
        self.assertEqual(requests[1]["total"], 1304)

    def test_turns_and_boundaries_ignore_meta_and_synthetic(self):
        entries = [
            prompt("one"),
            meta("<system-reminder>ctx</system-reminder>"),
            assistant("m1", tool_use("a"), 2, 0, 100),
            tool_result("a"),
            assistant("m2", text(), 2, 102, 50, "end_turn"),
            prompt("two"),
            {"type": "assistant", "message": {"id": "s", "model": "<synthetic>", "content": [text()]}},
            {"type": "assistant", "isApiErrorMessage": True, "message": {"id": "e", "model": "m", "content": [text()]}},
            assistant("m3", text(), 2, 154, 20, "end_turn"),
        ]
        requests = probe.parse_requests(entries)
        self.assertEqual([(request["turn"], request["after"]) for request in requests], [(1, "prompt"), (1, "tool_result"), (2, "prompt")])

    def test_missing_cache_fields_count_as_zero_and_missing_usage_is_unknown(self):
        self.assertEqual(probe._usage_totals({"input_tokens": 7})["total"], 7)
        self.assertIsNone(probe._usage_totals({"output_tokens": 3})["total"])

    def test_without_ids_only_consecutive_lines_merge(self):
        entries = [prompt("go"), assistant(None, tool_use("a"), 2, 0, 10), assistant(None, tool_use("b"), 2, 0, 10), tool_result("a")]
        for entry in entries[1:3]:
            del entry["requestId"]
        entries.append(assistant(None, text(), 2, 12, 5))
        del entries[-1]["requestId"]
        self.assertEqual(len(probe.parse_requests(entries)), 2)


class ClassifyChainTest(unittest.TestCase):
    def chain(self, entries):
        return probe.classify_chain(probe.parse_requests(entries), 0.95)

    def test_full_reuse_passes_and_tracks_thinking_prefix(self):
        result = self.chain(
            [
                prompt("go"),
                assistant("m1", thinking(), 2, 500, 500),
                assistant("m1", tool_use("a"), 2, 500, 500, "tool_use"),
                tool_result("a"),
                assistant("m2", tool_use("b"), 2, 1000, 200, "tool_use"),
                tool_result("b"),
                assistant("m3", text(), 2, 1202, 100, "end_turn"),
            ]
        )
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["hit_timeline"], "FIRST → HIT → HIT(t)")
        self.assertEqual((result["thinking_prefix_hits"], result["thinking_prefix_checks"]), (1, 1))

    def test_partial_and_zero_reads_are_reported(self):
        result = self.chain(
            [
                prompt("go"),
                assistant("m1", tool_use("a"), 2, 0, 1000, "tool_use"),
                tool_result("a"),
                assistant("m2", tool_use("b"), 2, 500, 600, "tool_use"),
                tool_result("b"),
                prompt("again"),
                assistant("m3", text(), 1200, 0, 0, "end_turn"),
            ]
        )
        self.assertEqual(result["verdict"], "PARTIAL REUSE")
        self.assertEqual([request["hit"] for request in result["requests"]], ["FIRST", "PARTIAL", "MISS"])
        self.assertEqual(result["first_non_hit"]["index"], 2)
        self.assertEqual(result["read_drops"], [{"index": 3, "kind": "zero"}])
        self.assertEqual((result["prompt_boundary_hits"], result["prompt_boundary_checks"]), (0, 1))
        self.assertEqual(result["missed_tokens"], 502 + 1102)

    def test_single_request_is_unknown(self):
        self.assertEqual(self.chain([prompt("go"), assistant("m1", text(), 2, 0, 10, "end_turn")])["verdict"], "REUSE UNKNOWN")


class SessionFilesTest(unittest.TestCase):
    def test_finds_main_and_subagent_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp)
            project = config / "projects" / "-tmp-ws"
            (project / "sid" / "subagents").mkdir(parents=True)
            lines = [prompt("go"), assistant("m1", text(), 2, 0, 10, "end_turn")]
            (project / "sid.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n")
            (project / "sid" / "subagents" / "agent-x.jsonl").write_text(json.dumps(prompt("sub")) + "\n")
            main, subagents = probe.find_session_logs(config, "sid")
            self.assertEqual(main, project / "sid.jsonl")
            self.assertEqual([path.name for path in subagents], ["agent-x.jsonl"])
            analysis = probe.analyze_session(main, subagents, 0.95)
            self.assertEqual([chain["name"] for chain in analysis["chains"]], ["main", "subagent:agent-x"])


class FixtureTest(unittest.TestCase):
    def test_fixture_fails_until_rounding_is_fixed(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            probe.write_fixture(workspace)
            command = [sys.executable, "-m", "unittest", "discover", "-s", "tests"]
            self.assertNotEqual(subprocess.run(command, cwd=workspace, capture_output=True).returncode, 0)
            core = workspace / "src/ledger/core.py"
            core.write_text(
                core.read_text()
                .replace("from decimal import Decimal", "from decimal import ROUND_HALF_UP, Decimal")
                .replace(
                    "return int(value) // step * step",
                    "return int((value / step).quantize(Decimal(1), rounding=ROUND_HALF_UP)) * step",
                )
            )
            self.assertEqual(subprocess.run(command, cwd=workspace, capture_output=True).returncode, 0)

    def test_parent_session_env_is_not_inherited(self):
        import argparse
        import os
        from unittest import mock

        args = argparse.Namespace(base_url="http://gw", api_key=None, auth_token=None)
        inherited = {
            "CLAUDECODE": "1",
            "CLAUDE_CODE_SESSION_ID": "parent",
            "CLAUDE_CONFIG_DIR": "/x",
            "ANTHROPIC_API_KEY": "env-key",
            "ANTHROPIC_AUTH_TOKEN": "env-token",
            "CLAUDE_CODE_OAUTH_TOKEN": "env-oauth",
        }
        with mock.patch.dict(os.environ, inherited):
            env = probe.build_env(args, Path("/iso"))
        for key in ("CLAUDECODE", "CLAUDE_CODE_SESSION_ID", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
            self.assertNotIn(key, env)
        self.assertEqual(env["CLAUDE_CONFIG_DIR"], "/iso")
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://gw")


if __name__ == "__main__":
    unittest.main()
