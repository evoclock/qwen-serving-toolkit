from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwen_reasoning_capture import (  # noqa: E402
    MAX_TOP_LOGPROBS,
    build_request,
    cap_stored_logprobs,
    diagnostics,
)
from qwen_budget_policy import resolve_budget  # noqa: E402


class BudgetPolicyTests(unittest.TestCase):
    def test_shared_effort_mapping(self):
        expected = {
            "minimal": 512,
            "low": 1024,
            "medium": 4096,
            "high": 8192,
            "xhigh": 16384,
            "max": 16384,
        }
        for effort, budget in expected.items():
            self.assertEqual(resolve_budget(profile="qwen38fn", effort=effort), budget)
            self.assertEqual(resolve_budget(profile="corpus", effort=effort), budget)

    def test_action_default_preserves_legacy_contract(self):
        self.assertEqual(resolve_budget(profile="action"), 512)
        self.assertEqual(resolve_budget(profile="action", effort="medium"), 384)

    def test_new_profile_defaults(self):
        self.assertEqual(resolve_budget(profile="qwen38fn"), 512)
        self.assertEqual(resolve_budget(profile="corpus"), 4096)
        self.assertEqual(resolve_budget(profile="action", effort="none"), 0)

    def test_explicit_budget_is_clamped(self):
        self.assertEqual(resolve_budget(profile="qwen38fn", explicit_budget=20000), 16384)
        self.assertEqual(resolve_budget(profile="qwen38fn", fixed_budget=2048, effort="high"), 2048)


class CaptureRequestTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            profile="corpus",
            model="qwen3.6-35b-corpus",
            top_logprobs=MAX_TOP_LOGPROBS,
            max_completion_tokens=16384,
        )

    def test_capture_is_no_tools_non_streaming_top_five(self):
        request, budget = build_request(
            {
                "id": "hard-1",
                "messages": [{"role": "user", "content": "Solve this."}],
                "reasoning_effort": "high",
            },
            self.args,
        )
        self.assertEqual(budget, 8192)
        self.assertFalse(request["stream"])
        self.assertNotIn("tools", request)
        self.assertEqual(request["logprobs"], True)
        self.assertEqual(request["top_logprobs"], 5)
        self.assertEqual(request["thinking_token_budget"], 8192)

    def test_tool_requests_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "no-tools"):
            build_request(
                {
                    "messages": [{"role": "user", "content": "Use a tool."}],
                    "tools": [{"type": "function"}],
                },
                self.args,
            )

    def test_stored_logprobs_are_capped_even_if_backend_overreturns(self):
        response = {
            "choices": [{
                "logprobs": {
                    "content": [{
                        "token": "x",
                        "top_logprobs": [{"token": str(i)} for i in range(20)],
                    }]
                }
            }]
        }
        original = copy.deepcopy(response)
        stored = cap_stored_logprobs(response)
        self.assertEqual(len(response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]), 20)
        self.assertEqual(len(stored["choices"][0]["logprobs"]["content"][0]["top_logprobs"]), 5)
        self.assertEqual(response, original)

    def test_diagnostics_identifies_budget_and_top_five(self):
        response = {
            "usage": {"reasoning_tokens": 8192},
        }
        logprobs = {"content": [{"top_logprobs": [{} for _ in range(5)]}]}
        value = diagnostics(response, 8192, "reasoning", logprobs)
        self.assertEqual(value["termination"], "budget")
        self.assertEqual(value["reasoning_tokens"], 8192)
        self.assertEqual(value["max_returned_top_logprobs"], 5)


if __name__ == "__main__":
    unittest.main()
