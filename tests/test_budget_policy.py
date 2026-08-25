# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from qwen_budget_policy import resolve_budget  # noqa: E402


class BudgetPolicyTests(unittest.TestCase):
    def test_action_profile(self):
        self.assertEqual(resolve_budget(), 512)
        self.assertEqual(resolve_budget(effort="minimal"), 128)
        self.assertEqual(resolve_budget(effort="low"), 256)
        self.assertEqual(resolve_budget(effort="medium"), 384)
        self.assertEqual(resolve_budget(effort="high"), 512)
        self.assertEqual(resolve_budget(effort="max"), 512)

    def test_none_disables_thinking(self):
        self.assertEqual(resolve_budget(effort="none"), 0)

    def test_corpus_profile(self):
        self.assertEqual(resolve_budget(profile="corpus"), 1024)
        self.assertEqual(resolve_budget(profile="corpus", effort="high"), 2048)
        self.assertEqual(resolve_budget(profile="corpus", effort="max"), 4096)

    def test_explicit_budget_is_clamped(self):
        self.assertEqual(resolve_budget(explicit_budget=100), 100)
        self.assertEqual(resolve_budget(explicit_budget=10000), 512)

    def test_fixed_budget_sets_default_and_ceiling(self):
        self.assertEqual(resolve_budget(fixed_budget=300), 300)
        self.assertEqual(resolve_budget(fixed_budget=300, effort="low"), 256)
        self.assertEqual(resolve_budget(fixed_budget=300, effort="high"), 300)
        self.assertEqual(resolve_budget(fixed_budget=300, explicit_budget=100), 100)
        self.assertEqual(resolve_budget(fixed_budget=-1), None)
        self.assertEqual(resolve_budget(fixed_budget=-1, explicit_budget=7000), 7000)

    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            resolve_budget(profile="unknown")
        with self.assertRaises(ValueError):
            resolve_budget(explicit_budget=-2)
        with self.assertRaises(ValueError):
            resolve_budget(fixed_budget=-2)


if __name__ == "__main__":
    unittest.main()
