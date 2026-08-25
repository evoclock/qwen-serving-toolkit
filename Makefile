# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

.PHONY: test check

test:
	python3 -m unittest discover -s tests -p 'test_*.py'

check:
	python3 -m py_compile src/qwen_budget_policy.py src/qwen3_thinking_budget.py
	bash -n bin/install-vllm-plugin bin/modelctl examples/aliases.sh
