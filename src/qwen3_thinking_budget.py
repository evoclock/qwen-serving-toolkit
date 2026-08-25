# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""vLLM adapter for bounded Qwen3-family reasoning.

The vLLM V1 sampler understands Qwen3 ``<think>...</think>`` state and forces
``</think>`` when ``request.thinking_token_budget`` is reached. This adapter
only resolves the request budget and registers the parser name.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# When vLLM loads this file directly from a plugin directory, make the sibling
# pure-policy module importable too.
_plugin_dir = str(Path(__file__).resolve().parent)
if _plugin_dir not in sys.path:
    sys.path.insert(0, _plugin_dir)

from qwen_budget_policy import resolve_budget  # noqa: E402
from vllm.parser.engine.registered_adapters import Qwen3ParserReasoningAdapter  # noqa: E402
from vllm.reasoning import ReasoningParserManager  # noqa: E402

_PROFILE_ENV = "QWEN_THINKING_PROFILE"
_FIXED_BUDGET_ENV = "QWEN_THINKING_TOKEN_BUDGET"
_DEFAULT_PROFILE = "action"


def _fixed_budget() -> int | None:
    raw = os.environ.get(_FIXED_BUDGET_ENV)
    if raw is None:
        return None
    value = int(raw.strip())
    if value < -1:
        raise ValueError(f"{_FIXED_BUDGET_ENV} must be >= -1")
    return value


def _request_effort(request) -> str | None:
    effort = getattr(request, "reasoning_effort", None)
    if effort is None:
        kwargs = getattr(request, "chat_template_kwargs", None) or {}
        effort = kwargs.get("reasoning_effort")
    return None if effort is None else str(effort).strip().lower()


class Qwen3BudgetedReasoningParser(Qwen3ParserReasoningAdapter):
    """Qwen3 parser with effort-aware server-side thinking limits."""

    def adjust_request(self, request):
        request = super().adjust_request(request)
        if not getattr(self._parser_engine, "_has_reasoning", True):
            return request
        initial_state = getattr(
            getattr(self._parser_engine, "parser_engine_config", None),
            "initial_state",
            None,
        )
        if getattr(initial_state, "name", None) == "CONTENT":
            return request

        explicit = getattr(request, "thinking_token_budget", None)
        budget = resolve_budget(
            profile=os.environ.get(_PROFILE_ENV, _DEFAULT_PROFILE),
            effort=_request_effort(request),
            explicit_budget=explicit,
            fixed_budget=_fixed_budget(),
        )
        if budget is not None:
            request.thinking_token_budget = budget
        return request


ReasoningParserManager.register_module(
    name="qwen3_budgeted",
    module=Qwen3BudgetedReasoningParser,
)
