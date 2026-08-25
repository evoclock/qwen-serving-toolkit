# SPDX-FileCopyrightText: 2026 Julen Gamboa <j.a.r.gamboa@gmail.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pure policy logic for bounded Qwen3-family reasoning.

This module intentionally has no vLLM dependency so the policy can be tested
on a laptop or in CI without a model, CUDA, or a server.
"""

from __future__ import annotations

from collections.abc import Mapping

PROFILES: Mapping[str, Mapping[str, int]] = {
    "action": {
        "default": 512,
        "max": 512,
        "minimal": 128,
        "low": 256,
        "medium": 384,
        "high": 512,
        "xhigh": 512,
        "max_effort": 512,
    },
    "corpus": {
        "default": 1024,
        "max": 4096,
        "minimal": 256,
        "low": 512,
        "medium": 1024,
        "high": 2048,
        "xhigh": 4096,
        "max_effort": 4096,
    },
}


def resolve_budget(
    *,
    profile: str = "action",
    effort: str | None = None,
    explicit_budget: int | None = None,
    fixed_budget: int | None = None,
) -> int | None:
    """Resolve a request's thinking budget.

    ``fixed_budget`` models ``QWEN_THINKING_TOKEN_BUDGET``. ``None`` means
    unset; ``-1`` means unbounded and should only be used by a dedicated
    capture/corpus process. A normal integer is both the default and ceiling.
    ``explicit_budget`` is a request-level value and is clamped to the active
    profile ceiling.
    """
    try:
        limits = PROFILES[profile.strip().lower()]
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"unknown reasoning profile: {profile!r}") from exc

    if fixed_budget is not None and fixed_budget < -1:
        raise ValueError("fixed_budget must be >= -1")

    normalized_effort = effort.strip().lower() if effort is not None else None
    if normalized_effort == "none":
        return 0

    # -1 disables automatic selection for a dedicated capture process. An
    # explicit request budget remains usable, but an unqualified request is
    # intentionally unbounded.
    if fixed_budget == -1:
        return explicit_budget

    maximum = fixed_budget if fixed_budget is not None else limits["max"]
    if explicit_budget is not None:
        if explicit_budget < 0:
            raise ValueError("explicit_budget must be non-negative")
        return min(explicit_budget, maximum)

    if normalized_effort in {"max", "max_effort"}:
        value = limits["max_effort"]
    elif normalized_effort in limits:
        value = limits[normalized_effort]
    elif fixed_budget is not None:
        value = fixed_budget
    else:
        value = limits["default"]
    return min(value, maximum)
