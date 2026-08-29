"""Shared effort-to-budget policy for Qwen vLLM serving and capture."""

from __future__ import annotations

from collections.abc import Mapping

# ``action`` preserves the existing production vLLM contract. The new
# qwen38fn and corpus profiles opt into the larger, common effort ladder used
# by the llama.cpp Qwen38fn service.
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
    "qwen38fn": {
        "default": 512,
        "max": 16384,
        "minimal": 512,
        "low": 1024,
        "medium": 4096,
        "high": 8192,
        "xhigh": 16384,
        "max_effort": 16384,
    },
    "corpus": {
        "default": 4096,
        "max": 16384,
        "minimal": 512,
        "low": 1024,
        "medium": 4096,
        "high": 8192,
        "xhigh": 16384,
        "max_effort": 16384,
    },
}


def resolve_budget(
    *,
    profile: str = "action",
    effort: str | None = None,
    explicit_budget: int | None = None,
    fixed_budget: int | None = None,
) -> int | None:
    """Resolve a request budget, returning ``None`` only for deliberate unbounded mode."""
    try:
        limits = PROFILES[profile.strip().lower()]
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"unknown reasoning profile: {profile!r}") from exc

    if fixed_budget is not None and fixed_budget < -1:
        raise ValueError("fixed_budget must be >= -1")

    normalized_effort = effort.strip().lower() if effort is not None else None
    if normalized_effort == "none":
        return 0

    if fixed_budget == -1:
        if explicit_budget is None:
            return None
        if explicit_budget < 0:
            raise ValueError("explicit_budget must be non-negative")
        return explicit_budget

    maximum = fixed_budget if fixed_budget is not None else limits["max"]
    if maximum < 0:
        raise ValueError("fixed_budget must be >= 0 or -1")

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
