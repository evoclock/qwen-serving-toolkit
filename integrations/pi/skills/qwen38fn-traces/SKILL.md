---
name: qwen38fn-traces
description: Query Qwen3.8-Flash-Next reasoning budgets, token accounting, termination, saved reasoning text, answers, and optional logprob diagnostics from the DGX Spark. Use when the user asks about a Qwen38fn trace, reasoning budget, forced or natural </think>, or top-logprobs.
---

# Qwen38fn traces

Use the `qwen38fn_trace` Pi tool. It talks to the read-only trace endpoint over native HTTP; do not use `bash`, `curl`, `ssh`, or a shell wrapper.

## Common requests

- Latest trace summary: call `qwen38fn_trace` with `action: "list"`, `limit: 1`.
- Recent traces for this Pi session: call `action: "list"` with the current session ID when available.
- Full JSON: list first, then call `action: "get"`, passing the returned `id` and `format: "json"`.
- Human-readable trace: list first, then call `action: "get"` with the returned `id` and `format: "text"`.
- Opt into a curated distillation capture: call `action: "probe"` with the difficult task as `prompt`, a budget such as `8192`, and an appropriate effort. This creates a separate no-tools, non-streaming teacher completion with top-5 logprobs and saves the prompt in the trace. It does not retroactively add logprobs to an earlier tool-call response.
- Retrieve the probe’s full token alternatives by calling `action: "get"` with its returned `trace_id` and `format: "json"`.

Report at least the budget, counted/reasoning tokens, remaining tokens, termination (`budget`, `natural`, or other), end-tag status, prompt/completion token counts, and whether the trace was saved. Do not claim that `top_k` is returned top-logprobs: returned alternatives are present only when the diagnostic request explicitly enabled logprobs, and this recipe caps returned alternatives at top-5 per token.

The Spark retains the authoritative full JSON/text trace. Pi also records a compact `qwen38fn-reasoning-trace` custom entry in the local session JSONL when the Qwen provider request carries the session headers.
