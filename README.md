# Qwen serving toolkit

[![License: AGPLv3 + Attribution](https://img.shields.io/badge/license-AGPLv3%20%2B%20Attribution-blue)](LICENSE)

Copyright (C) 2026 Julen Gamboa

Small, model-weight-free utilities for operating Qwen3-family reasoning
servers. The project focuses on reusable behavior rather than a collection of
machine-specific model recipes.

## Included

- an effort-aware vLLM reasoning adapter with bounded `<think>` generation;
- a dependency-free Qwen teacher-capture client with persistent top-5 logprobs;
- a llama.cpp observability patch with server-owned JSON/text traces;
- a pure-Python budget policy that can be tested without vLLM or a GPU;
- a conservative `modelctl` helper for systemd user services;
- request, service-template, and alias examples;
- reasoning-budget, trace-capture, and cache-management documentation;
- a no-shell Pi extension and named `qwen38fn-traces` skill.

## Quick start: budget policy tests

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
```

## vLLM integration

Install both source files into a persistent directory visible to the vLLM
plugin loader:

```sh
bin/install-vllm-plugin
```

Then configure a Qwen3-family vLLM server with the installed
`qwen3_thinking_budget.py` as its reasoning-parser plugin and register the
`qwen3_budgeted` parser. See `docs/thinking-budgets.md`.

For selected difficult teacher samples, use the no-tools, non-streaming
capture client. It records the prompt, separated reasoning, answer, token IDs,
exact request budget, and at most five top alternatives per token:

```sh
bin/qwen-distill-capture \
  --input tasks.jsonl \
  --output "$HOME/.local/state/qwen-serving-toolkit/vllm-corpus/captures.jsonl" \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3.8-flash-next \
  --profile qwen38fn \
  --top-logprobs 5
```

The capture output is quota-bounded at 512 MiB by default and never contains
full-vocabulary logits.

## llama.cpp integration

Apply `patches/llama-cpp-qwen38fn-observability.patch` to the pinned llama.cpp
Qwen4Exp revision used by the Qwen3.8 Flash-Next recipe. The patched server
maps effort labels to bounded reasoning budgets, keeps logprobs opt-in, caps
returned alternatives at top-5, and exposes read-only
`/v1/reasoning-traces` metadata/JSON/text access. The patch is independent of
vLLM APIs.

## Pi integration

Copy `integrations/pi/extensions/qwen38fn-traces.ts` into Pi's global extension
directory and `integrations/pi/skills/qwen38fn-traces/SKILL.md` into its global
skills directory. Set the private `QWEN38FN_TRACE_URL` client-side when the
server is remote, then use `/skill:qwen38fn-traces`. The extension uses native
`fetch()` and does not use `curl`, SSH, or shell wrappers.

This project is licensed under AGPL-3.0-only with the author-attribution
terms in `LICENSE`. Commercial use, forks, and substantial modifications are
permitted when all AGPL and attribution obligations are followed. A separate
commercial licence is available for organisations requiring proprietary
modifications or other terms incompatible with the AGPL.
