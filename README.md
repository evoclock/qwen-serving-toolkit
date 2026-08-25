# Qwen serving toolkit

[![License: AGPLv3 + Attribution](https://img.shields.io/badge/license-AGPLv3%20%2B%20Attribution-blue)](LICENSE)

Copyright (C) 2026 Julen Gamboa

Small, model-weight-free utilities for operating Qwen3-family reasoning
servers. The project focuses on reusable behavior rather than a collection of
machine-specific model recipes.

## Included

- an effort-aware vLLM reasoning adapter with bounded `<think>` generation;
- a pure-Python budget policy that can be tested without vLLM or a GPU;
- a conservative `modelctl` helper for systemd user services;
- request, service-template, and alias examples;
- reasoning-budget and cache-management documentation.

## Quick start: budget policy tests

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
```

## vLLM integration

Install both source files into a directory visible to the vLLM plugin loader:

```sh
bin/install-vllm-plugin /tmp/qwen3-budgeted
```

Then configure a Qwen3-family vLLM server with the installed
`qwen3_thinking_budget.py` as its reasoning-parser plugin and register the
`qwen3_budgeted` parser. See `docs/thinking-budgets.md`.

This project is licensed under the GNU Affero General Public License version
3 or later, with the author-attribution terms in `LICENSE`.
