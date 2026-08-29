# Qwen thinking budgets

Qwen3-family models expose two phases: a reasoning phase wrapped in
`<think>...</think>` and a final answer. A total completion limit does not
prevent the model from spending almost all of its response budget thinking.

The vLLM adapter in this toolkit sets `request.thinking_token_budget`. vLLM's
V1 reasoning parser counts tokens while the model is inside the thinking phase
and forces the closing `</think>` boundary when the budget is reached. The
budget limits reasoning only; it does not cap the final answer, and the model
may close reasoning earlier.

## Profiles

The legacy `action` profile is retained for existing production aliases. The
`qwen38fn` and `corpus` profiles use the larger ladder intended for Qwen3.8
Flash-Next and selected teacher captures:

| Request effort | Legacy `action` | `qwen38fn` / `corpus` |
|---|---:|---:|
| `none` | 0 | 0 |
| `minimal` | 128 | 512 |
| `low` | 256 | 1,024 |
| `medium` | 384 | 4,096 |
| `high` | 512 | 8,192 |
| `xhigh`/`max` | 512 | 16,384 |

`reasoning_effort` is accepted as a request field or inside
`chat_template_kwargs`. A request may also provide `thinking_token_budget`; it
is clamped to the active profile ceiling.

```json
{
  "model": "qwen3-family",
  "messages": [{"role": "user", "content": "Choose the next safe action."}],
  "reasoning_effort": "low",
  "max_completion_tokens": 1024
}
```

Use the model/template's supported toggle to disable thinking entirely:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

## Server configuration

The adapter is registered as `qwen3_budgeted` and is passed to vLLM with its
reasoning-parser plugin option. `QWEN_THINKING_PROFILE=action` is the default;
the Qwen3.8 Flash-Next vLLM recipe selects `qwen38fn`, while a dedicated
corpus lane selects `corpus`.

`QWEN_THINKING_TOKEN_BUDGET=N` makes one fixed budget the default and ceiling.
`QWEN_THINKING_TOKEN_BUDGET=-1` disables automatic bounding for a dedicated
capture process only. Never use that setting on an action seat.

## Distillation capture and top-5 logprobs

Use `bin/qwen-distill-capture` for selected difficult, no-tools teacher
requests. It sends a non-streaming completion with:

- the selected finite reasoning budget;
- `include_reasoning=true` and `return_token_ids=true`;
- opt-in `logprobs=true` and `top_logprobs=5`;
- session/request headers for provenance;
- no tool definitions or tool calls.

The capture record contains the prompt/messages, separated reasoning, answer,
token IDs, chosen-token logprob, at most five alternatives per token, request
and response metadata, and inferred budget/termination diagnostics. It uses a
persistent JSONL output path and a 512 MiB output quota by default. It never
requests or stores full-vocabulary logits.

A forced budget boundary is marked separately from a natural model closure.
For training, prefer natural-termination records or filter budget-terminated
records according to the intended curriculum. Top-5 logprobs are useful soft
labels and uncertainty signals, but they are not sufficient to reconstruct an
exact full-vocabulary KL objective.

The llama.cpp Qwen38fn implementation follows the same contract through
`patches/llama-cpp-qwen38fn-observability.patch`: normal tool-streaming
requests keep `n_probs=0`; explicit no-tools, non-streaming diagnostics can
combine reasoning budgets and top-5 logprobs. Its server-owned traces are
available at `/v1/reasoning-traces`.

## Compatibility and limitations

- The adapter uses vLLM reasoning-parser extension APIs, which are internal
  enough that a vLLM version pin and an integration smoke test are recommended.
- This is a verbosity/latency control, not a quality guarantee.
- A small thinking budget can truncate useful reasoning; compare task quality
  at each effort level before changing a production default.
- MTP and logprobs may be backend/version dependent; distillation capture
  disables tools and streaming for predictable alignment.
- Prompt instructions can still make the final answer verbose. Use a concise
  output instruction and an appropriate completion limit for that problem.
