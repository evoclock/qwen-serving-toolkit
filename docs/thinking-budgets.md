# Qwen thinking budgets

Qwen3-family models commonly expose two phases: a reasoning phase wrapped in
`<think>...</think>` and a final answer. A total completion limit does not
prevent the model from spending almost all of its response budget thinking.

This toolkit adds a server-side vLLM reasoning adapter that sets
`request.thinking_token_budget`. The vLLM V1 sampler counts tokens while the
model is inside the thinking phase and forces the closing `</think>` token when
the budget is reached. The model then continues with the normal answer.

The budget limits reasoning only; it does not cap the final answer. The model
may close its reasoning earlier.

## Action profile

The intended production profile is deliberately bounded:

| Request effort | Thinking budget |
|---|---:|
| `none` | 0 |
| `minimal` | 128 |
| `low` | 256 |
| `medium` | 384 |
| `high` | 512 |
| `max` | 512 |

The adapter accepts `reasoning_effort` either as a request field or inside
`chat_template_kwargs`. A request may also provide `thinking_token_budget`; it
is clamped to the profile ceiling.

```json
{
  "model": "qwen3-family",
  "messages": [{"role": "user", "content": "Choose the next safe action."}],
  "reasoning_effort": "low",
  "max_completion_tokens": 1024
}
```

To disable thinking entirely, use the model/template's supported toggle:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

## Server configuration

The adapter is registered as `qwen3_budgeted` and is passed to vLLM with its
reasoning-parser plugin option. `QWEN_THINKING_PROFILE=action` is the default.
A separate `corpus` profile is available for data capture, with larger budgets;
it should not be used on an interactive action seat.

`QWEN_THINKING_TOKEN_BUDGET=N` makes one fixed budget the default and ceiling.
`QWEN_THINKING_TOKEN_BUDGET=-1` disables automatic bounding for a dedicated
capture process only. Do not use that setting on a production action service.

## SGLang/DFlash2

The Qwen3.8 SGLang/DFlash2 path does not use the vLLM adapter. It uses the
runtime's strict-thinking switch and `SGLANG_MAX_THINK_TOKENS` as a global
reasoning-phase ceiling. The same conceptual distinction applies: the ceiling
limits the thinking phase, while the request's completion budget limits the
whole response.

## Compatibility and limitations

- The adapter uses vLLM reasoning-parser extension APIs, which are internal
  enough that a vLLM version pin and an integration smoke test are recommended.
- This is a verbosity/latency control, not a quality guarantee.
- A small thinking budget can truncate useful reasoning; compare task quality
  at each effort level before changing a production default.
- Prompt instructions can still make the final answer verbose. Use a concise
  output instruction and an appropriate completion limit for that problem.
