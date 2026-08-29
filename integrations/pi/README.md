# Pi integration

This integration reads Qwen38fn traces from the existing llama.cpp server without `curl`, SSH, or a shell wrapper.

## Install on a client machine

Copy these two files into Pi's global resource locations:

- `extensions/qwen38fn-traces.ts` → `~/.pi/agent/extensions/qwen38fn-traces.ts`
- `skills/qwen38fn-traces/SKILL.md` → `~/.pi/agent/skills/qwen38fn-traces/SKILL.md`

Then reload Pi (`/reload`) or start a new Pi process. No model-server restart is needed for the Pi integration.

Set the private client-side endpoint in the machine's Pi environment/configuration:

```text
QWEN38FN_TRACE_URL=http://<private-spark-address>:8000/v1/reasoning-traces
```

The extension uses Node's native `fetch()`. It adds `X-Session-ID` and `X-Request-ID` to Qwen requests, then appends a `qwen38fn-reasoning-trace` custom entry to the local Pi session JSONL after the server trace is available. Custom entries are persisted locally but are not sent back to the model, so they do not increase the long-context prompt.

## Use

```text
/skill:qwen38fn-traces
```

Or ask the agent: “show the latest Qwen38fn reasoning trace.” The `qwen38fn_trace` custom tool lists and retrieves Spark traces over HTTP. `action=list` returns bounded metadata; `action=get` can return JSON or text. For a curated distillation sample, use `action=probe` with a difficult prompt and budget; it creates a separate no-tools completion with top-5 logprobs and saves the prompt. Tool output is capped at 48 KB for context safety.
