# KV, prefix, and runtime cache management

A model server usually has several different kinds of state. Calling all of
it "the cache" makes operational behavior confusing.

## State categories

1. **Weights** — the loaded model and draft-model parameters. Clearing a KV
   cache should not require reloading weights, but a service restart normally
   reloads them.
2. **KV cache** — attention keys and values for active conversations. This is
   the state that makes a continuation fast.
3. **Prefix/radix cache** — reusable KV prefixes shared by requests with the
   same prompt prefix. It can improve time-to-first-token but retains prompt
   state after a request finishes.
4. **Mamba/SSM state** — recurrent state used by hybrid Qwen architectures.
   It is separate from ordinary attention KV and may have its own radix/cache
   policy.
5. **CUDA graphs and compiled kernels** — runtime allocations and compilation
   artifacts. Some are cleared only by restarting the process or container.

## Safe clearing rule

A cache reset is disruptive. A safe control command should:

1. verify that the expected model owns the endpoint;
2. inspect running and queued request counts;
3. refuse to reset while work is active unless an explicit force mechanism is
   intentionally provided;
4. restart the service;
5. wait for the expected model ID to become ready again.

The example `bin/modelctl clear-cache` follows this pattern for vLLM and
SGLang metric names. It restarts the service rather than pretending that a
provider-neutral cache-flush endpoint exists.

```sh
MODELCTL_CONFIG="$HOME/.config/modelctl/qwen.env" modelctl status
MODELCTL_CONFIG="$HOME/.config/modelctl/qwen.env" modelctl clear-cache
```

Do not issue a cache reset merely because a request is slow. First check
whether the request is prefill-bound, waiting for admission, sharing decode
with other requests, or missing its expected prefix.

## Precision and capacity

KV precision is a capacity/performance choice as well as a quality risk. BF16
KV is a useful default when memory allows; FP8 or other compressed formats may
be valid exceptions after quality testing. Changing KV precision generally
requires a clean service restart.

`max_model_len` is an admission/context limit, not a promise that every
configured sequence can simultaneously occupy the full window. Total KV pool
capacity, recurrent state, graph allocations, and the number of active
sequences all matter.

## Operational cautions

- Never reset a shared service while another client owns active work.
- Keep cache directories and logs out of public repositories; prompts may be
  sensitive.
- Treat prefix caches as potentially sensitive retained input state.
- Runtime-specific commands should be documented next to their provider and
  version, not presented as universal OpenAI API behavior.
