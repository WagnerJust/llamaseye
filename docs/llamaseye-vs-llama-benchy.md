# llamaseye vs. llama-benchy

## Core philosophy

|  | **llamaseye** | **llama-benchy** |
|---|---|---|
| Language | Go (single static binary) | Python |
| Target | llama.cpp directly (GGUF files) | Any OpenAI-compatible HTTP endpoint |
| Goal | Find the *optimal config* for a model | Measure performance at *specified parameters* |
| Approach | Exhaustive parameter search (8 phases) | Manual parameter sweep you define |

## What they benchmark

**llamaseye** sweeps hardware config knobs: GPU layers, flash attention, KV quant types, thread count, KV offload, batch sizes, and context ceiling. It answers: "what settings make this model fastest on this machine?"

**llama-benchy** sweeps inference workload knobs: prompt size (pp), generation length (tg), context depth, and concurrency. It answers: "how does this server perform under these conditions?"

## Key differentiators

### llamaseye

- Finds the optimal config automatically — no need to know what to test upfront
- Handles OOM gracefully (falls back through ctk/nkvo combos)
- Thermal throttle detection
- Resume interrupted sweeps (`--resume`)
- TurboQuant KV cache support (`--turbo-bench`)
- No running server required — drives `llama-bench` directly against GGUF files

### llama-benchy

- Backend-agnostic — works with vLLM, SGLang, llama-server, or any `/v1/chat/completions` endpoint
- Measures end-user experience (HTTP latency, TTFT, TTFR)
- Prefix caching benchmarking (`--enable-prefix-caching`)
- Concurrency/throughput scaling (`--concurrency`)
- Mean ± std across multiple runs (`--runs`)
- Time-series data for token generation throughput
- Uses realistic text (Project Gutenberg book) for prompts — better for speculative decoding measurement
- Pip-installable Python package with integration test suite

## Overlap

Both measure prompt processing (pp) t/s and token generation (tg) t/s at varying context depths. That's roughly where the overlap ends.

## Summary

They are complementary, not competing tools.

- **llamaseye** is a pre-deployment config optimizer: "how should I run this model?"
- **llama-benchy** is a production performance characterizer: "how does my running server behave under load?"

A typical workflow: use llamaseye first to find the optimal llama.cpp flags for a given model and machine, then use llama-benchy to validate end-to-end serving performance with those settings.

## References

- llamaseye: https://github.com/WagnerJust/llamaseye
- llama-benchy: https://github.com/eugr/llama-benchy
