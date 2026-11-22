# Changelog

## 0.1.0 (2026-11-21)

First release. Tested against a real llama.cpp server before release, which found and fixed:
- hybrid models: KV layers are read from `attention.recurrent_layers` / `full_attention_interval` / fused qkv beside SSM state;
- built-in MTP heads are counted per draft step, not per verified token;
- an OpenAI-style base URL ending in `/v1` no longer doubles the path;
- GPU-busy preflight uses a 5 s median (desktop compositing no longer refuses every run);
- `--launch-args` measures the flags a model manager adds at launch, so the baseline is the server really in use;
- adaptive draft cutoff and draft backend sampling levers; `-fa off` is skipped with a quantised V cache.

## 0.1.0-pre

First release.

- `uio probe`: machine facts (macOS Apple silicon, Linux), power source and low-power mode, a measured
  memory-bandwidth figure with the method named, installed engines.
- `uio ceiling`: pure-Python GGUF header reader, decode / prefill / KV ceilings with the arithmetic printed,
  hybrid-attention and mixture-of-experts aware, speculative accepted-length math.
- `uio bench`: streaming OpenAI-compatible benchmark; tokens counted from `usage`, seeded panel, unique prefix
  per request, median of rounds, looping / failed / idle flags, draft acceptance when exposed.
- `uio tune` (llama.cpp): one change per experiment, control/candidate/control, bootstrap 95% CI, regression
  guard, needle / greedy-equivalence / sanity gates, plateau and budget stops, environment guard, memory watchdog,
  exclusive lock, ledger and plan files, safe `--apply`.
- `uio climb`: the same statistics for any numeric metric.
- `mlx_lm.server` and `vllm` engines are documented stubs that refuse with `engine-not-implemented`.
