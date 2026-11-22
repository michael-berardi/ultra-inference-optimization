# Changelog

## 0.1.0

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
