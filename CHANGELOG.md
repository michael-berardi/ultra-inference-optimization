# Changelog

## 0.2.0 (2026-11-26)

- Add release assets and a one-line installer: `install.sh` downloads the wheel and `SHA256SUMS`, verifies the wheel before installing, and uses uv, pipx or a private venv under `~/.local/share/uio`, with a managed shim in `~/.local/bin` that refuses to overwrite a foreign command.
- Keep the offline checkout installer working in the same script; add `install.ps1` for Windows via WSL and `scripts/build-release.sh` for reproducible wheel, installer and checksum assets.
- Add installer tests that build a local release directory and run the installer against it with a temporary HOME, covering venv, uv, pipx, bad checksums and foreign-shim refusal.
- Add `uv.lock` and bump the version to 0.2.0.

## 0.1.1 (2026-11-25)

- `llamacpp.get_value` now follows `llama-server` last-wins semantics when a flag is overridden by `extraArgs` after `launchArgs`, and `catalog()` respects `--cache-type-v=<type>` (`=` syntax) and overrides when deciding whether `-fa off` is valid.
- Close Linux `/sys` and `/proc` file handles cleanly via context managers in `sysinfo.py`.
- Fix single-file unittest execution (`python3 -m unittest tests/test_bench.py`) via `tests/__init__.py` and place `if __name__ == "__main__":` guards at file ends so direct runs execute all test classes.
- Replace the busy-wait in `tests/support.py` (`precise_sleep`) with a virtual clock on `FakeServer` (retaining a real-clock end-to-end timing test in `tests/test_bench.py`) and lower the server shutdown poll interval so `python3 -m unittest discover -s tests` runs quickly and deterministically without weakening assertions.
- Add `SECURITY.md` and `CONTRIBUTING.md`.

## 0.1.0 (2026-11-21)

First release. Tested against a real llama.cpp server before release, which found and fixed:
- hybrid models: KV layers are read from `attention.recurrent_layers` / `full_attention_interval` / fused qkv beside SSM state;
- built-in MTP heads are counted per draft step, not per verified token;
- an OpenAI-style base URL ending in `/v1` no longer doubles the path;
- GPU-busy preflight uses a 5 s median (desktop compositing no longer refuses every run);
- `--launch-args` measures the flags a model manager adds at launch, so the baseline is the server really in use;
- adaptive draft cutoff and draft backend sampling levers; `-fa off` is skipped with a quantised V cache.

Review fixes, before release:
- result contract v2: `applied` is always present (`written`, exact `extraArgs`, `backup`, `reason`), plus `confirmation`,
  `percentOfCeiling` (speculative ceiling when acceptance was measured) and `stoppedBecause`;
- statistics: a gain must exceed the gap between its two controls; the bootstrap resamples whole rounds and compares each
  prompt with itself; a confirmation round (control/best/control/best) must reproduce the gain before `--apply`;
  decode is benchmarked at temperature 0 with a seed; a single control exits early only when worse by more than 3x the
  control gap; plateau counts only levers whose every variant was noise; a non-target drop below 0.97 rejects on its own;
  no-op candidates are detected on normalised flag maps;
- apply: only the per-flag difference from the launch flags is written and verified; refuses on a removed launch flag or a
  repeatable flag (`launch-flag-removed`), on a profile edited during the run (`profile-changed`), writes an fsynced temp file,
  unique backup names;
- safety: SIGTERM/SIGHUP run the cleanup; the server PID and start time are recorded and a stale one is reaped by the next run;
  the memory watchdog starts at spawn and also watches system free memory, peak recorded in the ledger;
  the time budget is checked inside rounds and gates;
- gates: thinking disabled per request, reasoning text counted, empty output fails, needle gate is no worse than the baseline;
- hostile GGUF input (wrong types, layer count over 4096, nesting over 2, bad alignment, oversized counts) fails with a named cause;
- any unexpected exception exits 3 with the cause; `--launch-args` must be a JSON list of strings;
- `install.sh` refuses outside a checkout and installs a fixed launcher run by the absolute python3 with `-I`; the README
  install section uses the pinned release URL; the neutrality scan covers `tests/`.

Second review round:
- the reported ceiling percentage uses the benchmark's basis (short prompts, the server's KV cache type) and is withheld
  with a named `ceilingNote` when any measured decode exceeds its own ceiling (a low bandwidth figure printed 328%);
- with no accepted change, "after" is the baseline itself;
- the needle gate stops at the budget deadline, and `--apply` reserves the confirmation round inside the budget;
- a failed bandwidth measurement refuses with `no-bandwidth` instead of an unexpected error;
- an inline `<think>` block is stripped from gate answers.

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
