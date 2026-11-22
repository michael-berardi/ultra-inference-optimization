---
name: ultra-inference-optimization
description: >
  Hill-climb the serving speed of a local LLM on one machine with evidence. Computes the theoretical
  ceiling first, benchmarks with noise controls, changes one thing at a time, keeps a ledger, gates the
  winner, and stops at a plateau. Use when asked to tune, speed up or benchmark a local model server,
  find the maximum tok/s a machine can give, or tune any numeric metric by trial. Command: `uio`.
---

# ultra-inference-optimization

You improve how fast one local model server runs on one machine, and you prove each gain.

## Loop

1. `uio probe`: machine, power state, measured memory bandwidth (the method is named in the output), engines.
2. `uio ceiling --gguf <file> --ctx N`: how fast decode can possibly be (bytes read per token / bandwidth),
   KV bytes per token, how many sequences fit under a memory cap. Read the printed arithmetic. If the user's
   target is above the ceiling, say so now.
3. `uio bench --base-url <url>`: baseline. Tokens come from the server's `usage` block; every request has a
   fresh random prefix; the result is a median over at least 6 rounds. Check `flags` (looping, failed, idle).
4. `uio tune --profile <profile.json> --budget 45m [--target decode|prefill|ttft|balanced] [--allow lossy]`:
   one change per experiment, control / candidate / control, accept only if the candidate beats both controls,
   the bootstrap 95% CI of the median ratio is above 1.0, and nothing else regresses by more than 3%.
   Winners must pass gates: needle retrieval at ~50% and ~100% of context, greedy equivalence for levers
   claimed lossless, output sanity. Add `--apply` to write `bestArgs` into the profile (backup, atomic, 0600).
5. `uio climb --metric-cmd "<prints a number>" --param NAME=a,b,c --apply-cmd "<cmd with {NAME}>"` for anything
   that is not an LLM server: same statistics, same ledger, same plateau rule.

## Rules

- Read the workdir's `PLAN.md` and `ledger.jsonl` before repeating an experiment. Never rerun a rejected one
  without a new reason.
- One change per experiment, derived from the current best. A lever the binary's `--help` does not list is skipped.
- Lossy levers (for example a quantised KV cache) are labelled and only run with `--allow lossy`.
- Do not benchmark on battery or in low-power mode; `uio` refuses and voids a run if that changes mid-run.
- Plateau = three consecutive levers with no accepted win. Stop there and report the percentage of the ceiling reached.
- Exit codes: 0 ok, 2 refused (the `error` field names the cause), 3 failed.

See `docs/METHOD.md` for the reasoning behind each rule.
