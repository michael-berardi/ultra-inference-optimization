# Method

This is the method `uio` automates. It works by hand too.

## 1. Ceilings first

Before changing anything, work out how fast the machine could possibly go.

- **Decode is bandwidth bound.** Each emitted token reads the active weights once plus the KV cache at the current
  context. `tok/s <= bandwidth / bytes per token`. Hybrid models only keep KV for their attention layers; mixture
  models only read the experts they route to.
- **Speculation** emits `L = 1 + a + ... + a^k` tokens per target pass (acceptance `a`, draft length `k`), but each pass
  also pays for the draft's own bytes. A better ceiling, not a free lunch.
- **Prefill is compute bound.** FLOPs per token are about twice the active parameters plus an attention term that grows
  with position. Divide a usable fraction of measured peak by that.
- **KV** per token times context tells you how many sequences fit under a memory cap.

Report progress as a percentage of the ceiling. A target above the ceiling is a finding, not a task.

## 2. One change at a time

Every experiment is a single change derived from the current best configuration, in its own config file. Two changes
in one run make the result unattributable.

## 3. Controls

Machines drift: thermals, background work, memory pressure. So each candidate is sandwiched: control, candidate,
control, all on the same configuration of the rest. A candidate is only a win if it beats both neighbouring controls and
by more than the gap between those two controls.

## 4. Statistics

- At least 6 samples per run; use the median, not the mean.
- Bootstrap the ratio of medians (candidate over pooled controls), resampling whole rounds and comparing each prompt with
  itself. The 95% interval's lower bound must exceed 1.0, and the point gain must clear a small minimum (default 2%) and the
  gap between the two controls, so statistically positive but useless changes do not pile up.
- No other metric may be more than 3% worse (a point drop below 0.97 rejects on its own).
- Before a result is applied, a fresh control/best/control/best round must reproduce the gain.

## 5. Honest measurement

- Count tokens from the server's `usage`, never from stream chunks (a chunk may carry several tokens).
- Build the prompts before the clock starts, from a seeded generator. Put a fresh random prefix on every request so a
  prefix cache cannot answer.
- Decode rate = tokens after the first / time since the first token. Prefill rate = prompt tokens / time to first token
  on a cold prompt.
- Do not benchmark on battery or in low-power mode. Re-check after every run and void the run if it changed.
- Watch memory. On unified memory a runaway allocation takes the whole machine, so a watchdog stops the server at a cap.

## 6. Ledger

Every run appends a line to `ledger.jsonl` and every experiment a line to `PLAN.md`: numbers, verdict, reason.
The ledger is how a later session avoids repeating a rejected idea.

## 7. Gates before "best"

A faster configuration that is wrong is not better.

- **Needle retrieval:** random codes planted in prompts that fill about half and about all of the context must be
  returned exactly.
- **Greedy equivalence:** a change claimed lossless must give identical text at temperature 0; report the first divergence.
- **Output sanity:** no looping, no empty or failed responses.

Lossy changes (quantised KV, for instance) are labelled, opt-in, and still pass the needle and sanity gates.

## 8. Plateau

Work through levers ordered by typical payoff. Three consecutive levers whose every variant was noise is a plateau: stop,
report the percentage of ceiling reached and what the remaining gap would need (a different model, quantisation,
engine or hardware are outside a launch-flag search).

## 9. Publish with pins

If you share a result, pin everything that produced it: engine version and commit, driver or OS version, the exact
launch arguments, the model file's hash, the panel seed, and the machine. A number without pins cannot be reproduced.
