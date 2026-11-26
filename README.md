# ultra-inference-optimization (`uio`)

Measure, bound and hill-climb the speed of a local LLM server, with evidence instead of vibes.
Pure Python 3.10+ standard library at runtime. No telemetry, no update checks, no network except
the server URL you point it at.

Inspired by the MIT-licensed `inference-optimisation` agent skill; this is an independent implementation.

## Install

Pinned release (works without a checkout):

```sh
DO_NOT_TRACK=1 uv tool install "git+https://github.com/michael-berardi/ultra-inference-optimization@v0.1.2"
# or: DO_NOT_TRACK=1 pipx install "git+https://github.com/michael-berardi/ultra-inference-optimization@v0.1.2"
```

From a checkout:

```sh
sh install.sh            # ~/.local/share/uio/<version> + shim ~/.local/bin/uio (refuses to overwrite a foreign uio)
```

`install.sh` refuses to run unless it sits in a checkout (it copies the tree next to it); piping it from a URL is refused.
The shim runs a fixed launcher file under the absolute `python3` found at install time, in isolated mode (`-I`).

## Commands

Every command prints one JSON object on stdout (`--human` for text). Exit codes: 0 ok, 2 refused (named cause), 3 failed.

| command | what it does |
| --- | --- |
| `uio probe` | chip, memory, cores, power state, measured memory bandwidth (method named), installed engines |
| `uio ceiling --gguf F [--ctx N] [--bandwidth GBs] [--draft-accept a --draft-n k] [--tflops T]` | theoretical decode / prefill ceilings, KV bytes per token, sequences under a memory cap, arithmetic printed |
| `uio bench --base-url URL [--samples 6] [--panel default\|FILE] [--long-context]` | decode tok/s, prefill tok/s, TTFT, draft acceptance, sanity flags |
| `uio tune --profile P.json [--budget 45m] [--target decode\|prefill\|ttft\|balanced] [--allow lossless\|lossy] [--apply]` | launches its own `llama-server`, hill-climbs the lever catalog, gates the winner |
| `uio climb --metric-cmd C --param NAME=v1,v2 --apply-cmd C` | the same statistics for any numeric metric |

`tune` reads a local profile JSON with `source.path` (model file), `ctx`, `extraArgs` (the baseline) and optionally `port`.
`--apply` rewrites only `extraArgs`, with a unique timestamped backup, an fsynced temp file, an atomic replace and mode 0600, and
only when every gate passed **and** a confirmation round reproduced the gain (see Guarantees). The result JSON always carries
`applied` (`written`, the exact `extraArgs` written, `backup`, and a `reason` when nothing was written, for example
`confirmation-failed`, `profile-changed` or `launch-flag-removed`). Only the per-flag difference from the launch flags is
written; a removed launch flag or a repeatable flag such as `--lora-scaled` refuses the write.

Method: see [docs/METHOD.md](docs/METHOD.md). Agent skill: [SKILL.md](SKILL.md).

## A real run

One real tune (`--budget 90m`, decode target), a hybrid model (17 of 65 layers with attention, built-in MTP draft head) as a
12.1 GB GGUF, llama.cpp, Apple M5 Max. Bandwidth was passed by hand as `--bandwidth 546` (an estimate, not a measurement:
the built-in stdlib probe reads 100-150 GB/s from CPU processes, a lower bound that is far below what the GPU streams on
unified memory; a no-speculation decode of 24.3 tok/s already exceeds the ceiling that figure would give). The numbers
below are from that run's `PLAN.md` and `ledger.jsonl`:

| | value |
| --- | --- |
| baseline (the profile's launch flags plus extra args) | decode 49.0 tok/s, prefill 396.9 tok/s, TTFT 2938 ms |
| a second run of the same baseline (control #11) | decode 50.7 tok/s |
| draft acceptance at the launch configuration | 0.57 (baseline), 0.60 (control) |
| draft acceptance across the draft-length sweep | 0.26 (16 drafted) to 0.80 (2 drafted); 0.74-0.90 with a confidence cutoff of 0.5-0.85 |
| ceiling, weights only (at the 546 GB/s estimate) | 46.8 tok/s (the speculative ceiling was not computed in that run) |
| levers reached | draft-length sweep (2, 3, 4, 6, 8, 12, 16 and speculation off), confidence cutoff (0.5, 0.7, 0.85), draft backend sampling |
| not reached | flash attention and ubatch size: the run stopped at the plateau rule after those three levers |
| accepted | none; every variant was equal to or worse than the baseline, and speculation off halved decode (24.3 tok/s) |

The useful result is the negative one: the launch flags were already the best of the levers tried, and the remaining gap to
the ceiling is draft acceptance (content dependent), not a flag. Two runs of the identical baseline differed by 3%, which is
why a gain must exceed the gap between its own two controls. That run used the earlier statistics (flat bootstrap, no
confirmation round, no baseline-relative needle gate); the rules below are what 0.1.0 now does.

## Guarantees

- Tokens are counted from the server's `usage`, never from stream chunks.
- Each request carries a fresh random prefix, so prefix caches cannot flatter the numbers.
- A change is kept only if it beats both neighbouring controls, its gain exceeds the gap between those two controls (the
  run-to-run drift of an unchanged config), the bootstrap 95% CI of the ratio is above 1.0 (whole rounds are resampled and
  each prompt is compared with itself), the gain is at least the minimum (2%), and no other metric is more than 3% worse
  (a point drop below 0.97 rejects on its own).
- A single control ends an experiment early only when the candidate is worse by more than 3x the control gap seen so far.
- Plateau means three consecutive levers whose every variant was indistinguishable from noise.
- Decode is benchmarked at temperature 0 with a fixed seed; every prompt keeps its unique random prefix.
- Gates ask the server not to think (`chat_template_kwargs` `enable_thinking: false` and `reasoning_format: none`), count
  reasoning text if it arrives anyway, and treat empty output as a failure. The needle gate is "no worse than the unchanged
  baseline", which is needle-tested once at the start.
- Before `--apply`, a confirmation round (baseline, best, baseline, best, run after the climb and outside its time budget)
  must reproduce the gain with the same rules, or nothing is written. When it runs, `baseline` and `best` in the result are
  the confirmation numbers.
- `percentOfCeiling` is best decode over the speculative ceiling when draft acceptance was measured (and the draft length is
  known), else over the weights-only ceiling.
- Tuning refuses on battery or in low-power mode, voids a run if that changes, kills the server at a memory cap or when
  system free memory runs out (watched from the moment the process spawns; the peak is recorded in the ledger), and holds an
  exclusive lock at `~/.cache/uio/lock`. SIGTERM and SIGHUP stop the server cleanly; a server left behind by a hard kill is
  found by PID and start time and stopped by the next run.
- The tool ships no model id, publisher, repository or quantisation default.

## Privacy

`uio` sends nothing over the network on its own. There is no telemetry, analytics, crash reporting, update check,
model download or upload of benchmark results, and no runtime dependencies. There is no shipped UI, web page, font
or remote script.

`uio bench` sends the benchmark prompts to the `--base-url` you supply (your prompts if you passed `--panel FILE`,
otherwise prompts generated locally). `uio tune` sends them to the `llama-server` it starts itself, bound to
`127.0.0.1`. If you point `--base-url` at a remote server, that server's operator receives the prompts and sees your
connection. `uio climb` runs the shell commands you give it; whatever they send is your own configuration.

Local state lives in `~/.cache/uio` (or `UIO_HOME`): run folders with plans, ledgers, configs and server logs, a lock
file holding a PID, a child-PID file, and a `last/` record holding profile paths and metrics. Files are 0600 and
directories newly created by `uio` are 0700; `~/.cache/uio` itself and everything below it are tightened to 0700
even if they already existed. Directories that already exist outside `~/.cache/uio` -- including an explicit
`--workdir` and a profile's parent -- are never chmodded and keep their own permissions. The newest ten run folders
are kept; older ones are deleted at the start of the next `tune` or `climb` (only for the default workdir; an
explicit `--workdir` is left alone). With `--apply`, a 0600 backup of the profile is written next to the profile.

Delete it all with `rm -rf ~/.cache/uio`, plus any `*.bak-*` files next to profiles you applied to. The install
commands above are run by you and fetch the pinned source; the installed tool does not fetch code.

## Tests

```sh
python3 -m unittest discover -s tests
```

## Licence

MIT, see [LICENSE](LICENSE).
