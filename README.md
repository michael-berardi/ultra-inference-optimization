# ultra-inference-optimization (`uio`)

Measure, bound and hill-climb the speed of a local LLM server, with evidence instead of vibes.
Pure Python 3.10+ standard library at runtime. No telemetry, no update checks, no network except
the server URL you point it at.

Inspired by the MIT-licensed `inference-optimisation` agent skill; this is an independent implementation.

## Install

```sh
sh install.sh            # ~/.local/share/uio/<version> + shim ~/.local/bin/uio (refuses to overwrite a foreign uio)
# or: pipx install .   /   uv tool install .
```

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
`--apply` rewrites only `extraArgs`, with a timestamped backup, an atomic replace and mode 0600, and only when every gate passed.

Method: see [docs/METHOD.md](docs/METHOD.md). Agent skill: [SKILL.md](SKILL.md).

## A real run

A 27B hybrid model (17 of 65 layers with attention, built-in MTP draft head), 3-bit GGUF, llama.cpp, Apple M5 Max:

| | value |
| --- | --- |
| baseline decode (median of 18) | 49-50 tok/s |
| prefill | ~400 tok/s |
| draft acceptance on the mixed prompt panel | 0.51 |
| ceiling, weights only (546 GB/s) | 46.8 tok/s |
| ceiling with the measured acceptance | 63.9 tok/s |
| levers tried | draft length 2-16 and off, confidence cutoff 0.5-0.85, draft backend sampling, flash attention, ubatch 256-2048 |
| accepted | none: the hand-tuned launch flags were already the best measured; speculation off halves decode |

The useful result is the negative one: the remaining gap to the ceiling is draft acceptance (content dependent), not a flag.
The first version of the ceiling also counted all 65 layers as KV layers and the MTP head as a per-token weight; the
fixed reader takes both from the file (`attention.recurrent_layers`, `nextn` tensors).

## Guarantees

- Tokens are counted from the server's `usage`, never from stream chunks.
- Each request carries a fresh random prefix, so prefix caches cannot flatter the numbers.
- A change is kept only if it beats both neighbouring controls, the bootstrap 95% CI of the median ratio is above 1.0,
  and nothing else regresses by more than 3%.
- Tuning refuses on battery or in low-power mode, voids a run if that changes, kills the server at a memory cap,
  and holds an exclusive lock at `~/.cache/uio/lock`.
- The tool ships no model id, publisher, repository or quantisation default.

## Tests

```sh
python3 -m unittest discover -s tests
```

## Licence

MIT, see [LICENSE](LICENSE).
