"""Normalised launch-argument maps: flag -> value, so two argument lists can be compared by meaning, not spelling."""
from __future__ import annotations

# Alias -> canonical long name (only spellings that are really the same option).
ALIASES = {
    "-fa": "--flash-attn", "-ctk": "--cache-type-k", "-ctv": "--cache-type-v", "-ub": "--ubatch-size",
    "-b": "--batch-size", "-t": "--threads", "-tb": "--threads-batch", "-np": "--parallel", "-c": "--ctx-size",
    "-ngl": "--n-gpu-layers", "--gpu-layers": "--n-gpu-layers", "-m": "--model", "-md": "--model-draft",
    "--draft-max": "--spec-draft-n-max", "--draft-n": "--spec-draft-n-max", "--draft": "--spec-draft-n-max",
    "--draft-p-min": "--spec-draft-p-min", "--draft-min": "--spec-draft-n-min",
}
# Flags that may legitimately appear several times; a flag -> value map cannot express them.
REPEATABLE = {"--lora", "--lora-scaled", "--override-kv", "--override-tensor", "-ot", "--control-vector",
              "--control-vector-scaled", "--override-tensor-draft", "-otd"}


def _is_flag(tok: str) -> bool:
    return len(tok) > 1 and tok.startswith("-") and not (tok[1].isdigit() or tok[1] == ".")


def canon(flag: str) -> str:
    return ALIASES.get(flag, flag)


def parse(args: list[str]) -> list[tuple[str, str, str | None]]:
    """[(canonical flag, spelling used, value or None for a bare flag)]. A leading non-flag token raises ValueError."""
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if not isinstance(a, str) or not _is_flag(a):
            raise ValueError(f"unexpected token {a!r} (not a flag)")
        if "=" in a:
            f, v = a.split("=", 1)
            out.append((canon(f), f, v))
            i += 1
        elif i + 1 < len(args) and not _is_flag(args[i + 1]):
            out.append((canon(a), a, args[i + 1]))
            i += 2
        else:
            out.append((canon(a), a, None))
            i += 1
    return out


def flag_map(args: list[str]) -> dict:
    """flag -> value (last wins). Repeatable flags map to a tuple of every value, in order."""
    m: dict = {}
    for f, _s, v in parse(args):
        if f in REPEATABLE or canon(f) in REPEATABLE:
            m[f] = tuple(m.get(f, ())) + (v,)
        else:
            m[f] = v
    return m


def same_args(a: list[str], b: list[str]) -> bool:
    try:
        return flag_map(a) == flag_map(b)
    except ValueError:
        return list(a) == list(b)


def repeatable_in(args: list[str]) -> list[str]:
    """Repeatable flags present, found by scanning tokens (some take several values, which parse() does not model)."""
    return sorted({str(a).split("=", 1)[0] for a in args if str(a).split("=", 1)[0] in REPEATABLE})
