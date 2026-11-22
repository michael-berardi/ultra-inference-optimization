"""Gates a candidate must pass before it can become 'best'."""
from __future__ import annotations

import random
import re
import secrets

from . import bench
from .bench import WORDS, stream_chat, get_text
import json
import http.client

GREEDY_PROMPTS = [
    "List the first twelve prime numbers separated by commas.",
    "Write a Python function that reverses a string, with a one-line docstring.",
    "Describe in three sentences how a bridge carries load.",
]


def _tokenize_len(base_url: str, text: str) -> int | None:
    conn, prefix = bench._conn(base_url, 60)
    try:
        conn.request("POST", prefix + "/tokenize", json.dumps({"content": text}).encode(),
                     {"Content-Type": "application/json"})
        r = conn.getresponse()
        data = r.read()
        if r.status != 200:
            return None
        return len(json.loads(data).get("tokens", [])) or None
    except (OSError, http.client.HTTPException, ValueError):
        return None
    finally:
        conn.close()


def fit_words(base_url: str, target_tokens: int, rng: random.Random) -> int:
    """How many filler words come to ~target tokens. Uses /tokenize when the server has it."""
    words = max(10, int(target_tokens / 1.5))  # conservative first guess
    for _ in range(3):
        n = _tokenize_len(base_url, bench.filler(random.Random(rng.random()), words))
        if n is None:
            return max(10, int(target_tokens / 1.7))
        words = max(10, int(words * target_tokens / n))
    return int(words * 0.98)


def make_needle_prompt(rng: random.Random, words: int, code: str) -> str:
    sents = bench.filler(rng, words).split(". ")
    pos = rng.randint(len(sents) // 10, max(len(sents) // 10, len(sents) * 9 // 10)) if len(sents) > 3 else 0
    sents.insert(pos, f"The vault passphrase is {code}")
    return (". ".join(sents) + "\n\nWhat is the vault passphrase? Answer with the passphrase only.")


THINK_BLOCK = re.compile(r"(?is)<think>.*?</think>")


def _norm(t: str) -> str:
    """The answer itself: a reasoning block left inline in the content (an empty <think></think> when thinking is switched off,
    or a whole block when the server does not split reasoning out) is not part of it (2026-11-21: a real model answered every
    needle exactly behind an empty think block and scored 0/6)."""
    t = THINK_BLOCK.sub("", t or "")
    return t.strip().strip("`'\".").strip()


def needle_gate(base_url: str, ctx: int, fractions=(0.5, 1.0), codes: int = 3, timeout: float = 1800.0,
                seed: int | None = None) -> dict:
    """Plant a random passphrase in a prompt filling ~50% and ~100% (92%, room for template + answer) of ctx."""
    rng = random.Random(seed if seed is not None else secrets.randbits(64))
    results = []
    for frac in fractions:
        target = int(ctx * (0.92 if frac >= 1.0 else frac))
        words = fit_words(base_url, target, rng)
        for _ in range(codes):
            code = f"{rng.choice(WORDS)}-{secrets.randbelow(9000) + 1000}"
            prompt = make_needle_prompt(rng, words, code)
            r = stream_chat(base_url, prompt, 24, temperature=0.0, seed=1, timeout=timeout, no_think=True)
            # An answer that arrives in the reasoning channel still counts; an empty answer never does.
            got = _norm(r["text"] or r["reasoning"]) if r["ok"] else None
            results.append({"fraction": frac, "promptTokens": r["promptTokens"], "ok": r["ok"],
                            "exact": bool(r["ok"] and got and got.lower() == code.lower()), "error": r["error"],
                            "expected": code, "got": got})
    exact = sum(1 for x in results if x["exact"])
    return {"passed": exact == len(results), "trials": results, "exact": exact, "total": len(results),
            "score": exact / len(results) if results else 0.0}


def needle_no_worse(cand: dict, baseline: dict | None) -> bool:
    """The needle gate is relative: the candidate may not retrieve fewer planted values than the unchanged baseline did.
    Without a usable baseline measurement it must pass outright."""
    if not cand.get("trials"):
        return False
    if not baseline or not baseline.get("trials"):
        return bool(cand.get("passed"))
    return cand.get("score", 0.0) >= baseline.get("score", 0.0)


def greedy_outputs(base_url: str, max_tokens: int = 48, timeout: float = 300.0) -> list[str | None]:
    out = []
    for p in GREEDY_PROMPTS:
        r = stream_chat(base_url, p, max_tokens, temperature=0.0, seed=1, timeout=timeout, no_think=True)
        out.append((r["reasoning"] + r["text"]) if r["ok"] else None)
    return out


def compare_greedy(a: list[str | None], b: list[str | None]) -> dict:
    for i, (x, y) in enumerate(zip(a, b)):
        if x is None or y is None or not x.strip() or not y.strip():
            return {"equal": False, "firstDivergence": {"prompt": i, "char": 0, "reason": "request failed or empty output"}}
        if x != y:
            j = next((k for k, (cx, cy) in enumerate(zip(x, y)) if cx != cy), min(len(x), len(y)))
            return {"equal": False, "firstDivergence": {"prompt": i, "char": j, "control": x[j:j + 24], "candidate": y[j:j + 24]}}
    return {"equal": len(a) == len(b), "firstDivergence": None}


def sanity(texts: list[str | None], flags: list[str]) -> dict:
    bad = []
    if "looping" in flags or any(t and bench.is_looping(t) for t in texts):
        bad.append("looping")
    if "failed" in flags or any(t is None for t in texts):
        bad.append("failed")
    if "idle" in flags or any(t is not None and not t.strip() for t in texts):
        bad.append("empty")
    return {"passed": not bad, "problems": bad}
