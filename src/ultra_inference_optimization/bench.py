"""OpenAI-compatible streaming benchmark.

Rules that keep numbers honest:
* token counts come from the server's `usage` block, never from counting stream chunks;
* the whole prompt panel is built before the clock starts, from a seeded generator;
* every request carries a fresh random prefix so a prefix/prompt cache cannot hit;
* decode rate = (completion tokens - 1) / (time from first token to end of stream);
* prefill rate = prompt tokens / time-to-first-token on a cold, unique prompt.
"""
from __future__ import annotations

import http.client
import json
import os
import random
import re
import time
import urllib.parse

from .errors import Failed, Refused
from .stats import median

WORDS = ("amber anchor basin birch bridge candle canyon cedar chalk cinder clover copper coral crest dune "
         "ember falcon fern field flint forest garnet glacier harbor hazel heron hollow indigo island ivory "
         "jasper juniper kettle lagoon lantern lichen maple marble meadow mist moss nectar oak orchard otter "
         "pebble pine prairie quartz quill raven reef ridge river saffron sage shore silver slate spruce "
         "summit tide timber valley velvet willow window yarrow zephyr").split()
TOPICS = ["tides", "bridges", "orchards", "weather", "libraries", "glaciers", "harbors", "gardens", "clocks", "markets"]
CODE_TASKS = ["parse a CSV line with quoted fields", "merge two sorted lists", "debounce a function",
              "compute a rolling average", "validate a nested config dict", "flatten a tree of dicts"]


def _sentence(rng: random.Random, n: int = 12) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n)).capitalize() + "."


def filler(rng: random.Random, words: int) -> str:
    out, count = [], 0
    while count < words:
        n = rng.randint(8, 16)
        out.append(_sentence(rng, n))
        count += n
    return " ".join(out)


def build_panel(seed: int = 1, long_context: bool = False, long_words: int = 6000) -> list[dict]:
    """Fixed, seeded panel. kind: decode (measured for tok/s) or prefill (measured for TTFT/prefill)."""
    rng = random.Random(seed)
    task = rng.choice(CODE_TASKS)
    ident = "_".join(rng.sample(WORDS, 2))
    panel = [
        {"name": "code", "kind": "decode", "max_tokens": 160,
         "user": f"Write a Python function named {ident} that can {task}. Include a docstring and two short usage examples."},
        {"name": "prose", "kind": "decode", "max_tokens": 160,
         "user": f"Write three paragraphs about {rng.choice(TOPICS)} for a general reader. Notes: " + filler(rng, 40)},
        {"name": "short", "kind": "decode", "max_tokens": 64,
         "user": f"In two sentences, say something about {rng.choice(WORDS)}."},
        {"name": "prefill", "kind": "prefill", "max_tokens": 8,
         "user": "Read the text and reply with the single word OK.\n\n" + filler(rng, 700)},
    ]
    if long_context:
        panel.append({"name": "long", "kind": "prefill", "max_tokens": 8,
                      "user": "Read the text and reply with the single word OK.\n\n" + filler(rng, long_words)})
    return panel


def load_panel(spec: str, seed: int, long_context: bool, long_words: int = 6000) -> list[dict]:
    if spec in ("default", "", None):
        return build_panel(seed, long_context, long_words)
    try:
        with open(spec) as fh:
            data = json.load(fh)
    except (OSError, ValueError) as e:
        raise Refused("bad-panel", f"cannot read panel file {spec}: {e}")
    if not isinstance(data, list) or not data:
        raise Refused("bad-panel", "panel file must be a non-empty JSON list")
    out = []
    for i, e in enumerate(data):
        if not isinstance(e, dict) or "user" not in e:
            raise Refused("bad-panel", f"panel entry {i} needs a 'user' field")
        out.append({"name": e.get("name", f"entry{i}"), "kind": e.get("kind", "decode"),
                    "max_tokens": int(e.get("max_tokens", 128)), "user": e["user"], "system": e.get("system")})
    return out


def unique_prefix() -> str:
    return f"[session {os.urandom(8).hex()}] "


# ---- looping ------------------------------------------------------------------------
def repeat_ratio(text: str) -> float:
    """Fraction of overlapping shingles that are repeats. Prose and code sit well below 0.5."""
    words = text.split()
    if len(words) >= 40:
        sh = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
    elif len(text) >= 200:
        sh = [text[i:i + 24] for i in range(len(text) - 23)]
    else:
        return 0.0
    return 1.0 - len(set(sh)) / len(sh) if sh else 0.0


def is_looping(text: str, threshold: float = 0.5) -> bool:
    return repeat_ratio(text) > threshold


# ---- one streaming request ----------------------------------------------------------
def server_root(base_url: str) -> str:
    """The server root. Accepts either the root or an OpenAI-style base URL ending in /v1 (what clients are usually given),
    so /v1/chat/completions, /metrics, /tokenize and /health resolve the same way for both (0.1.0 doubled /v1 and every
    request 404ed against a real llama-server)."""
    root = base_url.rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


def _conn(base_url: str, timeout: float):
    u = urllib.parse.urlparse(server_root(base_url))
    if u.scheme not in ("http", "https"):
        raise Refused("bad-url", f"unsupported base url {base_url!r}")
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    return cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=timeout), u.path.rstrip("/")


def stream_chat(base_url: str, user: str, max_tokens: int, system: str | None = None,
                temperature: float = 0.7, seed: int | None = None, timeout: float = 600.0) -> dict:
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    body = {"messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": temperature}
    if seed is not None:
        body["seed"] = seed
    res = {"ok": False, "error": None, "text": "", "completionTokens": None, "promptTokens": None,
           "ttftS": None, "decodeS": None, "timings": None, "chunks": 0}
    conn, prefix = _conn(base_url, timeout)
    try:
        payload = json.dumps(body).encode()
        t0 = time.perf_counter()
        conn.request("POST", prefix + "/v1/chat/completions", payload, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            res["error"] = f"http-{resp.status}: {resp.read(300).decode('utf-8', 'replace')}"
            return res
        t_first = None
        text = []
        usage = None
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            try:
                obj = json.loads(data)
            except ValueError:
                continue
            if obj.get("error"):
                res["error"] = f"stream-error: {str(obj['error'])[:200]}"
                return res
            res["chunks"] += 1
            if obj.get("usage"):
                usage = obj["usage"]
            if obj.get("timings"):
                res["timings"] = obj["timings"]
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                piece = (d.get("content") or "")
                think = d.get("reasoning_content") or ""
                if piece or think:
                    if t_first is None:
                        t_first = time.perf_counter()
                    text.append(piece)
        t_end = time.perf_counter()
        res["text"] = "".join(text)
        if usage is None or usage.get("completion_tokens") is None:
            res["error"] = "no-usage: server did not return stream usage; chunk counts are never used as token counts"
            return res
        res["completionTokens"] = int(usage["completion_tokens"])
        res["promptTokens"] = int(usage.get("prompt_tokens") or 0)
        if t_first is None:
            res["ok"] = True
            return res
        res["ttftS"] = t_first - t0
        res["decodeS"] = t_end - t_first
        res["ok"] = True
        return res
    except (OSError, http.client.HTTPException) as e:
        res["error"] = f"{type(e).__name__}: {e}"
        return res
    finally:
        conn.close()


def get_text(base_url: str, path: str, timeout: float = 5.0) -> str | None:
    base_url = server_root(base_url)
    conn, prefix = _conn(base_url, timeout)
    try:
        conn.request("GET", prefix + path)
        r = conn.getresponse()
        data = r.read().decode("utf-8", "replace")
        return data if r.status == 200 else None
    except (OSError, http.client.HTTPException):
        return None
    finally:
        conn.close()


def scrape_draft_counters(base_url: str) -> tuple[float, float] | None:
    """(accepted, drafted) from a Prometheus /metrics page, when the server exposes them."""
    txt = get_text(base_url, "/metrics")
    if not txt:
        return None
    acc = tot = None
    for line in txt.splitlines():
        if line.startswith("#") or "draft" not in line:
            continue
        m = re.match(r"^([\w:]+)(?:\{[^}]*\})?\s+([-\d.eE+]+)", line)
        if not m:
            continue
        name, val = m.group(1), float(m.group(2))
        if "accept" in name:
            acc = val
        elif "total" in name or "drafted" in name or name.endswith("_n"):
            tot = val
    return (acc, tot) if acc is not None and tot else None


def _round(x, n=3):
    return None if x is None else round(x, n)


def run_bench(base_url: str, samples: int = 6, panel: list[dict] | None = None, seed: int = 1,
              long_context: bool = False, timeout: float = 600.0, min_samples: int = 6) -> dict:
    if samples < min_samples:
        raise Refused("too-few-samples", f"need >= {min_samples} samples for a median, got {samples}")
    panel = panel or build_panel(seed, long_context)
    # Everything below the clock is prepared here: prompts with their unique prefixes.
    plan = [(i, e, unique_prefix() + e["user"]) for i in range(samples) for e in panel]
    before = scrape_draft_counters(base_url)
    rows = []
    for rnd, e, user in plan:
        r = stream_chat(base_url, user, e["max_tokens"], system=e.get("system"), timeout=timeout)
        row = {"round": rnd, "name": e["name"], "kind": e["kind"], "ok": r["ok"], "error": r["error"],
               "promptTokens": r["promptTokens"], "completionTokens": r["completionTokens"],
               "ttftMs": _round(r["ttftS"] * 1000, 2) if r["ttftS"] is not None else None,
               "decodeTokS": None, "prefillTokS": None, "looping": False, "draftN": None, "draftAccepted": None}
        if r["ok"]:
            ct, dt = r["completionTokens"], r["decodeS"]
            if e["kind"] == "decode" and ct and ct >= 16 and dt and dt > 0:
                row["decodeTokS"] = round((ct - 1) / dt, 3)
            if e["kind"] == "prefill" and r["ttftS"] and r["promptTokens"]:
                row["prefillTokS"] = round(r["promptTokens"] / r["ttftS"], 3)
            row["looping"] = is_looping(r["text"])
            row["emptyOutput"] = (ct == 0)
            t = r["timings"] or {}
            if t.get("draft_n"):
                row["draftN"], row["draftAccepted"] = t["draft_n"], t.get("draft_n_accepted", 0)
        rows.append(row)
    after = scrape_draft_counters(base_url)

    def col(key):
        return [r[key] for r in rows if r.get(key) is not None]

    dec, pre = col("decodeTokS"), col("prefillTokS")
    ttft = [r["ttftMs"] for r in rows if r["kind"] == "prefill" and r["ttftMs"] is not None]
    flags = []
    if any(not r["ok"] for r in rows):
        flags.append("failed")
    if any(r["looping"] for r in rows):
        flags.append("looping")
    if any(r.get("emptyOutput") for r in rows):
        flags.append("idle")  # server answered but produced zero tokens
    dn = sum(r["draftN"] or 0 for r in rows)
    da = sum(r["draftAccepted"] or 0 for r in rows)
    acc = da / dn if dn else None
    if acc is None and before and after and after[1] > before[1]:
        acc = (after[0] - before[0]) / (after[1] - before[1])

    def agg(xs):
        return {"median": _round(median(xs)) if xs else None, "samples": xs}

    return {"decodeTokS": agg(dec), "prefillTokS": agg(pre), "ttftMs": agg(ttft),
            "draftAcceptance": _round(acc, 4), "flags": flags,
            "panel": {"seed": seed, "entries": [e["name"] for e in panel], "rounds": samples,
                      "uniquePrefixPerRequest": True, "requests": len(rows), "failures": sum(not r["ok"] for r in rows)},
            "perSample": rows}


def metrics_of(result: dict) -> dict:
    """Sample lists keyed by the statistic names used by the climbing code."""
    return {"decode": result["decodeTokS"]["samples"], "prefill": result["prefillTokS"]["samples"],
            "ttft": result["ttftMs"]["samples"]}
