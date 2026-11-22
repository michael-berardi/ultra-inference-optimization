"""Theoretical ceilings from a GGUF header plus a bandwidth (and optionally a compute) figure.

Decode is bandwidth bound: tokens/s <= bandwidth / bytes read per emitted token.
Prefill is compute bound: tokens/s <= usable FLOP/s / FLOPs per token.
"""
from __future__ import annotations

from .gguf import GGUF

KV_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}
GIB = 1024 ** 3


def accepted_length(accept: float, n_draft: int) -> float:
    """Expected tokens emitted per verification pass when each drafted token is accepted with
    probability `accept` independently: 1 + a + a^2 + ... + a^n."""
    if accept >= 1.0:
        return float(n_draft + 1)
    return (1 - accept ** (n_draft + 1)) / (1 - accept)


def kv_bytes_per_token(g: GGUF, kv_type: str = "f16") -> float:
    kd, vd = g.head_dims
    heads = g.kv_head_counts
    per = KV_BYTES[kv_type]
    return sum(heads[i] * (kd + vd) * per for i in g.attention_layers if i < len(heads))


def active_weight_bytes(g: GGUF) -> tuple[float, int, list[str]]:
    """Bytes of weights touched per decoded token, active parameter count, and notes."""
    notes = []
    total = 0.0
    params = 0.0
    has_head = any(t.name == "output.weight" for t in g.tensors)
    frac = (g.expert_used / g.expert_count) if g.expert_count and g.expert_used else 1.0
    for t in g.tensors:
        if t.name == "token_embd.weight" and has_head:
            continue  # lookup reads one row per token, negligible
        if "_exps" in t.name and frac < 1.0:
            total += t.nbytes * frac
            params += t.elements * frac
        else:
            total += t.nbytes
            params += t.elements
    if frac < 1.0:
        notes.append(f"mixture of experts: {g.expert_used}/{g.expert_count} experts read per token")
    if not has_head:
        notes.append("no output.weight: tied embedding is read as the output head")
    return total, int(params), notes


def compute(g: GGUF, bandwidth_gbs: float, ctx: int = 4096, kv_type: str = "f16",
            draft_accept: float | None = None, draft_n: int | None = None,
            draft_gguf: GGUF | None = None, tflops: float | None = None,
            cap_bytes: float | None = None, compute_efficiency: float = 0.5) -> dict:
    bw = bandwidth_gbs * 1e9
    arith: list[str] = []
    w_bytes, params, notes = active_weight_bytes(g)
    arith.extend(notes)
    arith.append(f"active weight bytes per token = {w_bytes:,.0f} ({w_bytes / GIB:.3f} GiB); "
                 f"active parameters ~ {params:,}")
    kv_tok = kv_bytes_per_token(g, kv_type)
    att = g.attention_layers
    arith.append(f"KV bytes/token = sum over {len(att)} attention layers (of {g.n_layer}) of "
                 f"kv_heads x (key {g.head_dims[0]} + value {g.head_dims[1]}) x {KV_BYTES[kv_type]:.4f} B ({kv_type}) "
                 f"= {kv_tok:,.0f}")
    kv_read = kv_tok * ctx
    bytes_tok = w_bytes + kv_read
    plain0 = bw / w_bytes if w_bytes else None
    plain = bw / bytes_tok if bytes_tok else None
    arith.append(f"decode at empty ctx = {bandwidth_gbs:.1f} GB/s / {w_bytes:,.0f} B = "
                 f"{plain0:.2f} tok/s" if plain0 else "decode: no weights found")
    arith.append(f"decode at ctx {ctx} = {bandwidth_gbs:.1f} GB/s / ({w_bytes:,.0f} + KV read {kv_read:,.0f}) B = "
                 f"{plain:.2f} tok/s" if plain else "")
    spec = None
    if draft_accept is not None and draft_n is not None:
        L = accepted_length(draft_accept, draft_n)
        d_bytes = 0.0
        if draft_gguf is not None:
            d_bytes, _p, _n = active_weight_bytes(draft_gguf)
        cycle = bytes_tok + d_bytes * draft_n
        spec = bw / cycle * L if cycle else None
        arith.append(f"accepted length L = (1 - a^(k+1)) / (1 - a) with a={draft_accept}, k={draft_n} -> {L:.3f} tokens/pass")
        arith.append(f"speculative decode = L x bandwidth / (target bytes {bytes_tok:,.0f} + {draft_n} x draft bytes "
                     f"{d_bytes:,.0f}) = {spec:.2f} tok/s")
    prefill = None
    flops_tok = 2.0 * params + 4.0 * len(att) * g.n_embd * (ctx / 2.0)
    arith.append(f"prefill FLOPs/token ~ 2 x active params + 4 x attn layers x n_embd x ctx/2 = {flops_tok:,.0f}")
    if tflops:
        prefill = tflops * 1e12 * compute_efficiency / flops_tok
        arith.append(f"prefill ceiling = {tflops} TFLOP/s x {compute_efficiency} practical / FLOPs per token = {prefill:.1f} tok/s")
    else:
        arith.append("prefill ceiling needs a measured or stated compute peak: pass --tflops")
    seqs = None
    if cap_bytes:
        free = cap_bytes - g.file_size
        per_seq = kv_tok * ctx
        seqs = max(0, int(free // per_seq)) if per_seq else None
        arith.append(f"sequences under cap = floor(({cap_bytes / GIB:.2f} GiB cap - {g.file_size / GIB:.2f} GiB file) / "
                     f"({per_seq / GIB:.3f} GiB KV per {ctx}-token sequence)) = {seqs}")
    return {
        "model": g.summary(),
        "decodeTokS": {"weightsOnly": plain0, "atCtx": plain, "speculative": spec, "ctx": ctx,
                       "bandwidthGBs": bandwidth_gbs, "bytesPerToken": bytes_tok},
        "prefillTokS": prefill,
        "kvBytesPerToken": kv_tok,
        "kvType": kv_type,
        "maxSequencesUnderCap": seqs,
        "arithmetic": [a for a in arith if a],
    }
