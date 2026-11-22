"""Pure-Python GGUF header reader. Reads metadata and tensor directory only, never tensor data."""
from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field

from .errors import Failed

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_MAX_ARRAY_KEEP = 4096  # big arrays (vocab) are skipped, not stored
_MAX_LAYERS = 4096
_MAX_DEPTH = 2          # arrays of arrays: at most this many levels
_SIZES = {t: struct.calcsize(f) for t, f in _SCALARS.items()}


class _Reader:
    def __init__(self, fh, size: int = 1 << 62):
        self.fh, self.size = fh, size

    def remaining(self) -> int:
        return self.size - self.fh.tell()

    def read(self, n: int) -> bytes:
        b = self.fh.read(n)
        if len(b) != n:
            raise Failed("gguf-truncated", "file ended inside the header")
        return b

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        n = self.unpack("<Q")
        if n > 1 << 26:
            raise Failed("gguf-corrupt", f"implausible string length {n}")
        return self.read(n).decode("utf-8", "replace")

    def value(self, t: int, depth: int = 0):
        if t in _SCALARS:
            return self.unpack(_SCALARS[t])
        if t == 8:
            return self.string()
        if t == 9:
            if depth >= _MAX_DEPTH:
                raise Failed("gguf-corrupt", f"arrays nested deeper than {_MAX_DEPTH}")
            et = self.unpack("<I")
            if et not in _SCALARS and et not in (8, 9):
                raise Failed("gguf-corrupt", f"unknown array element type {et}")
            n = self.unpack("<Q")
            # every element takes at least this many bytes: a count the file cannot hold is corrupt (and would spin for ever)
            if n * _SIZES.get(et, 8 if et == 8 else 12) > self.remaining():
                raise Failed("gguf-truncated", f"array of {n} elements does not fit in the file")
            keep = n <= _MAX_ARRAY_KEEP
            out = []
            for _ in range(n):
                v = self.value(et, depth + 1)
                if keep:
                    out.append(v)
            return out if keep else {"arrayLength": n}
        raise Failed("gguf-corrupt", f"unknown metadata value type {t}")


def _int(v, what: str) -> int:
    """A header number must be a real integer; a string, float, list or bool there is a corrupt or hostile file."""
    if isinstance(v, bool) or not isinstance(v, int):
        raise Failed("gguf-corrupt", f"{what} must be an integer, got {type(v).__name__}")
    return v


@dataclass
class Tensor:
    name: str
    dims: list[int]
    ggml_type: int
    offset: int
    nbytes: int = 0

    @property
    def elements(self) -> int:
        n = 1
        for d in self.dims:
            n *= d
        return n


@dataclass
class GGUF:
    path: str
    version: int
    file_size: int
    metadata: dict
    tensors: list[Tensor] = field(default_factory=list)
    data_start: int = 0

    # ---- convenience accessors -------------------------------------------------
    @property
    def arch(self) -> str:
        return str(self.metadata.get("general.architecture", "unknown"))

    def key(self, suffix: str, default=None):
        return self.metadata.get(f"{self.arch}.{suffix}", default)

    @property
    def n_layer(self) -> int:
        n = _int(self.key("block_count", 0), "block_count")
        if not 0 <= n <= _MAX_LAYERS:
            raise Failed("gguf-corrupt", f"implausible layer count {n}")
        return n

    @property
    def n_embd(self) -> int:
        return _int(self.key("embedding_length", 0), "embedding_length")

    def _per_layer(self, suffix: str, default: int) -> list[int]:
        v = self.key(suffix)
        n = self.n_layer
        if isinstance(v, list):
            vals = [_int(x, suffix) for x in v]
            return vals[:n] + [vals[-1]] * max(0, n - len(vals)) if vals else [default] * n
        if v is None:
            return [default] * n
        if isinstance(v, dict):
            raise Failed("gguf-corrupt", f"{suffix} is an oversized array")
        return [_int(v, suffix)] * n

    @property
    def head_counts(self) -> list[int]:
        return self._per_layer("attention.head_count", 0)

    @property
    def kv_head_counts(self) -> list[int]:
        return self._per_layer("attention.head_count_kv", max(self.head_counts or [0]) if self.n_layer else 0)

    @property
    def attention_layers(self) -> list[int]:
        """Layers that keep a KV cache. Hybrid models have recurrent layers without attention."""
        # Hybrid models state it: <arch>.attention.recurrent_layers (a per-layer bool list) or full_attention_interval.
        flags = self.key("attention.recurrent_layers", None)
        if isinstance(flags, list) and len(flags) == self.n_layer:
            return [i for i, recurrent in enumerate(flags) if not recurrent]
        interval = self.key("full_attention_interval", None)
        if isinstance(interval, int) and interval > 1:
            return [i for i in range(self.n_layer) if (i + 1) % interval == 0]
        # Otherwise a layer keeps KV when it has attention projections and no recurrent (SSM) state. A layer with a fused
        # attn_qkv beside ssm_* tensors is a linear-attention layer: it keeps a fixed-size state, not a KV cache.
        attn, ssm = set(), set()
        pat = re.compile(r"^blk\.(\d+)\.(attn_k|attn_qkv|attn_k_b|attn_kv_a_mqa)\.")
        spat = re.compile(r"^blk\.(\d+)\.ssm_")
        for t in self.tensors:
            m = pat.match(t.name)
            if m:
                attn.add(int(m.group(1)))
            m = spat.match(t.name)
            if m:
                ssm.add(int(m.group(1)))
        have = attn - ssm
        if have:
            return sorted(have)
        return list(range(self.n_layer))

    @property
    def draft_layer_bytes(self) -> int:
        """Bytes of a built-in multi-token-prediction (MTP / nextn) draft layer, read on every draft step."""
        def layer_of(t):
            parts = t.name.split(".")
            return int(parts[1]) if t.name.startswith("blk.") and len(parts) > 2 and parts[1].isdigit() else None
        layers = {layer_of(t) for t in self.tensors if ".nextn." in t.name} - {None}
        return sum(t.nbytes for t in self.tensors if layer_of(t) in layers)

    @property
    def output_head_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors if t.name == "output.weight")

    @property
    def head_dims(self) -> tuple[int, int]:
        heads = max(self.head_counts or [0])
        default = self.n_embd // heads if heads else 0
        return (_int(self.key("attention.key_length", default), "key_length"),
                _int(self.key("attention.value_length", default), "value_length"))

    @property
    def total_tensor_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    @property
    def expert_count(self) -> int:
        return _int(self.key("expert_count", 0) or 0, "expert_count")

    @property
    def expert_used(self) -> int:
        return _int(self.key("expert_used_count", 0) or 0, "expert_used_count")

    def summary(self) -> dict:
        kd, vd = self.head_dims
        att = self.attention_layers
        return {
            "path": self.path, "fileSizeBytes": self.file_size, "ggufVersion": self.version,
            "architecture": self.arch, "nLayer": self.n_layer, "nEmbd": self.n_embd,
            "nHead": max(self.head_counts or [0]),
            "nHeadKv": [self.kv_head_counts[i] for i in att if i < len(self.kv_head_counts)][:1] and
                       sorted({self.kv_head_counts[i] for i in att if i < len(self.kv_head_counts)}),
            "keyLength": kd, "valueLength": vd,
            "attentionLayers": len(att), "recurrentLayers": self.n_layer - len(att),
            "contextLength": self.key("context_length"),
            "expertCount": self.expert_count, "expertUsedCount": self.expert_used,
            "tensorCount": len(self.tensors), "tensorBytes": self.total_tensor_bytes,
        }


def read_gguf(path: str) -> GGUF:
    """Parse a header. Any malformed or hostile content surfaces as Failed, never as a stray exception or a hang."""
    try:
        g = _read_gguf(path)
        # Touch every derived number now so a bad type fails here, with a name, and not later inside the ceiling arithmetic.
        _ = (g.n_layer, g.n_embd, g.head_counts, g.kv_head_counts, g.head_dims, g.expert_count, g.expert_used,
             g.attention_layers)
        return g
    except Failed:
        raise
    except (struct.error, ValueError, TypeError, OverflowError, RecursionError, MemoryError, IndexError, KeyError) as e:
        raise Failed("gguf-corrupt", f"malformed header: {type(e).__name__}: {e}")


def _read_gguf(path: str) -> GGUF:
    try:
        size = os.path.getsize(path)
        fh = open(path, "rb")
    except OSError as e:
        raise Failed("gguf-unreadable", str(e))
    with fh:
        r = _Reader(fh, size)
        if r.read(4) != b"GGUF":
            raise Failed("gguf-bad-magic", f"{path} is not a GGUF file")
        version = r.unpack("<I")
        if version < 2:
            raise Failed("gguf-version", f"GGUF v{version} is not supported (need >= 2)")
        n_tensors = r.unpack("<Q")
        n_kv = r.unpack("<Q")
        if n_tensors > 1 << 24 or n_kv > 1 << 20:
            raise Failed("gguf-corrupt", "implausible counts in header")
        meta: dict = {}
        if n_tensors * 24 > size or n_kv * 12 > size:
            raise Failed("gguf-truncated", "header counts do not fit in the file")
        for _ in range(n_kv):
            k = r.string()
            t = r.unpack("<I")
            meta[k] = r.value(t)
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            nd = r.unpack("<I")
            if nd > 8:
                raise Failed("gguf-corrupt", f"tensor {name} has {nd} dims")
            dims = [r.unpack("<Q") for _ in range(nd)]
            gt = r.unpack("<I")
            off = r.unpack("<Q")
            tensors.append(Tensor(name, dims, gt, off))
        header_end = fh.tell()
    align = meta.get("general.alignment", 32)
    if isinstance(align, bool) or not isinstance(align, int) or not 0 < align <= 1 << 20:
        raise Failed("gguf-corrupt", f"general.alignment must be a positive integer, got {align!r}")
    data_start = (header_end + align - 1) // align * align
    order = sorted(tensors, key=lambda t: t.offset)
    for i, t in enumerate(order):
        end = order[i + 1].offset if i + 1 < len(order) else size - data_start
        t.nbytes = max(0, end - t.offset)
    return GGUF(path, version, size, meta, tensors, data_start)
