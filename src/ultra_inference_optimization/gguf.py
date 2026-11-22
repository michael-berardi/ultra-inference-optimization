"""Pure-Python GGUF header reader. Reads metadata and tensor directory only, never tensor data."""
from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field

from .errors import Failed

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_MAX_ARRAY_KEEP = 4096  # big arrays (vocab) are skipped, not stored


class _Reader:
    def __init__(self, fh):
        self.fh = fh

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

    def value(self, t: int):
        if t in _SCALARS:
            return self.unpack(_SCALARS[t])
        if t == 8:
            return self.string()
        if t == 9:
            et = self.unpack("<I")
            n = self.unpack("<Q")
            keep = n <= _MAX_ARRAY_KEEP
            out = []
            for _ in range(n):
                v = self.value(et)
                if keep:
                    out.append(v)
            return out if keep else {"arrayLength": n}
        raise Failed("gguf-corrupt", f"unknown metadata value type {t}")


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
        return int(self.key("block_count", 0))

    @property
    def n_embd(self) -> int:
        return int(self.key("embedding_length", 0))

    def _per_layer(self, suffix: str, default: int) -> list[int]:
        v = self.key(suffix)
        n = self.n_layer
        if isinstance(v, list):
            return [int(x) for x in v][:n] + [int(v[-1])] * max(0, n - len(v)) if v else [default] * n
        if v is None:
            return [default] * n
        return [int(v)] * n

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
        layers = {int(t.name.split(".")[1]) for t in self.tensors if t.name.startswith("blk.") and ".nextn." in t.name}
        return sum(t.nbytes for t in self.tensors if t.name.startswith("blk.") and int(t.name.split(".")[1]) in layers)

    @property
    def output_head_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors if t.name == "output.weight")

    @property
    def head_dims(self) -> tuple[int, int]:
        heads = max(self.head_counts or [0])
        default = self.n_embd // heads if heads else 0
        return int(self.key("attention.key_length", default)), int(self.key("attention.value_length", default))

    @property
    def total_tensor_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    @property
    def expert_count(self) -> int:
        return int(self.key("expert_count", 0) or 0)

    @property
    def expert_used(self) -> int:
        return int(self.key("expert_used_count", 0) or 0)

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
    try:
        size = os.path.getsize(path)
        fh = open(path, "rb")
    except OSError as e:
        raise Failed("gguf-unreadable", str(e))
    with fh:
        r = _Reader(fh)
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
    align = int(meta.get("general.alignment", 32) or 32)
    data_start = (header_end + align - 1) // align * align
    order = sorted(tensors, key=lambda t: t.offset)
    for i, t in enumerate(order):
        end = order[i + 1].offset if i + 1 < len(order) else size - data_start
        t.nbytes = max(0, end - t.offset)
    return GGUF(path, version, size, meta, tensors, data_start)
