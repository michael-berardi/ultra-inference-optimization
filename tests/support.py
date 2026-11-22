"""Test support: synthetic GGUF writer, fake OpenAI-compatible streaming server, fake engine."""
import json
import re
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ultra_inference_optimization.engines.base import Engine, Server  # noqa: E402


# ---- GGUF ---------------------------------------------------------------------------------
def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def _val(v):
    if isinstance(v, bool):
        return struct.pack("<I", 7) + struct.pack("<?", v)
    if isinstance(v, int):
        return struct.pack("<I", 4) + struct.pack("<I", v)
    if isinstance(v, float):
        return struct.pack("<I", 6) + struct.pack("<f", v)
    if isinstance(v, str):
        return struct.pack("<I", 8) + _s(v)
    if isinstance(v, list) and v and isinstance(v[0], str):
        return struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(v)) + b"".join(_s(x) for x in v)
    if isinstance(v, list):
        return struct.pack("<I", 9) + struct.pack("<I", 4) + struct.pack("<Q", len(v)) + b"".join(struct.pack("<I", x) for x in v)
    raise TypeError(v)


def write_gguf(path, meta: dict, tensors: list, align: int = 32):
    """tensors: (name, dims, ggml_type, nbytes)"""
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(meta))
    for k, v in meta.items():
        head += _s(k) + _val(v)
    off = 0
    infos = b""
    offsets = []
    for name, dims, gt, nb in tensors:
        infos += _s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        infos += struct.pack("<I", gt) + struct.pack("<Q", off)
        offsets.append(off)
        off += (nb + align - 1) // align * align
    blob = head + infos
    pad = (-len(blob)) % align
    with open(path, "wb") as fh:
        fh.write(blob + b"\0" * pad)
        for (name, dims, gt, nb), o in zip(tensors, offsets):
            fh.write(b"\0" * ((nb + align - 1) // align * align))
    return path


def tiny_model(path, hybrid=True, moe=False):
    meta = {"general.architecture": "tiny", "tiny.block_count": 4, "tiny.embedding_length": 64,
            "tiny.attention.head_count": 8, "tiny.attention.head_count_kv": 2,
            "tiny.attention.key_length": 8, "tiny.attention.value_length": 8, "tiny.context_length": 2048,
            "tokenizer.ggml.tokens": ["a", "b"]}
    ts = [("token_embd.weight", [64, 100], 1, 1024), ("output.weight", [64, 100], 1, 1024)]
    for i in range(4):
        ts.append((f"blk.{i}.attn_q.weight", [64, 64], 1, 512))
        if not hybrid or i in (1, 3):
            ts.append((f"blk.{i}.attn_k.weight", [64, 16], 1, 128))
        ts.append((f"blk.{i}.ffn_down.weight", [64, 64], 1, 512))
    if moe:
        meta["tiny.expert_count"] = 8
        meta["tiny.expert_used_count"] = 2
        ts.append(("blk.0.ffn_up_exps.weight", [64, 64, 8], 1, 4096))
    return write_gguf(path, meta, ts)


def precise_sleep(d):
    end = time.perf_counter() + d
    while time.perf_counter() < end:
        time.sleep(0)


# ---- fake server --------------------------------------------------------------------------
class FakeBehavior:
    def __init__(self):
        self.mode = "normal"       # normal | multi | nousage | loop | fail
        self.tok_delay = 0.001
        self.prefill_delay_per_word = 0.00001
        self.cap_tokens = 16
        self.needle_ok = True
        self.greedy_suffix = ""
        self.requests = 0
        self.prompts = []


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def _chunk(self, obj):
        data = ("data: " + json.dumps(obj) + "\n\n").encode()
        self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
        self.wfile.flush()

    def do_POST(self):
        b = self.server.behavior
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        b.requests += 1
        prompt = body["messages"][-1]["content"]
        b.prompts.append(prompt)
        if b.mode == "fail":
            self.send_response(500)
            self.send_header("Content-Length", "4")
            self.end_headers()
            self.wfile.write(b"boom")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        words = len(prompt.split())
        m = re.search(r"vault passphrase is (\S+)", prompt)
        if m and "What is the vault passphrase" in prompt:
            toks = [m.group(1) if b.needle_ok else "unknown"]
        elif body.get("temperature") == 0 and words < 30:
            toks = ["greedy-%d%s" % (words % 7, b.greedy_suffix), "text", "here"] * 5
        elif b.mode == "loop":
            toks = ["the", "same", "words", "again"] * 30
        else:
            toks = ["tok%d" % i for i in range(min(body.get("max_tokens", 16), b.cap_tokens))]
        precise_sleep(words * b.prefill_delay_per_word)
        step = 2 if b.mode == "multi" else 1
        for i in range(0, len(toks), step):
            precise_sleep(b.tok_delay * step)
            piece = " ".join(toks[i:i + step]) + " "
            self._chunk({"choices": [{"delta": {"content": piece}}]})
        final = {"choices": [{"delta": {}, "finish_reason": "length"}]}
        if b.mode != "nousage":
            final["usage"] = {"prompt_tokens": words, "completion_tokens": len(toks)}
        self._chunk(final)
        self._chunk_raw(b"data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _chunk_raw(self, data):
        self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
        self.wfile.flush()


class FakeServer:
    def __init__(self, behavior=None):
        self.behavior = behavior or FakeBehavior()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.httpd.behavior = self.behavior
        self.httpd.daemon_threads = True
        self.httpd.handle_error = lambda *a: None
        self.port = self.httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._t = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._t.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


# ---- fake engine --------------------------------------------------------------------------
class _FakeHandle(Server):
    def __init__(self, fs):
        self.fs, self.base_url, self.pid = fs, fs.base_url, None

    def stop(self):
        self.fs.close()


class FakeEngine(Engine):
    """Speed and behaviour are pure functions of the argument list."""
    name = "fake"

    def __init__(self, delay=None, needle_ok=None, greedy_suffix=None):
        self.delay = delay or (lambda args: 0.001)
        self.needle_ok = needle_ok or (lambda args: True)
        self.greedy_suffix = greedy_suffix or (lambda args: "")
        self.starts = []

    def start(self, profile, args, port, log_path):
        self.starts.append(list(args))
        beh = FakeBehavior()
        beh.tok_delay = self.delay(args)
        beh.needle_ok = self.needle_ok(args)
        beh.greedy_suffix = self.greedy_suffix(args)
        return _FakeHandle(FakeServer(beh))
