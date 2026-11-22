import tempfile
import unittest
from pathlib import Path

from support import tiny_model
from ultra_inference_optimization import ceiling, gguf


class GgufTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.path = Path(self.d.name) / "m.gguf"

    def tearDown(self):
        self.d.cleanup()

    def test_reads_header_and_hybrid_layers(self):
        tiny_model(self.path)
        g = gguf.read_gguf(str(self.path))
        self.assertEqual(g.arch, "tiny")
        self.assertEqual((g.n_layer, g.n_embd), (4, 64))
        self.assertEqual(g.head_counts, [8] * 4)
        self.assertEqual(g.kv_head_counts, [2] * 4)
        self.assertEqual(g.attention_layers, [1, 3])
        self.assertEqual(g.head_dims, (8, 8))
        self.assertEqual(g.file_size, self.path.stat().st_size)
        by = {t.name: t.nbytes for t in g.tensors}
        self.assertEqual(by["token_embd.weight"], 1024)
        self.assertEqual(by["blk.1.attn_k.weight"], 128)
        self.assertEqual(g.metadata["tokenizer.ggml.tokens"], ["a", "b"])

    def test_non_hybrid_all_layers(self):
        tiny_model(self.path, hybrid=False)
        self.assertEqual(gguf.read_gguf(str(self.path)).attention_layers, [0, 1, 2, 3])

    def test_fused_qkv_linear_attention_with_ssm_state_keeps_no_kv(self):
        # Real hybrid layout (Qwen3.5-style): linear-attention layers carry a fused attn_qkv beside ssm_* state; only the full
        # attention layers keep KV. The first version counted every layer as KV (65 vs 17), a 3.8x KV error.
        from support import write_gguf
        F32 = 0
        meta = {"general.architecture": "hy", "hy.block_count": 4, "hy.embedding_length": 64, "hy.attention.head_count": 8,
                "hy.attention.head_count_kv": 2}
        tensors = []
        for i in range(4):
            if i % 2 == 0:
                tensors += [(f"blk.{i}.attn_qkv.weight", [64, 192], F32, 64), (f"blk.{i}.ssm_a", [16], F32, 64)]
            else:
                tensors += [(f"blk.{i}.attn_k.weight", [64, 16], F32, 64), (f"blk.{i}.attn_q.weight", [64, 64], F32, 64)]
        write_gguf(self.path, meta, tensors)
        self.assertEqual(gguf.read_gguf(str(self.path)).attention_layers, [1, 3])

    def test_recurrent_layers_flag_list_wins(self):
        from support import write_gguf
        meta = {"general.architecture": "hy", "hy.block_count": 4, "hy.embedding_length": 64, "hy.attention.head_count": 8,
                "hy.attention.recurrent_layers": [True, True, True, False]}
        write_gguf(self.path, meta, [(f"blk.{i}.attn_qkv.weight", [64, 192], 0, 64) for i in range(4)])
        self.assertEqual(gguf.read_gguf(str(self.path)).attention_layers, [3])

    def test_builtin_mtp_draft_is_counted_per_draft_step(self):
        from support import write_gguf
        meta = {"general.architecture": "hy", "hy.block_count": 2, "hy.embedding_length": 64, "hy.attention.head_count": 8}
        tensors = [("blk.0.attn_k.weight", [64, 16], 0, 1024), ("blk.1.attn_k.weight", [64, 16], 0, 1024),
                   ("blk.1.nextn.eh_proj.weight", [64, 64], 0, 512), ("output.weight", [64, 64], 0, 320)]
        write_gguf(self.path, meta, tensors)
        g = gguf.read_gguf(str(self.path))
        self.assertEqual(g.draft_layer_bytes, 1536)  # the whole nextn block (its attn_k plus the nextn projection)
        self.assertEqual(g.output_head_bytes, 320)
        w, _p, _n = ceiling.active_weight_bytes(g)
        self.assertEqual(w, 1024 + 1024 + 320)  # the nextn tensor itself is not a per-token target weight  # the nextn tensor is not counted as a per-token target weight

    def test_bad_magic_and_truncated(self):
        self.path.write_bytes(b"NOPE" + b"\0" * 40)
        with self.assertRaises(Exception) as c:
            gguf.read_gguf(str(self.path))
        self.assertEqual(c.exception.cause, "gguf-bad-magic")
        tiny_model(self.path)
        self.path.write_bytes(self.path.read_bytes()[:60])
        with self.assertRaises(Exception) as c:
            gguf.read_gguf(str(self.path))
        self.assertEqual(c.exception.cause, "gguf-truncated")


class CeilingTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.path = Path(self.d.name) / "m.gguf"

    def tearDown(self):
        self.d.cleanup()

    def test_kv_and_decode_arithmetic(self):
        tiny_model(self.path)
        g = gguf.read_gguf(str(self.path))
        # 2 attention layers x 2 kv heads x (8+8) x 2 bytes
        self.assertEqual(ceiling.kv_bytes_per_token(g), 128)
        self.assertEqual(ceiling.kv_bytes_per_token(g, "q8_0"), 2 * 2 * 16 * 34 / 32)
        w, params, _ = ceiling.active_weight_bytes(g)
        # token_embd skipped (output.weight present): 1024 + 4*512 + 4*512 + 2*128
        self.assertEqual(w, 1024 + 2048 + 2048 + 256)
        c = ceiling.compute(g, 10.0, ctx=1000, cap_bytes=g.file_size + 128 * 1000 * 5.5)
        self.assertAlmostEqual(c["decodeTokS"]["weightsOnly"], 10e9 / w)
        self.assertAlmostEqual(c["decodeTokS"]["atCtx"], 10e9 / (w + 128 * 1000))
        self.assertEqual(c["maxSequencesUnderCap"], 5)
        self.assertEqual(c["kvBytesPerToken"], 128)
        self.assertIsNone(c["prefillTokS"])
        self.assertTrue(any("decode at ctx 1000" in a for a in c["arithmetic"]))

    def test_speculative_accepted_length(self):
        self.assertAlmostEqual(ceiling.accepted_length(0.5, 3), 1.875)
        self.assertEqual(ceiling.accepted_length(1.0, 4), 5.0)
        tiny_model(self.path)
        g = gguf.read_gguf(str(self.path))
        c = ceiling.compute(g, 10.0, ctx=0, draft_accept=0.5, draft_n=3)
        w = c["decodeTokS"]["bytesPerToken"]
        self.assertAlmostEqual(c["decodeTokS"]["speculative"], 10e9 / w * 1.875)

    def test_moe_reads_active_experts_only(self):
        tiny_model(self.path, moe=True)
        g = gguf.read_gguf(str(self.path))
        w, _p, notes = ceiling.active_weight_bytes(g)
        self.assertEqual(w, 1024 + 2048 + 2048 + 256 + 4096 * 2 / 8)
        self.assertTrue(notes)

    def test_prefill_with_tflops(self):
        tiny_model(self.path)
        g = gguf.read_gguf(str(self.path))
        c = ceiling.compute(g, 10.0, ctx=100, tflops=1.0)
        self.assertGreater(c["prefillTokS"], 0)


if __name__ == "__main__":
    unittest.main()
