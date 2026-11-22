import unittest

from support import FakeServer, FakeBehavior
from ultra_inference_optimization import bench
from ultra_inference_optimization.errors import Refused


class BenchTests(unittest.TestCase):
    def run_bench(self, mode="normal", samples=6, **kw):
        b = FakeBehavior()
        b.mode = mode
        for k, v in kw.items():
            setattr(b, k, v)
        fs = FakeServer(b)
        try:
            return bench.run_bench(fs.base_url, samples=samples, timeout=20), b
        finally:
            fs.close()

    def test_usage_counting_and_rates(self):
        res, b = self.run_bench()
        self.assertEqual(res["flags"], [])
        self.assertEqual(len(res["decodeTokS"]["samples"]), 18)   # 3 decode entries x 6 rounds
        self.assertEqual(len(res["prefillTokS"]["samples"]), 6)
        self.assertEqual(len(res["ttftMs"]["samples"]), 6)
        # 1 ms/token fake: decode must land in a sane band (sleep overhead makes it slower, never faster)
        self.assertTrue(100 < res["decodeTokS"]["median"] < 1100, res["decodeTokS"]["median"])
        self.assertTrue(res["panel"]["uniquePrefixPerRequest"])

    def test_chunk_count_trap(self):
        """Two tokens per chunk: counting chunks would halve the token count. Usage must win."""
        one, _ = self.run_bench("normal", tok_delay=0.002)
        two, _ = self.run_bench("multi", tok_delay=0.002)
        r = two["decodeTokS"]["median"] / one["decodeTokS"]["median"]
        self.assertTrue(0.7 < r < 1.4, r)  # same real speed; a chunk counter would report ~0.5

    def test_missing_usage_is_a_failure_not_a_guess(self):
        res, _ = self.run_bench("nousage")
        self.assertIn("failed", res["flags"])
        self.assertEqual(res["decodeTokS"]["samples"], [])
        self.assertTrue(any("no-usage" in (r["error"] or "") for r in res["perSample"]))

    def test_looping_detected(self):
        res, _ = self.run_bench("loop")
        self.assertIn("looping", res["flags"])

    def test_failed_requests_flagged(self):
        res, _ = self.run_bench("fail")
        self.assertIn("failed", res["flags"])
        self.assertEqual(res["panel"]["failures"], res["panel"]["requests"])

    def test_unique_prefix_per_request_and_fixed_panel(self):
        _res, b = self.run_bench()
        self.assertEqual(len(set(b.prompts)), len(b.prompts))
        stripped = {p.split("] ", 1)[1] for p in b.prompts if p.startswith("[session ")}
        self.assertEqual(len(stripped), 4)  # four distinct panel prompts, repeated each round

    def test_panel_is_seeded(self):
        self.assertEqual(bench.build_panel(7), bench.build_panel(7))
        self.assertNotEqual(bench.build_panel(7), bench.build_panel(8))
        self.assertEqual(bench.build_panel(1, long_context=True)[-1]["name"], "long")

    def test_too_few_samples_refused(self):
        with self.assertRaises(Refused):
            bench.run_bench("http://127.0.0.1:1", samples=3)

    def test_repeat_ratio(self):
        self.assertLess(bench.repeat_ratio(" ".join(f"w{i}" for i in range(100))), 0.1)
        self.assertGreater(bench.repeat_ratio("a b c d " * 50), 0.9)
        self.assertGreater(bench.repeat_ratio("abcdefgh" * 40), 0.9)


if __name__ == "__main__":
    unittest.main()
