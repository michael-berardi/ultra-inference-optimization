import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from support import FakeEngine, tiny_model
from ultra_inference_optimization import sysinfo
from ultra_inference_optimization.engines.base import Lever, Variant
from ultra_inference_optimization.errors import Refused
from ultra_inference_optimization.tune import Tuner


def add(flag):
    return lambda a: a + [flag]


def lever(name, flag, lossless=True):
    return Lever(name, [Variant(flag, add(flag))], lossless)


# speed model: --fast = 35% faster, --meh = no effect, --slow = slower
def delay(args):
    d = 0.0008
    if "--fast" in args or "--fast-bad-needle" in args or "--fast-drift" in args:
        d *= 0.65
    if "--slow" in args:
        d *= 1.5
    return d


class TuneBase(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.root = Path(self.d.name)
        self.env = mock.patch.dict(os.environ, {"UIO_HOME": str(self.root / "home")})
        self.env.start()
        self.model = tiny_model(self.root / "m.gguf")
        self.profile = self.root / "profile.json"
        self.profile.write_text(json.dumps({"id": "p", "source": {"path": str(self.model)}, "ctx": 1500,
                                            "extraArgs": ["-b", "512"], "other": 1}))
        os.chmod(self.profile, 0o644)
        self.ac = mock.patch.object(sysinfo, "read_power", lambda: {"powerSource": "ac", "powerProfile": "normal"})
        self.ac.start()

    def tearDown(self):
        self.ac.stop()
        self.env.stop()
        self.d.cleanup()

    def tuner(self, levers, engine=None, **kw):
        eng = engine or FakeEngine(delay=delay, needle_ok=lambda a: "--fast-bad-needle" not in a,
                                   greedy_suffix=lambda a: "X" if "--fast-drift" in a else "")
        kw.setdefault("apply", False)
        kw.setdefault("min_gain", 0.05)
        return Tuner(str(self.profile), eng, workdir=self.root / "wd", levers=levers, preflight=False,
                     bandwidth_gbs=100.0, **kw)


class TuneTests(TuneBase):
    def test_accepts_real_win_and_writes_artifacts(self):
        r = self.tuner([lever("speed", "--fast")]).tune()
        self.assertEqual(r["bestArgs"], ["-b", "512", "--fast"])
        self.assertEqual(len(r["accepted"]), 1)
        a = r["accepted"][0]
        self.assertTrue(a["gates"]["passed"])
        self.assertGreater(a["ci95"][0], 1.0)
        self.assertGreater(r["best"]["decode"], r["baseline"]["decode"] * 1.2)
        wd = self.root / "wd"
        for f in ("PLAN.md", "THEORY.md", "ledger.jsonl", "best-config.json"):
            self.assertTrue((wd / f).exists(), f)
        self.assertTrue(list((wd / "configs").glob("*.json")))
        roles = [json.loads(l)["role"] for l in (wd / "ledger.jsonl").read_text().splitlines()]
        self.assertEqual(roles[:3], ["baseline", "candidate", "control"])  # interleaved
        self.assertIn("gate", roles)

    def test_rejects_noise(self):
        r = self.tuner([lever("nothing", "--meh")]).tune()
        self.assertEqual(r["bestArgs"], ["-b", "512"])
        self.assertEqual(r["accepted"], [])
        self.assertIn(r["rejected"][0]["verdict"], ("noise", "worse"))

    def test_rejects_slowdown(self):
        r = self.tuner([lever("slow", "--slow")]).tune()
        self.assertEqual(r["accepted"], [])

    def test_stops_at_plateau_after_three_levers(self):
        levers = [lever(f"n{i}", f"--meh{i}") for i in range(6)]
        r = self.tuner(levers).tune()
        self.assertTrue(r["plateau"])
        self.assertEqual(r["stopReason"], "plateau")
        self.assertEqual(len({e["lever"] for e in r["rejected"]}), 3)

    def test_win_then_continues_from_new_best(self):
        r = self.tuner([lever("a", "--fast"), lever("b", "--meh")]).tune()
        self.assertEqual(r["bestArgs"], ["-b", "512", "--fast"])
        self.assertEqual(r["rejected"][0]["args"], ["-b", "512", "--fast", "--meh"])  # derived from current best

    def test_failed_needle_gate_blocks_best_and_apply(self):
        before = self.profile.read_text()
        r = self.tuner([lever("bad", "--fast-bad-needle")], apply=True).tune()
        self.assertEqual(r["accepted"], [])
        self.assertEqual(r["rejected"][0]["verdict"], "rejected-gate")
        self.assertFalse(r["rejected"][0]["gates"]["needle"]["passed"])
        self.assertEqual(r["bestArgs"], ["-b", "512"])
        self.assertEqual(self.profile.read_text(), before)
        self.assertFalse(list(self.root.glob("profile.json.bak-*")))

    def test_lossless_lever_with_greedy_drift_is_rejected(self):
        r = self.tuner([lever("drift", "--fast-drift")]).tune()
        self.assertEqual(r["accepted"], [])
        g = r["rejected"][0]["gates"]["greedyEquivalence"]
        self.assertFalse(g["equal"])
        self.assertIsNotNone(g["firstDivergence"])

    def test_lossy_lever_skipped_unless_allowed(self):
        lv = lever("lossy", "--fast-drift", lossless=False)
        r = self.tuner([lv]).tune()
        self.assertEqual(r["skipped"][0]["lever"], "lossy")
        r2 = self.tuner([lv], allow="lossy").tune()
        self.assertEqual(len(r2["accepted"]), 1)  # drift reported but not gating for lossy

    def test_apply_writes_backup_atomic_0600(self):
        r = self.tuner([lever("speed", "--fast")], apply=True).tune()
        data = json.loads(self.profile.read_text())
        self.assertEqual(data["extraArgs"], ["-b", "512", "--fast"])
        self.assertEqual(data["other"], 1)
        self.assertEqual(stat.S_IMODE(self.profile.stat().st_mode), 0o600)
        baks = list(self.root.glob("profile.json.bak-*"))
        self.assertEqual(len(baks), 1)
        self.assertEqual(json.loads(baks[0].read_text())["extraArgs"], ["-b", "512"])
        self.assertEqual(stat.S_IMODE(baks[0].stat().st_mode), 0o600)
        self.assertTrue(r["applied"]["written"])

    def test_budget_enforced(self):
        r = self.tuner([lever("a", "--fast")], budget_s=0.0).tune()
        self.assertEqual(r["stopReason"], "budget")
        self.assertEqual(r["accepted"], [])

    def test_environment_change_voids_run_and_stops(self):
        states = iter([{"powerSource": "ac", "powerProfile": "normal"}] * 9)

        def power():
            return next(states, {"powerSource": "battery", "powerProfile": "normal"})
        with mock.patch.object(sysinfo, "read_power", power):
            r = self.tuner([lever("a", "--fast")], apply=True).tune()
        self.assertEqual(r["stopReason"], "environment-changed")
        self.assertFalse(r["applied"]["written"])
        ledger = [json.loads(l) for l in (self.root / "wd" / "ledger.jsonl").read_text().splitlines()]
        self.assertTrue(any(x.get("voided") for x in ledger))

    def test_stub_engines_refuse_by_name(self):
        from ultra_inference_optimization.engines import get_engine
        for name in ("mlx_lm.server", "vllm"):
            with self.assertRaises(Refused) as c:
                get_engine(name).start({}, [], 1, None)
            self.assertEqual(c.exception.cause, "engine-not-implemented")
        with self.assertRaises(Refused):
            get_engine("nope")


class PreflightTests(TuneBase):
    def test_refuses_on_battery(self):
        with mock.patch.object(sysinfo, "read_power", lambda: {"powerSource": "battery", "powerProfile": "normal"}):
            t = Tuner(str(self.profile), FakeEngine(), workdir=self.root / "wd", levers=[], bandwidth_gbs=1.0)
            with self.assertRaises(Refused) as c:
                t.tune()
        self.assertEqual(c.exception.cause, "on-battery")
        self.assertEqual(c.exception.exit_code, 2)

    def test_refuses_low_power_mode(self):
        with mock.patch.object(sysinfo, "read_power", lambda: {"powerSource": "ac", "powerProfile": "low-power"}):
            with self.assertRaises(Refused) as c:
                Tuner(str(self.profile), FakeEngine(), workdir=self.root / "wd", levers=[]).tune()
        self.assertEqual(c.exception.cause, "low-power-mode")

    def test_refuses_busy_port_gpu_memory(self):
        prof = json.loads(self.profile.read_text())
        prof["port"] = 8999
        self.profile.write_text(json.dumps(prof))
        mk = lambda: Tuner(str(self.profile), FakeEngine(), workdir=self.root / "wd", levers=[])
        with mock.patch.object(sysinfo, "port_in_use", lambda p: True):
            with self.assertRaises(Refused) as c:
                mk().tune()
            self.assertEqual(c.exception.cause, "port-busy")
        with mock.patch.object(sysinfo, "port_in_use", lambda p: False), \
                mock.patch.object(sysinfo, "gpu_utilization_percent", lambda: 90.0):
            with self.assertRaises(Refused) as c:
                mk().tune()
            self.assertEqual(c.exception.cause, "gpu-busy")
        with mock.patch.object(sysinfo, "port_in_use", lambda p: False), \
                mock.patch.object(sysinfo, "gpu_utilization_percent", lambda: 0.0), \
                mock.patch.object(sysinfo, "free_memory_bytes", lambda: 1000):
            with self.assertRaises(Refused) as c:
                mk().tune()
            self.assertEqual(c.exception.cause, "low-memory")

    def test_refuses_when_lock_held(self):
        from ultra_inference_optimization.lock import RunLock
        with RunLock():
            with self.assertRaises(Refused) as c:
                self.tuner([lever("a", "--fast")]).tune()
        self.assertEqual(c.exception.cause, "lock-held")


if __name__ == "__main__":
    unittest.main()


class GpuBusyTests(unittest.TestCase):
    def test_compositor_bursts_do_not_refuse_but_a_real_gpu_job_does(self):
        from unittest import mock
        from ultra_inference_optimization import sysinfo
        bursts = iter([0.0, 39.0, 41.0, 36.0, 36.0])  # a Mac in use: WindowServer/browser bursts (live reading 2026-11-21)
        with mock.patch.object(sysinfo, "gpu_utilization_percent", lambda: next(bursts)), mock.patch.object(sysinfo.time, "sleep", lambda s: None):
            self.assertEqual(sysinfo.check_gpu_idle(), 36.0)
        busy = iter([88.0, 92.0, 90.0, 91.0, 89.0])  # a second model server
        with mock.patch.object(sysinfo, "gpu_utilization_percent", lambda: next(busy)), mock.patch.object(sysinfo.time, "sleep", lambda s: None):
            with self.assertRaises(Exception) as c:
                sysinfo.check_gpu_idle()
            self.assertEqual(c.exception.cause, "gpu-busy")


class LaunchArgsTests(unittest.TestCase):
    def test_baseline_measures_the_managers_launch_flags_and_apply_writes_only_the_extra(self):
        from ultra_inference_optimization import profile as prof
        from ultra_inference_optimization.engines.base import Engine
        launch = ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--cache-ram", "2048"]
        p = {"source": {"path": "/m.gguf"}, "ctx": 4096, "extraArgs": ["--spec-draft-n-max", "5"], "launchArgs": launch}
        self.assertEqual(Engine().baseline_args(p), launch + ["--spec-draft-n-max", "5"])
        self.assertEqual(prof.extra_args_to_write(launch + ["--spec-draft-n-max", "6"], launch), ["--spec-draft-n-max", "6"])
        changed = ["-fa", "on", "-ctk", "f16", "-ctv", "f16", "--cache-ram", "2048", "--spec-draft-n-max", "5"]
        self.assertEqual(prof.extra_args_to_write(changed, launch), changed)  # a changed launch flag is written whole (last value wins)


class CatalogTests(unittest.TestCase):
    HELP = ("-fa, --flash-attn [on|off|auto]\n-ctk, --cache-type-k TYPE\n-ctv, --cache-type-v TYPE\n--spec-draft-n-max N\n"
            "--spec-draft-p-min, --draft-p-min P\n--spec-draft-backend-sampling, --no-spec-draft-backend-sampling\n-ub, --ubatch-size N\n")

    def test_adaptive_draft_levers_appear_with_speculation_and_fa_off_is_skipped_with_quantised_v(self):
        from ultra_inference_optimization.engines import llamacpp
        base = ["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--spec-type", "draft-mtp", "--spec-draft-n-max", "5"]
        names = {lv.name: lv for lv in llamacpp.catalog(self.HELP, base, {"perfCores": 6})}
        self.assertIn("speculative-confidence-cutoff", names)
        self.assertIsNone(names["speculative-confidence-cutoff"].skip_reason)
        self.assertIn("draft-backend-sampling", names)
        self.assertEqual([v.label for v in names["flash-attention"].variants], ["-fa on"])
        no_spec = {lv.name for lv in llamacpp.catalog(self.HELP, ["-fa", "on"], {"perfCores": 6})}
        self.assertNotIn("speculative-confidence-cutoff", no_spec)
