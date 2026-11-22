"""One test per finding of the release review. Each fails against the pre-fix code."""
import json
import os
import signal
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from support import FakeBehavior, FakeEngine, FakeServer, tiny_model, write_gguf, _s
from test_tune import TuneBase, delay, lever
from ultra_inference_optimization import argmap, bench, cli, experiment, gates, gguf, profile, reaper, sysinfo
from ultra_inference_optimization.engines import llamacpp
from ultra_inference_optimization.engines.base import Lever, Variant
from ultra_inference_optimization.errors import BudgetExceeded, Failed, Refused
from ultra_inference_optimization.stats import Tagged, _resample, _rounds, point_ratio, ratio_ci
from ultra_inference_optimization.tune import Tuner

ROOT = Path(__file__).resolve().parent.parent
DEC = [("decode", True)]


def roles(root):
    return [json.loads(l)["role"] for l in (root / "wd" / "ledger.jsonl").read_text().splitlines()]


class AppliedContract(TuneBase):  # 1, 15
    def test_applied_always_present_with_reason(self):
        r = self.tuner([lever("speed", "--fast")]).tune()  # no --apply
        self.assertEqual(r["applied"], {"written": False, "extraArgs": None, "backup": None, "reason": "--apply not given"})
        self.assertEqual(r["stoppedBecause"], r["stopReason"])
        for k in ("decode", "prefill", "ttft", "args"):
            self.assertIn(k, r["baseline"])
            self.assertIn(k, r["best"])
        self.assertEqual(set(r["confirmation"]), {"ran", "gainRatio", "ci", "passed"})

    def test_nothing_accepted_still_reports_applied(self):
        r = self.tuner([lever("nothing", "--meh")], apply=True).tune()
        self.assertFalse(r["applied"]["written"])
        self.assertIsNone(r["applied"]["extraArgs"])
        self.assertTrue(r["applied"]["reason"])

    def test_applied_is_exactly_what_was_written(self):
        r = self.tuner([lever("speed", "--fast")], apply=True).tune()
        self.assertTrue(r["applied"]["written"])
        self.assertEqual(r["applied"]["extraArgs"], json.loads(self.profile.read_text())["extraArgs"])
        self.assertEqual(r["applied"]["extraArgs"], ["-b", "512", "--fast"])
        self.assertTrue(Path(r["applied"]["backup"]).exists())
        self.assertTrue(r["confirmation"]["ran"] and r["confirmation"]["passed"])
        self.assertGreater(r["confirmation"]["gainRatio"], 1.0)

    def test_profile_changed_during_the_run_is_not_overwritten(self):
        def edit(a):
            d = json.loads(self.profile.read_text())
            d["extraArgs"] = ["-b", "999"]
            self.profile.write_text(json.dumps(d))
            return a + ["--fast"]
        lv = Lever("speed", [Variant("--fast", edit)], True)
        r = self.tuner([lv], apply=True).tune()
        self.assertEqual(r["applied"]["reason"], "profile-changed")
        self.assertFalse(r["applied"]["written"])
        self.assertEqual(json.loads(self.profile.read_text())["extraArgs"], ["-b", "999"])
        self.assertFalse(list(self.root.glob("profile.json.bak-*")))

    def test_write_json_fsyncs_before_rename(self):
        from ultra_inference_optimization.util import write_json
        with mock.patch("os.fsync") as fs:
            write_json(self.root / "x.json", {"a": 1})
        self.assertTrue(fs.called)

    def test_backup_names_are_unique_within_a_second(self):
        with mock.patch.object(profile, "stamp", lambda: "20260101-000000"):
            a = profile.apply_extra_args(str(self.profile), ["-b", "1"])
            b = profile.apply_extra_args(str(self.profile), ["-b", "2"])
        self.assertNotEqual(a["backup"], b["backup"])
        self.assertTrue(Path(a["backup"]).exists() and Path(b["backup"]).exists())


class ConfirmationRound(TuneBase):  # 5c
    def test_gain_that_does_not_reproduce_is_not_applied(self):
        count = {"fast": 0}

        def d(args):
            if "--fast" in args:
                count["fast"] += 1
                if count["fast"] <= 2:  # candidate run and its gate run look fast; the confirmation runs do not
                    return 0.0005
            return 0.0008
        before = self.profile.read_text()
        r = self.tuner([lever("speed", "--fast")], engine=FakeEngine(delay=d), apply=True).tune()
        self.assertEqual(len(r["accepted"]), 1)  # the climb believed it
        self.assertTrue(r["confirmation"]["ran"])
        self.assertFalse(r["confirmation"]["passed"])
        self.assertEqual(r["applied"]["reason"], "confirmation-failed")
        self.assertFalse(r["applied"]["written"])
        self.assertEqual(self.profile.read_text(), before)
        self.assertEqual(roles(self.root)[-4:], ["confirm-control", "confirm-best", "confirm-control", "confirm-best"])

    def test_confirmation_only_runs_for_apply(self):
        r = self.tuner([lever("speed", "--fast")]).tune()
        self.assertEqual(r["confirmation"], {"ran": False, "gainRatio": None, "ci": None, "passed": None})


class SignalsAndChildren(unittest.TestCase):  # 4
    def test_sigterm_becomes_an_exception_so_cleanup_runs(self):
        cleaned = []

        def fake_probe(**kw):
            try:
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(2)
            finally:
                cleaned.append(True)
        before = signal.getsignal(signal.SIGTERM)
        with mock.patch.object(cli.probe, "probe", fake_probe), mock.patch("builtins.print"):
            code = cli.main(["probe", "--no-measure"])
        self.assertEqual(code, 3)
        self.assertEqual(cleaned, [True])
        self.assertEqual(signal.getsignal(signal.SIGTERM), before)

    def test_sighup_too(self):
        def fake_probe(**kw):
            os.kill(os.getpid(), signal.SIGHUP)
            time.sleep(2)
        with mock.patch.object(cli.probe, "probe", fake_probe), mock.patch("builtins.print") as pr:
            self.assertEqual(cli.main(["probe", "--no-measure"]), 3)
        self.assertIn("interrupted", pr.call_args[0][0])


class Reaper(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(os.environ, {"UIO_HOME": self.d.name})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.d.cleanup()

    def test_stale_child_is_stopped_by_pid_and_start_time(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            reaper.record(proc.pid, Path(self.d.name) / "wd")
            self.assertTrue((Path(self.d.name) / "wd" / "child.json").exists())
            self.assertEqual(reaper.reap_stale(grace=5), "terminated")
            proc.wait(timeout=10)
            self.assertFalse(reaper.global_marker().exists())
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_a_reused_pid_with_another_start_time_is_left_alone(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            reaper.record(proc.pid)
            info = json.loads(reaper.global_marker().read_text())
            info["startTime"] = "Mon Jan  1 00:00:00 2001"
            reaper.global_marker().write_text(json.dumps(info))
            self.assertEqual(reaper.reap_stale(grace=1), "gone")
            self.assertIsNone(proc.poll())
        finally:
            proc.kill()
            proc.wait()

    def test_tune_records_and_clears_the_child(self):
        seen = []

        class Eng(FakeEngine):
            def start(s, *a, **k):
                h = super().start(*a, **k)
                seen.append(reaper.global_marker().exists())
                return h
        # pid is set so the recording path is used; the pid is this process, which the reaper never signals
        with TuneBaseHelper() as h:
            h.tuner([lever("a", "--meh")], engine=Eng(pid=os.getpid())).tune()
            self.assertFalse(reaper.global_marker().exists())


class TuneBaseHelper(TuneBase):
    def __enter__(self):
        self.setUp()
        return self

    def __exit__(self, *a):
        self.tearDown()

    def runTest(self):
        pass


class Statistics(unittest.TestCase):  # 5a, 5b, 8
    def test_gain_smaller_than_the_gap_between_controls_is_noise(self):
        c1 = {"decode": [100, 100.5, 99.5, 100.2, 99.8, 100.1] * 3}
        c2 = {"decode": [90, 90.5, 89.5, 90.2, 89.8, 90.1] * 3}
        cand = {"decode": [101.5, 102, 101, 101.8, 101.2, 101.6] * 3}
        d = experiment.decide([c1, c2], cand, DEC, None, min_gain=0.02)
        self.assertGreater(d["controlGap"], 1.1)
        self.assertFalse(d["accept"], d)
        # the same candidate against two agreeing controls is a clear win
        c2b = {"decode": [100, 100.2, 99.9, 100.1, 99.7, 100.3] * 3}
        cand2 = {"decode": [106, 106.5, 105.5, 106.2, 105.8, 106.1] * 3}
        self.assertTrue(experiment.decide([c1, c2b], cand2, DEC, None, min_gain=0.02)["accept"])

    def test_comparison_is_paired_by_prompt(self):
        # the control lost its slow prompt in four rounds: a flat median would call the candidate 45% slower
        ctrl = Tagged([100.0] * 6 + [10.0] * 2, [(i, "a") for i in range(6)] + [(0, "b"), (1, "b")])
        cand = Tagged([100.0] * 6 + [10.0] * 6, [(i, "a") for i in range(6)] + [(i, "b") for i in range(6)])
        self.assertAlmostEqual(point_ratio({"decode": cand}, {"decode": ctrl}, DEC), 1.0, places=6)
        p, lo, hi = ratio_ci({"decode": cand}, {"decode": ctrl}, DEC, iters=200)
        self.assertLessEqual(lo, 1.0 + 1e-9)
        self.assertGreaterEqual(hi, 1.0 - 1e-9)

    def test_bootstrap_resamples_whole_rounds(self):
        import random
        data = {"decode": Tagged([1, 2, 3, 4, 5, 6], [(0, "a"), (0, "b"), (1, "a"), (1, "b"), (2, "a"), (2, "b")])}
        rounds = _rounds(data, ["decode"])
        for seed in range(20):
            out = _resample(rounds, random.Random(seed), ["decode"])["decode"]
            per_round = {}
            for v, (r, p) in zip(out, out.tags):
                per_round.setdefault(r, []).append(p)
            self.assertEqual(len(per_round), 3)
            self.assertTrue(all(sorted(ps) == ["a", "b"] for ps in per_round.values()), per_round)
            # a drawn round brings its values together: (1,2) (3,4) or (5,6)
            pairs = [(out[i], out[i + 1]) for i in range(0, 6, 2)]
            self.assertTrue(all(p in [(1, 2), (3, 4), (5, 6)] for p in pairs), pairs)

    def test_non_target_drop_below_97_percent_rejects_alone(self):
        cand = {"decode": [130, 131, 129, 130, 132, 128], "prefill": [80, 120, 90, 98, 85, 115]}
        ctrl = {"decode": [100, 101, 99, 100, 102, 98], "prefill": [100, 60, 140, 105, 95, 102]}
        d = experiment.decide([ctrl, ctrl], cand, DEC, [("prefill", True)])
        self.assertTrue(d["regressions"], d)
        self.assertEqual(d["verdict"], "rejected-regression")
        self.assertFalse(d["accept"])


class TunerStatistics(TuneBase):  # 5d, early exit, 7, 9
    def test_bench_is_greedy_with_a_seed_and_unique_prefixes(self):
        b = FakeBehavior()
        fs = FakeServer(b)
        try:
            bench.run_bench(fs.base_url, samples=6, seed=7, timeout=20)
        finally:
            fs.close()
        self.assertTrue(all(x["temperature"] == 0.0 and x["seed"] == 7 for x in b.bodies))
        prefixes = [x["messages"][-1]["content"].split("]")[0] for x in b.bodies]
        self.assertEqual(len(set(prefixes)), len(prefixes))
        self.assertTrue(all(p.startswith("[session ") for p in prefixes))

    def test_a_single_control_does_not_end_an_experiment_while_the_gap_is_unknown(self):
        self.tuner([lever("slow", "--slow")]).tune()
        self.assertEqual(roles(self.root)[:3], ["baseline", "candidate", "control"])

    def test_clearly_worse_than_three_control_gaps_exits_early(self):
        t = self.tuner([lever("slow", "--slow")])
        t.control_gap_seen = 1.002
        r = t.tune()
        self.assertNotIn("control", roles(self.root))
        self.assertEqual(r["rejected"][0]["verdict"], "worse")
        self.assertEqual(r["rejected"][0]["controls"], 1)

    def test_plateau_counts_only_levers_whose_every_variant_was_noise(self):
        r = self.tuner([lever(f"s{i}", f"--slow{i}") for i in range(1)] +
                       [Lever(f"w{i}", [Variant("--slow", lambda a: a + ["--slow"])], True) for i in range(4)]).tune()
        self.assertNotEqual(r["stoppedBecause"], "plateau")
        self.assertEqual(len({e["lever"] for e in r["rejected"]}), 5)

    def test_noop_detected_by_normalised_flag_map(self):
        eng = FakeEngine(delay=delay)
        lv = Lever("alias", [Variant("--batch-size 512", lambda a: ["--batch-size", "512"])], True)
        r = self.tuner([lv], engine=eng).tune()
        self.assertEqual((r["accepted"], r["rejected"]), ([], []))
        self.assertEqual(len(eng.starts), 1)  # the baseline only: the "candidate" meant the same thing
        self.assertTrue(argmap.same_args(["-fa", "on", "-ub", "8"], ["--ubatch-size=8", "--flash-attn", "on"]))
        self.assertFalse(argmap.same_args(["-fa", "on"], ["-fa", "off"]))


class ApplyPlanning(TuneBase):  # 6
    LAUNCH = ["-fa", "on", "-ctk", "q8_0", "--cache-ram", "2048"]

    def test_only_the_difference_from_launch_args_is_written(self):
        extra, cause, _ = profile.plan_write(self.LAUNCH + ["--fast"], self.LAUNCH)
        self.assertEqual((extra, cause), (["--fast"], "ok"))
        extra, cause, _ = profile.plan_write(["-fa", "on", "-ctk", "f16", "--cache-ram", "2048"], self.LAUNCH)
        self.assertEqual(extra, ["-ctk", "f16"])
        # launch + written means the same as the tested args
        self.assertTrue(argmap.same_args(self.LAUNCH + extra, ["-fa", "on", "-ctk", "f16", "--cache-ram", "2048"]))

    def test_a_removed_launch_flag_or_a_repeatable_flag_refuses(self):
        extra, cause, _ = profile.plan_write(["-fa", "on", "--cache-ram", "2048"], self.LAUNCH)
        self.assertEqual((extra, cause), (None, "launch-flag-removed"))
        lora = ["--lora-scaled", "a.gguf", "0.5"]
        extra, cause, _ = profile.plan_write(lora + ["--fast"], lora)
        self.assertEqual((extra, cause), (None, "launch-flag-removed"))

    def test_tuner_apply_writes_the_difference_and_refuses_removal(self):
        self.profile.write_text(json.dumps({"source": {"path": str(self.model)}, "ctx": 1500, "extraArgs": []}))
        r = self.tuner([lever("speed", "--fast")], apply=True, launch_args=["-b", "512"]).tune()
        self.assertEqual(r["applied"]["extraArgs"], ["--fast"])
        self.assertEqual(json.loads(self.profile.read_text())["extraArgs"], ["--fast"])
        self.assertEqual(r["best"]["args"], ["-b", "512", "--fast"])

    def test_tuner_refuses_when_the_best_config_drops_a_launch_flag(self):
        self.profile.write_text(json.dumps({"source": {"path": str(self.model)}, "ctx": 1500, "extraArgs": []}))
        lv = Lever("drop", [Variant("drop -b, add --fast", lambda a: ["--fast"])], True)
        before = self.profile.read_text()
        r = self.tuner([lv], apply=True, launch_args=["-b", "512"]).tune()
        self.assertEqual(len(r["accepted"]), 1)
        self.assertEqual(r["applied"], {"written": False, "extraArgs": None, "backup": None, "reason": "launch-flag-removed"})
        self.assertEqual(self.profile.read_text(), before)


class Watchdog(TuneBase):  # 10
    def test_free_system_memory_trips_and_peak_is_recorded(self):
        tripped = []
        wd = sysinfo.MemoryWatchdog(1, None, lambda: tripped.append(1), interval=0.05, reader=lambda p: 5,
                                    min_free_bytes=1000, free_reader=lambda: 10)
        with wd:
            time.sleep(0.3)
        self.assertTrue(wd.tripped)
        self.assertIn("free memory", wd.reason)
        self.assertEqual((wd.peak, wd.min_free, tripped), (5, 10, [1]))

    def test_watchdog_starts_when_the_process_is_spawned_not_when_healthy(self):
        order = []
        eng = llamacpp.LlamaCppEngine.__new__(llamacpp.LlamaCppEngine)
        eng.binary, eng.startup_timeout, eng._help = "/bin/true", 1, ""
        eng.spawn_hook = lambda srv: order.append("hook")
        proc = mock.Mock(pid=4321)
        with mock.patch.object(llamacpp.subprocess, "Popen", return_value=proc), \
                mock.patch.object(llamacpp, "wait_healthy", lambda *a: order.append("healthy")):
            eng.start({"source": {"path": "m"}, "ctx": 8}, [], 1, self.root / "l.log")
        self.assertEqual(order, ["hook", "healthy"])

    def test_peak_memory_lands_in_the_ledger(self):
        eng = FakeEngine(delay=delay, pid=os.getpid())
        with mock.patch.object(sysinfo, "process_memory_bytes", lambda pid: 12345), \
                mock.patch.object(sysinfo, "free_memory_bytes", lambda: 10 ** 12):
            self.tuner([lever("a", "--meh")], engine=eng, mem_cap_bytes=10 ** 12).tune()
        rows = [json.loads(l) for l in (self.root / "wd" / "ledger.jsonl").read_text().splitlines()]
        self.assertEqual(rows[0]["peakMemoryBytes"], 12345)
        self.assertEqual(rows[0]["minSystemFreeBytes"], 10 ** 12)


class Gates(TuneBase):  # 11
    def test_requests_disable_thinking_and_reasoning_text_still_counts(self):
        b = FakeBehavior()
        b.reasoning_only = True
        fs = FakeServer(b)
        try:
            n = gates.needle_gate(fs.base_url, 1500, codes=1, seed=3)
            g = gates.greedy_outputs(fs.base_url)
        finally:
            fs.close()
        self.assertTrue(n["passed"], n)
        self.assertTrue(all(g) and all(x.strip() for x in g))
        for body in b.bodies:
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
            self.assertEqual(body["reasoning_format"], "none")

    def test_empty_output_is_a_failure(self):
        self.assertFalse(gates.compare_greedy(["", "a", "b"], ["", "a", "b"])["equal"])
        self.assertFalse(gates.sanity([""], [])["passed"])

    def test_needle_gate_is_no_worse_than_the_baseline(self):
        r = self.tuner([lever("speed", "--fast")], engine=FakeEngine(delay=delay, needle_ok=lambda a: False)).tune()
        self.assertEqual(len(r["accepted"]), 1, r["rejected"])  # the model cannot find needles at all: not the candidate's fault
        self.assertEqual(r["accepted"][0]["gates"]["needle"]["baseline"]["score"], 0.0)
        self.assertTrue(gates.needle_no_worse({"trials": [1], "score": 1.0, "passed": True}, {"trials": [1], "score": 1.0}))
        self.assertFalse(gates.needle_no_worse({"trials": [1], "score": 0.5, "passed": False}, {"trials": [1], "score": 1.0}))


def hostile(path, meta_bytes, n_kv=1, extra_counts=None):
    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", n_kv) + meta_bytes
    Path(path).write_bytes(head + b"\0" * 64)
    return path


class HostileGguf(unittest.TestCase):  # 12
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.p = Path(self.d.name) / "h.gguf"

    def tearDown(self):
        self.d.cleanup()

    def fails(self):
        with self.assertRaises(Failed):
            gguf.read_gguf(str(self.p))

    def meta(self, **kv):
        m = {"general.architecture": "tiny", "tiny.block_count": 4, "tiny.embedding_length": 64}
        m.update(kv)
        write_gguf(self.p, m, [("output.weight", [4, 4], 1, 64)])

    def test_string_layer_count(self):
        self.meta(**{"tiny.block_count": "many"})
        self.fails()

    def test_float_embedding(self):
        self.meta(**{"tiny.embedding_length": 64.5})
        self.fails()

    def test_layer_count_capped(self):
        self.meta(**{"tiny.block_count": 5000})
        self.fails()
        self.meta(**{"tiny.block_count": 4096})
        self.assertEqual(gguf.read_gguf(str(self.p)).n_layer, 4096)

    def test_non_integer_alignment(self):
        self.meta(**{"general.alignment": 2.5})
        self.fails()
        self.meta(**{"general.alignment": "32"})
        self.fails()
        self.meta(**{"general.alignment": 0})
        self.fails()

    def test_string_in_per_layer_list(self):
        self.meta(**{"tiny.attention.head_count": ["a", "b"]})
        self.fails()

    def test_nesting_deeper_than_two(self):
        inner = struct.pack("<I", 4) + struct.pack("<Q", 1) + struct.pack("<I", 7)
        lvl2 = struct.pack("<I", 9) + struct.pack("<Q", 1) + inner
        lvl3 = struct.pack("<I", 9) + struct.pack("<Q", 1) + lvl2
        meta = _s("x") + struct.pack("<I", 9) + lvl3[0:0] + struct.pack("<I", 9) + struct.pack("<Q", 1) + lvl3
        hostile(self.p, meta)
        self.fails()

    def test_array_count_larger_than_the_file(self):
        meta = _s("x") + struct.pack("<I", 9) + struct.pack("<I", 4) + struct.pack("<Q", 1 << 40)
        hostile(self.p, meta)
        self.fails()

    def test_unknown_array_element_type(self):
        meta = _s("x") + struct.pack("<I", 9) + struct.pack("<I", 99) + struct.pack("<Q", 1)
        hostile(self.p, meta)
        self.fails()


class Ceiling(TuneBase):  # 13
    def profile_with_spec(self):
        self.profile.write_text(json.dumps({"source": {"path": str(self.model)}, "ctx": 1500,
                                            "extraArgs": ["--spec-draft-n-max", "4"]}))

    def test_percent_of_ceiling_uses_the_speculative_ceiling_when_acceptance_was_measured(self):
        self.profile_with_spec()
        r = self.tuner([lever("a", "--meh")], engine=FakeEngine(delay=delay, draft=(10, 6))).tune()
        spec = r["ceiling"]["decodeTokS"]["speculative"]
        self.assertIsNotNone(spec)
        self.assertAlmostEqual(r["percentOfCeiling"], round(100 * r["best"]["decode"] / spec, 1), places=1)

    def test_without_acceptance_it_is_the_weights_only_ceiling(self):
        r = self.tuner([lever("a", "--meh")]).tune()
        self.assertIsNone(r["ceiling"]["decodeTokS"]["speculative"])
        w = r["ceiling"]["decodeTokS"]["weightsOnly"]
        self.assertAlmostEqual(r["percentOfCeiling"], round(100 * r["best"]["decode"] / w, 1), places=1)


class Cli(unittest.TestCase):  # 14
    def test_unexpected_exception_exits_3_with_the_cause(self):
        with mock.patch.object(cli.probe, "probe", side_effect=ZeroDivisionError("oops")), mock.patch("builtins.print") as pr:
            self.assertEqual(cli.main(["probe", "--no-measure"]), 3)
        out = json.loads(pr.call_args[0][0])
        self.assertEqual(out["error"], "unexpected")
        self.assertIn("ZeroDivisionError: oops", out["message"])

    def test_launch_args_must_be_a_json_list_of_strings(self):
        for bad in ('{"a": 1}', '[1, 2]', 'not json', '"-fa"', '[["-fa"]]'):
            with self.assertRaises(Refused, msg=bad) as c:
                cli.parse_launch_args(bad)
            self.assertEqual(c.exception.cause, "bad-launch-args")
        self.assertEqual(cli.parse_launch_args('["-fa", "on"]'), ["-fa", "on"])
        self.assertIsNone(cli.parse_launch_args(None))


class Budget(TuneBase):  # 16
    def test_a_run_started_after_the_deadline_is_refused_and_ends_the_climb(self):
        t = self.tuner([])
        t.deadline = time.monotonic() - 1
        with self.assertRaises(BudgetExceeded):
            t.run_config(["-b", "512"], "candidate")

    def test_budget_ending_inside_a_candidate_stops_the_run_and_discards_it(self):
        holder = {}

        def expire(a):
            holder["t"].deadline = time.monotonic() - 1  # the budget runs out while this experiment is being set up
            return a + ["--fast"]
        t = self.tuner([Lever("x", [Variant("--fast", expire)], True)])
        holder["t"] = t
        r = t.tune()
        self.assertEqual(r["stoppedBecause"], "budget")
        self.assertEqual((r["accepted"], r["rejected"]), ([], []))

    def test_bench_stops_between_rounds_when_the_deadline_passes(self):
        b = FakeBehavior()
        b.tok_delay = 0.01
        fs = FakeServer(b)
        try:
            with self.assertRaises(BudgetExceeded):
                bench.run_bench(fs.base_url, samples=6, timeout=20, deadline=time.monotonic() + 0.05)
        finally:
            fs.close()
        self.assertLess(b.requests, 24)


class Install(unittest.TestCase):  # 3, 17
    def test_refuses_when_not_run_from_a_checkout(self):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run(["sh"], input=(ROOT / "install.sh").read_text(), capture_output=True, text=True, cwd=d,
                               env={**os.environ, "UIO_INSTALL_ROOT": d + "/s", "UIO_BIN_DIR": d + "/b"})
            self.assertEqual(r.returncode, 2, r.stdout)
            self.assertIn("@v0.1.0", r.stderr)
            self.assertFalse((Path(d) / "s").exists())
            lone = Path(d) / "install.sh"
            lone.write_text((ROOT / "install.sh").read_text())
            r = subprocess.run(["sh", str(lone)], capture_output=True, text=True, cwd=d,
                               env={**os.environ, "UIO_INSTALL_ROOT": d + "/s", "UIO_BIN_DIR": d + "/b"})
            self.assertEqual(r.returncode, 2)
            self.assertFalse((Path(d) / "s").exists())

    def test_launcher_runs_isolated_under_an_absolute_python_and_a_hostile_path(self):
        with tempfile.TemporaryDirectory() as d:
            root, bindir = Path(d) / "it's $(touch pwned) a dir", Path(d) / "bin"
            r = subprocess.run(["sh", str(ROOT / "install.sh")], capture_output=True, text=True,
                               env={**os.environ, "UIO_INSTALL_ROOT": str(root), "UIO_BIN_DIR": str(bindir)})
            self.assertEqual(r.returncode, 0, r.stderr)
            shim = (bindir / "uio").read_text()
            self.assertIn(" -I ", shim)
            self.assertTrue(shim.split("exec '")[1].startswith("/"))
            self.assertTrue((root / "0.1.0" / "uio_launcher.py").exists())
            out = subprocess.run([str(bindir / "uio"), "--version"], capture_output=True, text=True, cwd=d)
            self.assertIn("uio 0.1.0", out.stdout, out.stderr)
            self.assertFalse((Path(d) / "pwned").exists())
            self.assertFalse((Path.cwd() / "pwned").exists())


class Docs(unittest.TestCase):  # 3, 8, 13
    def test_readme_matches_the_behaviour(self):
        t = (ROOT / "README.md").read_text()
        self.assertIn("@v0.1.0", t)
        self.assertIn("confirmation", t.lower())
        self.assertIn("flash attention and ubatch size", t)
        self.assertIn("not reached", t)


if __name__ == "__main__":
    unittest.main()


class NeedleNormTests(unittest.TestCase):
    def test_inline_think_block_is_not_part_of_the_answer(self):
        from ultra_inference_optimization import gates
        self.assertEqual(gates._norm("<think>\n\n</think>\n\nsummit-6763"), "summit-6763")
        self.assertEqual(gates._norm("<think>the code is x</think> `lagoon-7759`."), "lagoon-7759")
        self.assertEqual(gates._norm("plain-1234"), "plain-1234")


class CeilingAndBudgetTests(unittest.TestCase):
    def test_needle_gate_stops_at_the_deadline(self):
        import time
        from ultra_inference_optimization import gates
        from ultra_inference_optimization.errors import BudgetExceeded
        with self.assertRaises(BudgetExceeded):
            gates.needle_gate("http://127.0.0.1:9", 4096, deadline=time.monotonic() - 1)

    def test_kv_type_is_read_from_the_tested_args(self):
        from ultra_inference_optimization.tune import Tuner
        self.assertEqual(Tuner._kv_type(["-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"]), "q8_0")
        self.assertEqual(Tuner._kv_type(["-fa", "on"]), "f16")
