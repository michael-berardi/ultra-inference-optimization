import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import support  # noqa: F401  (puts src on sys.path)
from ultra_inference_optimization import climb, cli, sysinfo
from ultra_inference_optimization.engines import llamacpp
from ultra_inference_optimization.errors import Refused
from ultra_inference_optimization.lock import RunLock
from ultra_inference_optimization.stats import ratio_ci

ROOT = Path(__file__).resolve().parent.parent


class StatsTests(unittest.TestCase):
    def test_ci_clear_win_and_noise(self):
        win = ratio_ci({"m": [13, 13.1, 12.9, 13.2, 13, 12.8]}, {"m": [10, 10.1, 9.9, 10.2, 10, 9.8]}, [("m", True)])
        self.assertGreater(win[1], 1.0)
        noise = ratio_ci({"m": [10, 10.5, 9.6, 10.2, 9.9, 10.1]}, {"m": [10.1, 9.7, 10.4, 10, 10.2, 9.8]}, [("m", True)])
        self.assertLess(noise[1], 1.0)
        self.assertGreater(noise[2], 1.0)

    def test_lower_is_better(self):
        r = ratio_ci({"t": [80, 81, 79, 80, 82, 78]}, {"t": [100, 101, 99, 100, 102, 98]}, [("t", False)])
        self.assertGreater(r[1], 1.0)


class LockTests(unittest.TestCase):
    def test_exclusive_and_released(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "lock"
            with RunLock(p):
                with self.assertRaises(Refused) as c:
                    RunLock(p).__enter__()
                self.assertEqual(c.exception.cause, "lock-held")
            with RunLock(p):
                pass


class GuardTests(unittest.TestCase):
    def test_battery_refused(self):
        with mock.patch.object(sysinfo, "read_power", lambda: {"powerSource": "battery", "powerProfile": "normal"}):
            with self.assertRaises(Refused) as c:
                sysinfo.check_power()
        self.assertEqual(c.exception.cause, "on-battery")

    def test_pmset_parsing(self):
        outs = {("pmset", "-g", "batt"): "Now drawing from 'Battery Power'\n", ("pmset", "-g"): " lowpowermode         1\n"}
        with mock.patch.object(sysinfo, "is_darwin", lambda: True), \
                mock.patch.object(sysinfo, "run", lambda cmd, timeout=10: outs[tuple(cmd)]):
            self.assertEqual(sysinfo.read_power(), {"powerSource": "battery", "powerProfile": "low-power"})

    def test_vm_stat_parsing(self):
        out = "Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free: 10.\nPages inactive: 5.\nPages speculative: 1.\nPages purgeable: 4.\n"
        with mock.patch.object(sysinfo, "is_darwin", lambda: True), mock.patch.object(sysinfo, "run", lambda c, timeout=10: out):
            self.assertEqual(sysinfo.free_memory_bytes(), 20 * 16384)

    def test_watchdog_kills_only_the_pid_we_started(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            with sysinfo.MemoryWatchdog(proc.pid, 100, lambda: llamacpp.stop_process(proc, 5, 5),
                                        interval=0.05, reader=lambda pid: 10_000) as wd:
                proc.wait(timeout=20)
            self.assertTrue(wd.tripped)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


class StopTests(unittest.TestCase):
    def test_sigint_then_sigterm_escalation(self):
        code = "import signal,time; signal.signal(signal.SIGINT, signal.SIG_IGN); time.sleep(60)"
        proc = subprocess.Popen([sys.executable, "-c", code])
        try:
            time.sleep(0.4)
            how = llamacpp.stop_process(proc, sigint_wait=0.5, sigterm_wait=5)
            self.assertEqual(how, "sigterm")
            self.assertIsNotNone(proc.poll())
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


class LeverCatalogTests(unittest.TestCase):
    HELP = "-fa, --flash-attn [on|off|auto]\n-ub, --ubatch-size N\n-b, --batch-size N\n-ctk, --cache-type-k\n-ctv\n-t, --threads\n-tb\n--cache-ram N\n"

    def test_skips_levers_binary_does_not_list(self):
        lv = {l.name: l for l in llamacpp.catalog(self.HELP, ["--spec-draft-n-max", "4"], {"perfCores": 8}, 1000)}
        self.assertIsNone(lv["flash-attention"].skip_reason)
        self.assertIn("--help does not list", lv["speculative-draft-length"].skip_reason)
        self.assertIn("--help does not list", lv["parallel-slots"].skip_reason)
        self.assertFalse(lv["kv-cache-type"].lossless)

    def test_order_and_arg_helpers(self):
        names = [l.name for l in llamacpp.catalog(self.HELP, [], {"perfCores": 8})]
        self.assertEqual(names[:3], ["speculative-draft-length", "flash-attention", "kv-cache-type"])
        self.assertEqual(llamacpp.set_value(["-b", "512", "-x"], ("-b", "--batch-size"), 2048), ["-x", "-b", "2048"])
        self.assertEqual(llamacpp.drop(["--batch-size=1", "-x"], ("-b", "--batch-size")), ["-x"])
        fa = next(l for l in llamacpp.catalog(self.HELP, [], {}) if l.name == "flash-attention")
        self.assertEqual(fa.variants[0].apply([]), ["-fa", "on"])

    def test_spec_lever_needs_draft_source(self):
        h = self.HELP + "--spec-draft-n-max N\n"
        lv = next(l for l in llamacpp.catalog(h, [], {}) if l.name == "speculative-draft-length")
        self.assertIn("no speculative draft source", lv.skip_reason)
        lv = next(l for l in llamacpp.catalog(h, ["-md", "x.gguf"], {}) if l.name == "speculative-draft-length")
        self.assertIsNone(lv.skip_reason)
        self.assertEqual(lv.variants[-1].apply(["-md", "x.gguf", "--spec-draft-n-max", "4"]), [])


class ClimbTests(unittest.TestCase):
    def run_climb(self, metric_fn, params, **kw):
        state = {}

        def run(cmd, timeout=0):
            m = re.match(r"set (.*)", cmd)
            if m:
                state.update(dict(p.split("=") for p in m.group(1).split()))
                return ""
            return str(metric_fn(state))
        with tempfile.TemporaryDirectory() as d:
            c = climb.Climber("measure", climb.parse_params(params), "set a={a} b={b}" if len(params) > 1 else "set a={a}",
                              workdir=d, run=run, **kw)
            return c.climb(), state

    def test_finds_win_and_leaves_best_applied(self):
        n = [0]

        def metric(s):
            n[0] += 1
            return {"1": 10, "2": 14, "3": 12}[s["a"]] + (n[0] % 3) * 0.02
        r, state = self.run_climb(metric, ["a=1,2,3"])
        self.assertEqual(r["bestParams"], {"a": "2"})
        self.assertEqual(state["a"], "2")
        self.assertEqual(len(r["accepted"]), 1)

    def test_minimise_and_plateau(self):
        n = [0]

        def metric(s):
            n[0] += 1
            return 5 + (n[0] % 3) * 0.01
        r, _ = self.run_climb(metric, ["a=1,2", "b=1,2", "c=1,2", "d=1,2"] if False else ["a=1,2,3"], direction="min")
        self.assertEqual(r["accepted"], [])
        self.assertTrue(r["plateau"])

    def test_param_parsing(self):
        self.assertEqual(climb.parse_params(["a=1,2", "b=x"]), {"a": ["1", "2"], "b": ["x"]})
        with self.assertRaises(Refused):
            climb.parse_params(["nonsense"])

    def test_real_shell_commands(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "v"
            c = climb.Climber(f"cat {f}", climb.parse_params(["a=1,5"]), f"echo {{a}} > {f}", workdir=Path(d) / "w")
            r = c.climb()
            self.assertEqual(r["bestParams"], {"a": "5"})


class CliTests(unittest.TestCase):
    def run_cli(self, *argv):
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli.main(list(argv))
        return code, json.loads(buf.getvalue())

    def test_exit_codes_and_json(self):
        code, out = self.run_cli("ceiling", "--gguf", "/nonexistent.gguf", "--bandwidth", "10")
        self.assertEqual(code, 3)
        self.assertEqual(out["error"], "gguf-unreadable")
        code, out = self.run_cli("climb", "--metric-cmd", "echo 1", "--param", "bad", "--apply-cmd", "true")
        self.assertEqual(code, 2)
        code, out = self.run_cli("probe", "--no-measure")
        self.assertEqual(code, 0)
        self.assertIn("machine", out)
        self.assertIn("llama-server", out["engines"])

    def test_ceiling_cli(self):
        from support import tiny_model
        with tempfile.TemporaryDirectory() as d:
            p = tiny_model(Path(d) / "m.gguf")
            code, out = self.run_cli("ceiling", "--gguf", str(p), "--bandwidth", "50", "--ctx", "512")
            self.assertEqual(code, 0)
            self.assertEqual(out["kvBytesPerToken"], 128)
            self.assertTrue(out["arithmetic"])


# The patterns are assembled from fragments so this file does not match its own scan.
def _j(*parts):
    return "".join(parts)


FORBIDDEN = [_j("qw", "en"), _j("gem", "ma"), _j("mis", "tral"), _j("mix", "tral"), _j("deep", "seek"), _j("llama-?", "3"),
             _j("unslo", "th"), _j("bartow", "ski"), _j("thebl", "oke"), _j("hugging", "face"), _j("hf", r"\.co"),
             _j(r"\bq[2-8]", "_k"), _j(r"\biq[1-4]", "_"), _j("mxf", "p4"), _j("ste", "ak"), _j("ultra", "term"), _j("ultra", "bot"),
             _j("liberty", "design"), _j("ber", "ardi"), _j(r"\bmi", r"ke\b"), _j("anthro", "pic"), _j("clau", "de")]


PINNED_URL = "github.com/" + _j("mich", "ael-", "ber", "ardi") + "/ultra-inference-optimization"  # the one place the repository's own address must appear


def scan(text: str) -> list:
    text = text.replace(PINNED_URL, "")
    return [p for p in FORBIDDEN if re.search(p, text, re.I)]


class NeutralityTests(unittest.TestCase):
    def test_scanner_catches_names(self):
        self.assertTrue(scan(_j("model = '", "Qw", "en3-8B-Q4", "_K_M'")))
        self.assertTrue(scan(_j("from hugging", "face.co")))
        self.assertEqual(scan("a plain neutral sentence about decode speed"), [])

    def test_no_model_publisher_quant_person_or_project_names_ship(self):
        # shipped code and the tests alike (tests/ is scanned too)
        files = [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts
                 and p.name != "LICENSE" and "__pycache__" not in p.parts and p.suffix in
                 (".py", ".md", ".sh", ".toml", ".json", ".txt", "")]
        self.assertGreater(len(files), 10)
        self.assertTrue(any("tests" in p.parts for p in files))
        bad = {str(p.relative_to(ROOT)): scan(p.read_text(errors="replace")) for p in files}
        bad = {k: v for k, v in bad.items() if v}
        self.assertEqual(bad, {})


class InstallTests(unittest.TestCase):
    def run_install(self, root, bindir, *extra):
        env = {**os.environ, "UIO_INSTALL_ROOT": str(root), "UIO_BIN_DIR": str(bindir)}
        return subprocess.run(["sh", str(ROOT / "install.sh"), *extra], capture_output=True, text=True, env=env)

    def test_install_shim_works_and_refuses_foreign(self):
        with tempfile.TemporaryDirectory() as d:
            root, bindir = Path(d) / "share", Path(d) / "bin"
            bindir.mkdir()
            foreign = bindir / "uio"
            foreign.write_text("#!/bin/sh\necho foreign\n")
            r = self.run_install(root, bindir)
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertIn("not installed by this script", r.stderr)
            self.assertEqual(foreign.read_text(), "#!/bin/sh\necho foreign\n")
            foreign.unlink()
            r = self.run_install(root, bindir)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            out = subprocess.run([str(bindir / "uio"), "--version"], capture_output=True, text=True, cwd=d)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertIn("uio 0.2.0", out.stdout)
            r = self.run_install(root, bindir)  # re-install over our own shim is fine
            self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
