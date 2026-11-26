"""Privacy regressions: no telemetry, no network outside the server you supply, private local state."""
import io
import json
import os
import socket
import stat
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import support  # noqa: F401  (puts src on sys.path)
from support import FakeBehavior, FakeServer, tiny_model
from ultra_inference_optimization import bench, cli, climb, profile, util
from ultra_inference_optimization.engines import llamacpp
from ultra_inference_optimization.lock import RunLock

ROOT = Path(__file__).resolve().parent.parent


# Fragments keep this test file from matching its own source scan.
def _frag(*parts):
    return "".join(parts)


PACKAGES = [_frag("sentry", "_sdk"), _frag("post", "hog"), "segment", "mixpanel", "amplitude", "datadog",
            _frag("open", "telemetry"), _frag("hugging", "face_hub"), "gradio", "analytics"]
ENDPOINTS = [_frag("sentry", ".io"), _frag("posthog", ".com"), _frag("segment", ".io"), _frag("amplitude", ".com"),
             _frag("mixpanel", ".com"), _frag("hugging", "face", ".co"), _frag("hf", ".co")]


class NoTelemetry(unittest.TestCase):
    def test_no_telemetry_dependency_or_endpoint_in_shipped_code(self):
        files = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "pyproject.toml", ROOT / "install.sh"]
        text = "\n".join(p.read_text(errors="replace") for p in files)
        for name in PACKAGES + ENDPOINTS:
            self.assertNotIn(name, text, name)
        self.assertIn("dependencies = []", (ROOT / "pyproject.toml").read_text())

    def test_offline_commands_make_no_network_call(self):
        with tempfile.TemporaryDirectory() as d:
            model = tiny_model(Path(d) / "m.gguf")
            calls = []
            real_connect = socket.socket.connect

            def record(self, address, *a, **kw):
                calls.append(address)
                return real_connect(self, address, *a, **kw)

            with mock.patch.object(socket.socket, "connect", record):
                code, out = self._cli("probe", "--no-measure")
                self.assertEqual((code, out.get("error")), (0, None))
                code, out = self._cli("ceiling", "--gguf", str(model), "--bandwidth", "50")
                self.assertEqual((code, out.get("error")), (0, None))
            self.assertEqual(calls, [])

    def test_benchmark_talks_only_to_the_supplied_server(self):
        behavior = FakeBehavior()
        server = FakeServer(behavior)
        targets = []
        real_connect = socket.socket.connect

        def record(self, address, *a, **kw):
            targets.append(address)
            return real_connect(self, address, *a, **kw)

        try:
            with mock.patch.object(socket.socket, "connect", record):
                res = bench.run_bench(server.base_url, samples=6, timeout=20)
        finally:
            server.close()
        self.assertTrue(res["decodeTokS"]["samples"])
        self.assertEqual(len(targets), behavior.requests + 2)  # chat POSTs plus two /metrics GETs
        for target in targets:
            host = target[0] if isinstance(target, tuple) else target
            self.assertIn(host, ("127.0.0.1", "::1", "localhost"), target)

    def test_readme_privacy_is_short_and_privacy_md_has_the_details(self):
        text = (ROOT / "README.md").read_text()
        self.assertIn("## Privacy", text)
        section = text.split("## Privacy", 1)[1].split("\n## ", 1)[0]
        lines = [x for x in section.splitlines() if x.strip()]
        self.assertLessEqual(len(lines), 6, lines)
        for phrase in ("no telemetry", "~/.cache/uio", "rm -rf ~/.cache/uio", "release check", "UIO_NO_UPDATE_CHECK=1",
                       "PRIVACY.md"):
            self.assertIn(phrase, section.lower() if phrase == "no telemetry" else section, phrase)
        self.assertNotIn("no update check", text.lower())
        details = (ROOT / "PRIVACY.md").read_text()
        for phrase in ("no telemetry", "127.0.0.1", "--base-url", "--panel FILE", "uio-update-check",
                       "at most once per 24 hours", "UIO_NO_UPDATE_CHECK=1", "~/.cache/uio", "0600", "0700",
                       "--workdir", "newest ten run folders", "*.bak-*", "rm -rf ~/.cache/uio", "DO_NOT_TRACK=1",
                       "update --clear-state", "256 KB"):
            self.assertIn(phrase, details, phrase)
        self.assertNotIn("no update check", details.lower())

    @staticmethod
    def _cli(*argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = cli.main(list(argv))
        return code, json.loads(buf.getvalue())


class PrivateLocalState(unittest.TestCase):
    def test_files_and_directories_are_owner_only(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"UIO_HOME": os.path.join(d, "home")}):
                base = util.uio_home()
                folder = base / "runs" / "x"
                util.write_json(folder / "a.json", {"x": 1})
                util.append_jsonl(folder / "l.jsonl", {"x": 1})
                util.append_text(folder / "l.txt", "x")
                with RunLock():
                    self.assertEqual(stat.S_IMODE((base / "lock").stat().st_mode), 0o600)
                for p in (base, base / "runs", folder):
                    self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o700, p)
                for name in ("a.json", "l.jsonl", "l.txt"):
                    self.assertEqual(stat.S_IMODE((folder / name).stat().st_mode), 0o600, name)

    def test_server_log_is_owner_only(self):
        with tempfile.TemporaryDirectory() as d:
            eng = llamacpp.LlamaCppEngine.__new__(llamacpp.LlamaCppEngine)
            eng.binary, eng.startup_timeout, eng._help = "/bin/true", 1, ""
            log = Path(d) / "logs" / "1.log"
            with mock.patch.object(llamacpp.subprocess, "Popen", return_value=mock.Mock(pid=4321)), \
                    mock.patch.object(llamacpp, "wait_healthy", lambda *a: None):
                server = eng.start({"source": {"path": "m"}, "ctx": 8}, [], 1, log)
                server.stop()
            self.assertEqual(stat.S_IMODE(log.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(log.parent.stat().st_mode), 0o700)

    def test_default_run_retention_is_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"UIO_HOME": d}):
                runs = util.uio_home() / "runs"
                runs.mkdir(parents=True)
                made = []
                for i in range(12):
                    p = runs / f"old-{i}"
                    p.mkdir()
                    (p / "ledger.jsonl").write_text("x")
                    t = time.time() - (12 - i) * 60
                    os.utime(p, (t, t))
                    made.append(p)
                removed = util.prune_runs()
                self.assertEqual(set(removed), set(made[:2]))
                self.assertFalse(any(p.exists() for p in made[:2]))
                self.assertTrue(all(p.exists() for p in made[2:]))


class DirectoryOwnership(unittest.TestCase):
    """private_mkdir must not tighten directories the user already owns outside UIO_HOME."""

    def test_existing_workdir_outside_uio_home_keeps_0755(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with mock.patch.dict(os.environ, {"UIO_HOME": str(root / "home")}):
                workdir = root / "project"
                workdir.mkdir(mode=0o755)
                os.chmod(workdir, 0o755)

                # Direct call, then the path a CLI --workdir takes through the runner.
                util.private_mkdir(workdir)
                self.assertEqual(stat.S_IMODE(workdir.stat().st_mode), 0o755)

                values = iter(range(1, 10000))

                def run(_cmd, timeout=0):
                    return str(next(values) / 1000.0)

                c = climb.Climber("measure", climb.parse_params(["a=1,2"]), "set a={a}",
                                  workdir=workdir, run=run)
                c.climb()
                self.assertEqual(stat.S_IMODE(workdir.stat().st_mode), 0o755)

    def test_existing_profile_parent_keeps_0755_and_profile_is_0600(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with mock.patch.dict(os.environ, {"UIO_HOME": str(root / "home")}):
                folder = root / "profiles"
                folder.mkdir(mode=0o755)
                os.chmod(folder, 0o755)
                p = folder / "model.json"
                p.write_text(json.dumps({"source": {"path": "m"}, "ctx": 8, "extraArgs": []}))
                os.chmod(p, 0o644)

                result = profile.apply_extra_args(str(p), ["-b", "1024"])
                self.assertTrue(result["written"])
                self.assertEqual(stat.S_IMODE(folder.stat().st_mode), 0o755)
                self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o600)
                self.assertEqual(stat.S_IMODE(Path(result["backup"]).stat().st_mode), 0o600)

    def test_new_directories_outside_uio_home_are_0700(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with mock.patch.dict(os.environ, {"UIO_HOME": str(root / "home")}):
                first, second, leaf = root / "outside", root / "outside" / "deep", root / "outside" / "deep" / "new"
                util.private_mkdir(leaf)
                for p in (first, second, leaf):
                    self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o700, p)

    def test_existing_uio_home_and_subtree_are_0700(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            home, runs, leaf = root / "home", root / "home" / "runs", root / "home" / "runs" / "x"
            for p in (home, runs, leaf):
                p.mkdir(mode=0o755)
                os.chmod(p, 0o755)
            with mock.patch.dict(os.environ, {"UIO_HOME": str(home)}):
                util.private_mkdir(leaf)
            for p in (home, runs, leaf):
                self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o700, p)

    def test_ancestors_above_uio_home_are_left_alone(self):
        with tempfile.TemporaryDirectory() as d:
            mid = Path(d) / "mid"
            mid.mkdir(mode=0o755)
            os.chmod(mid, 0o755)
            home = mid / "home"
            with mock.patch.dict(os.environ, {"UIO_HOME": str(home)}):
                util.private_mkdir(home / "runs" / "x")
            self.assertEqual(stat.S_IMODE(mid.stat().st_mode), 0o755)
            for p in (home, home / "runs", home / "runs" / "x"):
                self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o700, p)


if __name__ == "__main__":
    unittest.main()
