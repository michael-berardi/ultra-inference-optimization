"""Update contract v1: check, notice, `uio update`, opt-in automatic installs, install safety and no debris.

No test uses the network: the release record comes from a loopback HTTP server and installers are fakes or the
repository's own install.sh writing into a temporary directory.
"""
import contextlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import support  # noqa: F401  (puts src on sys.path)
from ultra_inference_optimization import __version__, cli, update
from ultra_inference_optimization.errors import Failed, Refused
from ultra_inference_optimization.lock import RunLock

ROOT = Path(__file__).resolve().parent.parent
REAL_SPAWN = update.spawn_check
NOTICE = [f"uio 9.9.9 available (you have {__version__}) · run: uio update",
          "turn on automatic updates: uio update --auto on · skip: uio update --skip"]


def record(tag="v9.9.9", body="first note\nsecond note", **kw):
    """The shape of a GitHub latest-release record."""
    return {"tag_name": tag, "body": body, "draft": False, "prerelease": False, "html_url": "https://example.invalid/r",
            **kw}


class ReleaseServer:
    """A loopback stand-in for the release API; UIO_UPDATE_API points at it while it runs."""

    def __init__(self, rec=None, status=200):
        self.rec, self.status, self.requests = rec or record(), status, []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                outer.requests.append({"path": self.path, "headers": dict(self.headers)})
                body = json.dumps(outer.rec).encode()
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/repos/o/r/releases/latest"

    def __enter__(self):
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        os.environ["UIO_UPDATE_API"] = self.url
        return self

    def __exit__(self, *a):
        os.environ.pop("UIO_UPDATE_API", None)
        self.httpd.shutdown()
        self.httpd.server_close()


class TTY(io.StringIO):
    def isatty(self):
        return True


def run_cli(argv, tty=False, stdin=""):
    out, err = (TTY(), TTY()) if tty else (io.StringIO(), io.StringIO())
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
            mock.patch.object(sys, "stdin", TTY(stdin) if tty else io.StringIO(stdin)):
        code = cli.main(argv)
    return code, out.getvalue(), err.getvalue()


class EnvBase(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.home = Path(td.name)
        env = mock.patch.dict(os.environ, {"UIO_HOME": str(self.home), "NO_COLOR": "1"})
        env.start()
        self.addCleanup(env.stop)
        for key in ("UIO_NO_UPDATE_CHECK", "CI", "UIO_AGENT", "AGENT", "UIO_UPDATE_API"):
            os.environ.pop(key, None)
        spawn = mock.patch.object(update, "spawn_check")
        self.spawn = spawn.start()
        self.addCleanup(spawn.stop)

    def state(self, **kw):
        update.save_state({**update.DEFAULTS, **kw})

    def patch(self, target, name, value):
        p = mock.patch.object(target, name, value)
        p.start()
        self.addCleanup(p.stop)

    def files(self):
        return sorted(p.name for p in self.home.iterdir())


class Versions(unittest.TestCase):
    def test_newer_equal_older_garbage_and_prerelease(self):
        self.assertTrue(update.is_newer("9.9.9"))
        self.assertTrue(update.is_newer("v0.10.0", "0.9.9"))  # numeric, not text order
        self.assertFalse(update.is_newer(__version__))
        self.assertFalse(update.is_newer("0.0.1"))
        for bad in ("garbage", "", None, "1.2", "1.2.3.4", "1.2.3-rc1", "v1.2.3-beta.2", " 1.2.3"):
            self.assertIsNone(update.version_tuple(bad), bad)
            self.assertFalse(update.is_newer(bad), bad)

    def test_only_stable_published_releases_count(self):
        for rec, want in ((record(), "9.9.9"), (record(prerelease=True), None), (record(draft=True), None),
                          (record("nightly"), None), (record("v9.9.9-rc.1"), None), (record("9.9.9"), "9.9.9")):
            with ReleaseServer(rec):
                self.assertEqual(update.fetch_latest()["version"], want, rec)


class Request(unittest.TestCase):
    def test_one_anonymous_get_with_a_fixed_user_agent(self):
        with ReleaseServer() as srv:
            update.fetch_latest()
        self.assertEqual(len(srv.requests), 1)
        req = srv.requests[0]
        self.assertEqual(req["path"], "/repos/o/r/releases/latest")  # no query string
        self.assertEqual(req["headers"]["User-Agent"], "uio-update-check")
        self.assertEqual(set(req["headers"]), {"Host", "User-Agent", "Accept", "Accept-Encoding", "Connection"})
        self.assertNotIn(__version__, json.dumps(req))

    def test_short_timeout_and_bounded_record(self):
        seen = {}

        def urlopen(req, timeout):
            seen["timeout"] = timeout
            return contextlib.nullcontext(io.BytesIO(b"x" * (update.MAX_RECORD + 10)))

        with mock.patch.object(update.urllib.request, "urlopen", urlopen):
            with self.assertRaises(ValueError):
                update.fetch_latest()
        self.assertLessEqual(seen["timeout"], 4.0)

    def test_repository_comes_from_the_package_declaration(self):
        toml = (ROOT / "pyproject.toml").read_text()
        self.assertIn(f'Source = "{update.SOURCE}"', toml)
        self.assertIn(update.SOURCE, (ROOT / "install.sh").read_text())
        slug = update.SOURCE.split("github.com/")[1]
        with mock.patch.dict(os.environ):
            os.environ.pop("UIO_UPDATE_API", None)
            self.assertEqual(update.api_url(), f"https://api.github.com/repos/{slug}/releases/latest")
            os.environ["UIO_UPDATE_API"] = "file:///etc/hosts"
            with self.assertRaises(Refused):
                update.api_url()


class Cache(EnvBase):
    def test_at_most_one_check_per_day_and_never_on_the_startup_path(self):
        with mock.patch.object(update, "spawn_check", REAL_SPAWN), \
                mock.patch.object(update.subprocess, "Popen") as popen, \
                contextlib.redirect_stdout(TTY()), contextlib.redirect_stderr(TTY()):
            update.after_command(True)
            update.after_command(True)
            self.assertEqual(popen.call_count, 1)
            args, kw = popen.call_args
            self.assertTrue(kw["start_new_session"])
            self.assertEqual(args[0][:2], [sys.executable, "-I"])
            st = update.load_state()
            self.assertLess(time.time() - st["checkedAt"], 5)
            self.assertIn("did not finish", st["failure"])  # a check that dies is recorded, not retried
            update.save_state({**st, "checkedAt": time.time() - update.DAY - 1})
            update.after_command(True)
            self.assertEqual(popen.call_count, 2)

    def test_detached_check_fills_the_cache(self):
        with ReleaseServer() as srv:
            t0 = time.time()
            child = REAL_SPAWN(update.load_state())
            self.assertLess(time.time() - t0, 1.0)  # the command does not wait for the request
            self.assertEqual(child.wait(30), 0)
        st = update.load_state()
        self.assertEqual((st["latest"], st["failure"]), ("9.9.9", None))
        self.assertEqual(len(srv.requests), 1)
        self.assertEqual(self.files(), ["update.json"])

    def test_failure_is_recorded_with_its_cause(self):
        def offline():
            raise OSError("network is unreachable")

        st, rel = update.check(offline)
        self.assertIsNone(rel)
        self.assertIn("network is unreachable", update.load_state()["failure"])
        with ReleaseServer(status=500):
            code, out, _ = run_cli(["update", "--check"])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)["error"], "check-failed")
        self.assertIn("500", json.loads(out)["message"])

    def test_state_is_one_small_0600_file(self):
        with ReleaseServer():
            update.check()
        self.assertEqual(self.files(), ["update.json"])
        p = self.home / "update.json"
        self.assertEqual(stat.S_IMODE(p.stat().st_mode), 0o600)
        self.assertLess(p.stat().st_size, 1024)
        self.assertNotIn("first note", p.read_text())  # release notes are not cached


class Silence(EnvBase):
    def test_non_tty_ci_agent_and_opt_outs_are_silent(self):
        self.state(checkedAt=0.0, latest="9.9.9")
        for env in ({"CI": "1"}, {"UIO_AGENT": "1"}, {"AGENT": "1"}, {"UIO_NO_UPDATE_CHECK": "1"}):
            with mock.patch.dict(os.environ, env):
                code, out, err = run_cli(["probe", "--no-measure"], tty=True)
            self.assertEqual((code, err), (0, ""), env)
        code, out, err = run_cli(["probe", "--no-measure"], tty=False)
        self.assertEqual(err, "")
        run_cli(["update", "--check-auto", "off"])
        code, out, err = run_cli(["probe", "--no-measure"], tty=True)
        self.assertEqual(err, "")
        self.spawn.assert_not_called()

    def test_nothing_is_written_outside_a_terminal(self):
        run_cli(["probe", "--no-measure"], tty=False)
        self.assertEqual(self.files(), [])


class Notice(EnvBase):
    def test_after_the_command_on_stderr_once_a_day_never_in_json(self):
        self.state(checkedAt=time.time(), latest="9.9.9")
        code, out, err = run_cli(["probe", "--no-measure"], tty=True)
        self.assertEqual(code, 0)
        self.assertIn("machine", json.loads(out))  # stdout is still exactly one JSON object
        self.assertEqual(err.splitlines(), NOTICE)
        self.assertEqual(run_cli(["probe", "--no-measure"], tty=True)[2], "")
        self.spawn.assert_not_called()

    def test_second_line_is_dim_on_a_colour_terminal(self):
        self.state(latest="9.9.9")
        os.environ.pop("NO_COLOR")
        with mock.patch.dict(os.environ, {"TERM": "xterm-256color"}):
            self.assertEqual(update.notice_lines(update.load_state())[1], f"\033[2m{NOTICE[1]}\033[0m")

    def test_skipped_version_is_silent_and_a_newer_one_notifies(self):
        self.state(checkedAt=time.time(), latest="9.9.9")
        code, out, _ = run_cli(["update", "--skip"])
        self.assertEqual((code, json.loads(out)), (0, {"status": "skipped", "version": "9.9.9"}))
        self.assertEqual(run_cli(["probe", "--no-measure"], tty=True)[2], "")
        update.save_state({**update.load_state(), "latest": "9.9.10"})
        self.assertIn("uio 9.9.10 available", run_cli(["probe", "--no-measure"], tty=True)[2])


class Commands(EnvBase):
    def setUp(self):
        super().setUp()
        self.ran = []
        self.patch(update, "install_kind", lambda: "uv")
        self.patch(update.shutil, "which", lambda name: f"/fake/{name}")
        self.patch(update, "_run", lambda args, **kw: self.ran.append(args))

    def test_check_exit_codes(self):
        for rec, code, status in ((record(), 10, "update-available"), (record("v" + __version__), 0, "up-to-date"),
                                  (record("v0.0.1"), 0, "up-to-date")):
            with ReleaseServer(rec):
                got, out, err = run_cli(["update", "--check"])
            self.assertEqual((got, json.loads(out)["status"]), (code, status))
            self.assertEqual(err, "")
        self.assertEqual(self.ran, [])  # report only

    def test_auto_and_check_switches_persist(self):
        self.assertFalse(update.load_state()["auto"])  # off by default
        self.assertTrue(json.loads(run_cli(["update", "--auto", "on"])[1])["installAutomatically"])
        self.assertTrue(json.loads(run_cli(["update", "--auto", "status"])[1])["installAutomatically"])
        self.assertFalse(json.loads(run_cli(["update", "--auto", "off"])[1])["installAutomatically"])
        self.assertFalse(update.load_state()["auto"])
        self.assertFalse(json.loads(run_cli(["update", "--check-auto", "off"])[1])["checkAutomatically"])
        self.assertFalse(json.loads(run_cli(["update", "--check-auto", "status"])[1])["checkAutomatically"])
        self.assertEqual(self.files(), ["update.json"])

    def test_terminal_asks_to_install_then_once_about_automatic_updates(self):
        with ReleaseServer():
            code, out, err = run_cli(["update"], tty=True, stdin="\nn\n")
        self.assertEqual(code, 0, out)
        self.assertIn(f"uio {__version__} -> 9.9.9\nfirst note\nsecond note", err)
        self.assertIn("Install 9.9.9 now? [Y/n]", err)
        self.assertIn("Install future updates automatically? [y/N]", err)
        self.assertEqual(self.ran, [["/fake/uv", "tool", "install", "--force", f"git+{update.SOURCE}@v9.9.9"]])
        res = json.loads(out)
        self.assertEqual((res["status"], res["installAutomatically"]), ("installed", False))
        self.assertEqual(res["ran"], f"uv tool install --force git+{update.SOURCE}@v9.9.9")  # no local paths
        with ReleaseServer(record("v9.9.10")):
            code, out, err = run_cli(["update"], tty=True, stdin="y\n")
        self.assertNotIn("automatically?", err)  # "n" is remembered
        self.assertFalse(update.load_state()["auto"])

    def test_declining_or_end_of_input_installs_nothing(self):
        for answer in ("n\n", ""):
            with ReleaseServer():
                code, out, _ = run_cli(["update"], tty=True, stdin=answer)
            self.assertEqual((code, json.loads(out)["status"], self.ran), (0, "declined", []), answer)

    def test_without_a_terminal_the_command_is_consent(self):
        with ReleaseServer():
            code, out, err = run_cli(["update"], tty=False)
        self.assertEqual(json.loads(out)["status"], "installed")
        self.assertNotIn("[Y/n]", err)
        self.assertEqual(len(self.ran), 1)
        self.assertEqual(update.load_state()["previous"]["tag"], "v" + __version__)


class Detection(EnvBase):
    def test_install_kinds(self):
        self.assertEqual(update.install_kind(), "source")  # these tests run from the checkout
        with tempfile.TemporaryDirectory() as d:
            pkg = Path(d) / "lib" / "python3" / "site-packages" / "ultra_inference_optimization"
            pkg.mkdir(parents=True)
            with mock.patch.object(update, "__file__", str(pkg / "update.py")), \
                    mock.patch.object(update.sys, "prefix", d):
                self.assertEqual(update.install_kind(), "pip")
                (Path(d) / "pipx_metadata.json").write_text("{}")
                self.assertEqual(update.install_kind(), "pipx")
                (Path(d) / "uv-receipt.toml").write_text("")
                self.assertEqual(update.install_kind(), "uv")

    def test_package_managers_run_their_own_command_or_are_named(self):
        rel = {"tag": "v9.9.9", "version": "9.9.9"}
        with mock.patch.object(update.shutil, "which", lambda name: f"/fake/{name}"):
            for kind, cmd in (("uv", ["/fake/uv", "tool", "install", "--force"]), ("pipx", ["/fake/pipx", "install", "--force"])):
                with mock.patch.object(update, "install_kind", lambda: kind):
                    self.assertEqual(update.install_plan(rel)["args"], cmd + [f"git+{update.SOURCE}@v9.9.9"])
        with mock.patch.object(update, "install_kind", lambda: "pip"):
            with self.assertRaises(Refused) as c:
                update.install_plan(rel)
        self.assertIn(f"pip install --upgrade 'git+{update.SOURCE}@v9.9.9'", c.exception.message)
        with self.assertRaises(Refused) as c:
            update.install_plan(rel)
        self.assertEqual(c.exception.cause, "source-checkout")
        with mock.patch.object(update, "install_kind", lambda: "uv"), mock.patch.object(update.shutil, "which", lambda n: None):
            with self.assertRaises(Refused) as c:
                update.install_plan(rel)
        self.assertIn("run: uv tool install --force", c.exception.message)


def fake_clone(version, install_sh=None):
    """Stands in for `git clone --branch <tag>`: this checkout, reporting `version`."""
    def clone(tag, dest):
        dest.mkdir(parents=True)
        shutil.copy(ROOT / "pyproject.toml", dest)
        shutil.copytree(ROOT / "src", dest / "src", ignore=shutil.ignore_patterns("__pycache__"))
        init = dest / "src" / "ultra_inference_optimization" / "__init__.py"
        init.write_text(init.read_text().replace(f'"{__version__}"', f'"{version}"'))
        (dest / "install.sh").write_text(install_sh or (ROOT / "install.sh").read_text())
    return clone


class InstallSafety(EnvBase):
    def setUp(self):
        super().setUp()
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        base = Path(work.name).resolve()
        self.root, self.bindir, self.tmp = base / "share", base / "bin", base / "tmp"
        self.tmp.mkdir()
        r = subprocess.run(["sh", str(ROOT / "install.sh")], capture_output=True, text=True,
                           env={**os.environ, "UIO_INSTALL_ROOT": str(self.root), "UIO_BIN_DIR": str(self.bindir)})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.shim = self.bindir / "uio"
        self.patch(update, "__file__", str(self.root / __version__ / "ultra_inference_optimization" / "update.py"))
        self.patch(tempfile, "tempdir", str(self.tmp))
        self.patch(update.shutil, "which", lambda n: str(self.shim) if n == "uio" else None)

    def version(self):
        out = subprocess.run([str(self.shim), "--version"], capture_output=True, text=True)
        return out.stdout.strip()

    def test_shim_update_keeps_one_rollback_and_rollback_restores(self):
        old = self.root / "0.0.1"
        old.mkdir()
        (old / "uio_launcher.py").write_text("")
        self.assertEqual(update.install_kind(), "shim")
        plan = update.install_plan({"tag": "v9.9.9", "version": "9.9.9"})
        self.assertEqual((plan["root"], plan["bin"]), (str(self.root), str(self.bindir)))
        with mock.patch.object(update, "clone", fake_clone("9.9.9")):
            update.run_install(plan)
        self.assertEqual(self.version(), "uio 9.9.9")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), sorted([__version__, "9.9.9"]))
        self.assertEqual(list(self.tmp.iterdir()), [])  # the clone's temp dir is gone
        res = update.rollback()
        self.assertEqual(res, {"status": "rolled-back", "version": __version__, "skipped": "9.9.9"})
        self.assertEqual(self.version(), f"uio {__version__}")
        self.assertEqual(update.load_state()["skip"], "9.9.9")

    def test_a_tag_that_does_not_verify_is_refused_before_the_swap(self):
        before = self.shim.read_text()
        with mock.patch.object(update, "clone", fake_clone("6.6.6")):
            with self.assertRaises(Failed) as c:
                update.run_install(update.install_plan({"tag": "v9.9.9", "version": "9.9.9"}))
        self.assertIn("reports version 6.6.6", c.exception.message)
        self.assertEqual(self.shim.read_text(), before)
        self.assertFalse((self.root / "9.9.9").exists())
        self.assertEqual(list(self.tmp.iterdir()), [])
        self.assertIsNone(update.load_state()["previous"])

    def test_a_failed_install_puts_the_running_version_back(self):
        broken = ('#!/bin/sh\nmkdir -p "$UIO_INSTALL_ROOT/9.9.9/ultra_inference_optimization"\n'
                  'printf broken > "$UIO_BIN_DIR/uio"\nexit 1\n')
        with mock.patch.object(update, "clone", fake_clone("9.9.9", broken)):
            with self.assertRaises(Failed):
                update.run_install(update.install_plan({"tag": "v9.9.9", "version": "9.9.9"}))
        self.assertEqual(self.version(), f"uio {__version__}")
        self.assertFalse((self.root / "9.9.9").exists())
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_a_failed_package_manager_install_reinstalls_the_running_tag(self):
        calls = []

        def run(args, **kw):
            calls.append(args[-1])
            if args[-1].endswith("@v9.9.9"):
                raise Failed("install-failed", "uv exited 1: build failed")

        plan = {"kind": "uv", "tag": "v9.9.9", "version": "9.9.9", "display": "uv",
                "args": ["/fake/uv", "tool", "install", "--force", f"git+{update.SOURCE}@v9.9.9"]}
        with mock.patch.object(update, "_run", run):
            with self.assertRaises(Failed) as c:
                update.run_install(plan)
        self.assertEqual(calls, [f"git+{update.SOURCE}@v9.9.9", f"git+{update.SOURCE}@v{__version__}"])
        self.assertIn("build failed", c.exception.message)
        self.assertIsNone(update.load_state()["previous"])

    def test_run_reports_the_exit_code_and_output_tail(self):
        with self.assertRaises(Failed) as c:
            update._run([sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"])
        self.assertIn("exited 3: boom", c.exception.message)


class NoDebris(EnvBase):
    def rollback_layout(self, to, age=0.0):
        root = self.home / "share"
        (root / "0.0.1").mkdir(parents=True)
        (root / "0.0.1" / "uio_launcher.py").write_text("")
        running = root / __version__ / "ultra_inference_optimization"
        running.mkdir(parents=True)
        self.state(previous={"kind": "shim", "tag": "v0.0.1", "version": "0.0.1", "dir": str(root / "0.0.1"),
                             "root": str(root.resolve()), "bin": str(self.home), "to": to, "at": time.time() - age})
        return root, mock.patch.object(update, "__file__", str(running / "update.py"))

    def test_rollback_folder_goes_after_the_new_version_ran_once(self):
        root, here = self.rollback_layout(to=__version__)
        with here:
            update.after_command(True)
        self.assertFalse((root / "0.0.1").exists())
        self.assertIsNone(update.load_state()["previous"])

    def test_rollback_folder_stays_until_then(self):
        root, here = self.rollback_layout(to="9.9.9")
        with here:
            update.after_command(False)  # a failed run does not count
            update.after_command(True)  # neither does a run of another version
        self.assertTrue((root / "0.0.1").exists())

    def test_rollback_folder_goes_after_seven_days_at_most(self):
        root, here = self.rollback_layout(to="9.9.9", age=update.KEEP_ROLLBACK + 1)
        with here:
            update.after_command(True)
        self.assertFalse((root / "0.0.1").exists())
        self.assertTrue((root / __version__).exists())  # the running release is never removed

    def test_background_log_is_one_bounded_file(self):
        (self.home / update.LOG).write_bytes(b"x" * (update.MAX_LOG + 100))
        for i in range(3):
            update.log(f"line {i}")
        data = (self.home / update.LOG).read_bytes()
        self.assertLessEqual(len(data), update.MAX_LOG)
        self.assertTrue(data.endswith(b"line 2\n"))
        self.assertEqual(self.files(), [update.LOG])
        self.assertEqual(stat.S_IMODE((self.home / update.LOG).stat().st_mode), 0o600)

    def test_clear_state_removes_the_cache_and_the_log(self):
        self.state(latest="9.9.9")
        update.log("entry")
        self.assertEqual(json.loads(run_cli(["update", "--clear-state"])[1])["status"], "state-cleared")
        self.assertEqual(self.files(), [])


class AutoInstall(EnvBase):
    def setUp(self):
        super().setUp()
        self.installed = []
        self.patch(update, "install_kind", lambda: "uv")
        self.patch(update.shutil, "which", lambda n: f"/fake/{n}")

    def fetch(self, version="9.9.9"):
        return lambda: {"version": version, "tag": "v" + version, "notes": ""}

    def test_off_by_default(self):
        update.background(self.fetch(), install=self.installed.append)
        self.assertEqual(self.installed, [])

    def test_on_installs_in_the_background_and_logs(self):
        self.state(auto=True)
        update.background(self.fetch(), install=self.installed.append)
        self.assertEqual([p["tag"] for p in self.installed], ["v9.9.9"])
        self.assertIn("automatic install of 9.9.9 done", (self.home / update.LOG).read_text())

    def test_a_failure_records_the_cause_falls_back_to_the_notice_and_is_not_retried(self):
        self.state(auto=True, checkedAt=time.time(), latest="9.9.9")
        self.assertFalse(update.show_notice(update.load_state(), err=io.StringIO()))  # auto is handling it

        def fail(plan):
            self.installed.append(plan)
            raise Failed("install-failed", "uv exited 1: offline")

        update.background(self.fetch(), install=fail)
        st = update.load_state()
        self.assertEqual(st["autoFailed"], {"version": "9.9.9", "cause": "uv exited 1: offline"})
        err = io.StringIO()
        self.assertTrue(update.show_notice(st, err=err))
        self.assertEqual(err.getvalue().splitlines()[0], NOTICE[0])
        update.background(self.fetch(), install=fail)
        self.assertEqual(len(self.installed), 1)
        update.background(self.fetch("9.9.10"), install=self.installed.append)  # a newer release tries again
        self.assertEqual(len(self.installed), 2)

    def test_waits_while_a_uio_run_holds_the_lock(self):
        self.state(auto=True)
        with RunLock(self.home / "lock"):
            update.background(self.fetch(), install=self.installed.append)
        self.assertEqual(self.installed, [])
        self.assertIn("deferred", (self.home / update.LOG).read_text())
        self.assertIsNone(update.load_state()["autoFailed"])


class Docs(unittest.TestCase):
    def section(self, text, title):
        body = text.split(f"\n## {title}\n", 1)[1].split("\n## ", 1)[0]
        return [line for line in body.strip().splitlines() if line.strip()]

    def test_marker_sections_and_versions(self):
        self.assertIn("<!-- update-contract: v1 -->", (ROOT / "docs" / "updates.md").read_text().splitlines())
        readme = (ROOT / "README.md").read_text()
        self.assertLessEqual(len(self.section(readme, "Updates")), 5)
        privacy = self.section(readme, "Privacy")
        self.assertTrue(3 <= len(privacy) <= 6, privacy)
        self.assertIn("PRIVACY.md", " ".join(privacy))
        self.assertIn("uio-update-check", (ROOT / "PRIVACY.md").read_text())
        self.assertIn(f'version = "{__version__}"', (ROOT / "pyproject.toml").read_text())
        for name in ("README.md", "install.sh"):
            self.assertIn(f"@v{__version__}", (ROOT / name).read_text())
        self.assertIn(f"## {__version__} ", (ROOT / "CHANGELOG.md").read_text())
        self.assertRegex(update.__doc__, re.compile("anonymous GET", re.I))


if __name__ == "__main__":
    unittest.main()
