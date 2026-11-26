"""Release check, update notice and opt-in automatic updates (update contract v1), standard library only.

Without an explicit `uio update`, the only network use is one anonymous GET of the public latest-release record, at
most once per 24 hours, made by a detached background process and only after a command run from an interactive
terminal (not CI, not an agent, not opted out). State is one small 0600 file; background installs log to one file
capped at 256 KB.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from . import __version__
from .errors import Failed, Refused, UioError
from .util import private_mkdir, uio_home, write_json

TOOL = "uio"
SOURCE = "https://github.com/michael-berardi/ultra-inference-optimization"  # the same as [project.urls] Source
USER_AGENT = "uio-update-check"
DAY = 24 * 60 * 60
KEEP_ROLLBACK = 7 * DAY
TIMEOUT = 4.0
MAX_RECORD = 1024 * 1024
MAX_LOG = 256 * 1024
STATE = "update.json"
LOG = "update.log"
SHIM_MARKER = "# uio-shim: managed by ultra-inference-optimization install.sh"
_VERSION = re.compile(r"v?(\d+)\.(\d+)\.(\d+)")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
DEFAULTS = {"checkedAt": 0.0, "latest": None, "failure": None, "skip": None, "auto": False,
            "checkAutomatically": True, "askedAuto": False, "shownAt": 0.0, "shownVersion": None,
            "autoFailed": None, "previous": None}


class Report(Exception):
    """A complete `uio update` answer with a non-zero exit code (10 update available, 1 check failed)."""

    def __init__(self, payload: dict, code: int):
        super().__init__(payload)
        self.payload, self.code = payload, code


def version_tuple(text):
    """Only a stable x.y.z (optional leading v) is a version; anything else is not newer."""
    m = _VERSION.fullmatch(str(text or ""))
    return tuple(int(x) for x in m.groups()) if m else None


def is_newer(latest, current=__version__) -> bool:
    a, b = version_tuple(latest), version_tuple(current)
    return bool(a and b and a > b)


def _cause(e) -> str:
    text = e.message if isinstance(e, UioError) else f"{type(e).__name__}: {e}"
    return _CONTROL.sub("", text)[:300]


# ---- state: one small 0600 file -----------------------------------------------------------------------------------
def load_state() -> dict:
    st = dict(DEFAULTS)
    try:
        saved = json.loads((uio_home() / STATE).read_text())
        st.update({k: v for k, v in saved.items()
                   if k in DEFAULTS and (DEFAULTS[k] is None or type(v) is type(DEFAULTS[k]))})
    except (OSError, ValueError, AttributeError):
        pass  # a missing or damaged cache is never a reason to make noise
    return st


def save_state(st: dict) -> None:
    write_json(uio_home() / STATE, st, 0o600)


def clear_state() -> None:
    for name in (STATE, LOG):
        with contextlib.suppress(FileNotFoundError):
            (uio_home() / name).unlink()


def log(message: str) -> None:
    """One log file for background installs, cut to its last 256 KB."""
    path = uio_home() / LOG
    try:
        old = path.read_bytes()
    except OSError:
        old = b""
    line = time.strftime("%Y-%m-%dT%H:%M:%S ") + _CONTROL.sub(" ", message)[:2000] + "\n"
    private_mkdir(path.parent)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as fh:
        fh.write((old + line.encode())[-MAX_LOG:])
    os.replace(tmp, path)


# ---- check ---------------------------------------------------------------------------------------------------------
def api_url() -> str:
    override = os.environ.get("UIO_UPDATE_API", "").strip()  # tests and manual proofs only
    if override:
        if not override.startswith(("http://", "https://")):
            raise Refused("bad-update-api", "UIO_UPDATE_API must be an http(s) URL")
        return override
    return "https://api.github.com/repos/" + SOURCE.split("github.com/", 1)[1] + "/releases/latest"


def fetch_latest() -> dict:
    """One anonymous GET with a fixed User-Agent: no id, version, machine or usage data is sent."""
    req = urllib.request.Request(api_url(), headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        raw = r.read(MAX_RECORD + 1)
    if len(raw) > MAX_RECORD:
        raise ValueError("the release record is too large")
    rec = json.loads(raw)
    if not isinstance(rec, dict):
        raise ValueError("the release record is not an object")
    tag = str(rec.get("tag_name") or "")
    ver = version_tuple(tag)
    stable = bool(ver) and not rec.get("draft") and not rec.get("prerelease")
    return {"version": ".".join(map(str, ver)) if stable else None, "tag": tag if stable else None,
            "notes": str(rec.get("body") or "")}


def check(fetch=None):
    """Fetch now and record the result, or the failure with its cause, in the cache."""
    st = load_state()
    st["checkedAt"] = time.time()
    try:
        rel = (fetch or fetch_latest)()
        st["latest"], st["failure"] = rel["version"], None
    except Exception as e:
        rel, st["failure"] = None, _cause(e)
    save_state(st)
    return st, rel


def available(st: dict) -> bool:
    return is_newer(st.get("latest")) and st.get("latest") != st.get("skip")


def checks_enabled() -> bool:
    """Automatic checks and notices: interactive terminals only, never in CI, agents or after an opt-out."""
    env = os.environ
    if env.get("UIO_NO_UPDATE_CHECK", "") not in ("", "0") or env.get("CI") \
            or env.get("UIO_AGENT", "") not in ("", "0") or env.get("AGENT", "") not in ("", "0"):
        return False
    try:
        return sys.stdout.isatty() and sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


def notice_lines(st: dict) -> list:
    first = f"{TOOL} {st['latest']} available (you have {__version__}) · run: {TOOL} update"
    second = f"skip: {TOOL} update --skip" if st.get("auto") else \
        f"turn on automatic updates: {TOOL} update --auto on · skip: {TOOL} update --skip"
    if not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb":
        second = f"\033[2m{second}\033[0m"
    return [first, second]


def show_notice(st: dict, err=None) -> bool:
    """Two stderr lines, at most once per 24 h per version; silent while an automatic install is handling it."""
    latest = st.get("latest")
    if not available(st) or (st.get("auto") and not _auto_failed_for(st, latest)):
        return False
    if st.get("shownVersion") == latest and 0 <= time.time() - st.get("shownAt", 0.0) < DAY:
        return False
    print("\n".join(notice_lines(st)), file=err or sys.stderr)
    st.update(shownAt=time.time(), shownVersion=latest)
    save_state(st)
    return True


def _auto_failed_for(st: dict, version) -> bool:
    return isinstance(st.get("autoFailed"), dict) and st["autoFailed"].get("version") == version


def _due(st: dict) -> bool:
    return not 0 <= time.time() - st.get("checkedAt", 0.0) < DAY


def spawn_check(st: dict):
    """Start the daily check as a detached, low-priority process; this command never waits for it."""
    # Recorded first: a check that dies or hangs is a failure with a cause, not a retry on every run.
    st.update(checkedAt=time.time(), failure="the background check did not finish")
    save_state(st)
    code = ("import sys; sys.path.insert(0, sys.argv[1]); "
            "from ultra_inference_optimization import update; update.background()")
    return subprocess.Popen([sys.executable, "-I", "-c", code, str(Path(__file__).resolve().parents[1])],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True, close_fds=True)


def after_command(succeeded: bool) -> None:
    """Called once a command has printed its result. Never raises, never blocks on the network."""
    try:
        if succeeded:
            confirm_new_version_ran()
        if not checks_enabled():
            return
        st = load_state()
        if not st["checkAutomatically"]:
            return
        show_notice(st)
        if _due(st):
            spawn_check(st)
    except Exception:
        pass  # best effort: the update check never adds noise to a command's output


def background(fetch=None, install=None) -> None:
    """The detached daily check. With automatic updates on, it also installs a new release."""
    with contextlib.suppress(AttributeError, OSError):
        os.nice(10)
    st, rel = check(fetch)
    if rel and st["auto"] and available(st) and not _auto_failed_for(st, rel["version"]):
        auto_install(rel, install)


def auto_install(rel: dict, install=None) -> None:
    """Opt-in only. A failure leaves the running version in place, records the cause and brings the notice back."""
    from .lock import RunLock  # the lock that `uio tune` holds; never swap files under a running job
    lock = uio_home() / "lock"
    try:
        with RunLock(lock) if lock.exists() else contextlib.nullcontext():
            (install or run_install)(install_plan(rel))
    except Refused as e:
        if e.cause == "lock-held":
            log(f"automatic install of {rel['version']} deferred: a uio run is active")
            return
        _record_auto_failure(rel, e)
    except Exception as e:
        _record_auto_failure(rel, e)
    else:
        log(f"automatic install of {rel['version']} done; it applies from the next start")


def _record_auto_failure(rel: dict, e) -> None:
    st = load_state()
    st["autoFailed"] = {"version": rel["version"], "cause": _cause(e)}
    save_state(st)
    log(f"automatic install of {rel['version']} failed: {_cause(e)}")


# ---- install -------------------------------------------------------------------------------------------------------
def install_kind() -> str:
    pkg = Path(__file__).resolve().parent
    if (pkg.parent / "uio_launcher.py").is_file():
        return "shim"  # install.sh layout: <root>/<version>/{uio_launcher.py,ultra_inference_optimization}
    if (pkg.parents[1] / "pyproject.toml").is_file():
        return "source"  # running from a checkout (editable install or src on the path)
    if (Path(sys.prefix) / "uv-receipt.toml").is_file():
        return "uv"
    if (Path(sys.prefix) / "pipx_metadata.json").is_file():
        return "pipx"
    return "pip"


def _manager_command(kind: str, tag: str) -> list:
    spec = f"git+{SOURCE}@{tag}"
    return ["uv", "tool", "install", "--force", spec] if kind == "uv" else ["pipx", "install", "--force", spec]


def _shim_launcher(text: str):
    m = re.search(r"-I '((?:[^']|'\\'')*)' \"\$@\"", text)
    return Path(m.group(1).replace("'\\''", "'")) if m else None


def find_shim():
    """The install.sh shim that runs this copy (on PATH or in ~/.local/bin)."""
    here = Path(__file__).resolve().parents[1]
    for cand in dict.fromkeys(filter(None, [shutil.which(TOOL), str(Path.home() / ".local" / "bin" / TOOL)])):
        try:
            text = Path(cand).read_text()
        except (OSError, UnicodeDecodeError):
            continue
        launcher = _shim_launcher(text) if SHIM_MARKER in text else None
        if launcher and launcher.resolve().parent == here:
            return {"bin": str(Path(cand).parent), "root": str(here.parent), "dir": str(here)}
    return None


def install_plan(rel: dict) -> dict:
    """How this install kind is updated. Package managers run their own command; nothing they own is overwritten."""
    tag, version, kind = rel["tag"], rel["version"], install_kind()
    if not version_tuple(tag):
        raise Refused("bad-tag", "the release tag is not a stable x.y.z version")
    plan = {"kind": kind, "tag": tag, "version": version}
    if kind in ("uv", "pipx"):
        cmd = _manager_command(kind, tag)
        exe = shutil.which(cmd[0])
        if not exe:
            raise Refused(f"{kind}-not-found", f"{cmd[0]} is not on PATH; run: {' '.join(cmd)}")
        return {**plan, "args": [exe] + cmd[1:], "display": " ".join(cmd)}
    if kind == "shim":
        shim = find_shim()
        if not shim:
            raise Refused("shim-not-found", "the shim that runs this install was not found; reinstall with install.sh")
        return {**plan, **shim, "display": f"install.sh from {tag}"}
    if kind == "source":
        raise Refused("source-checkout", "this uio runs from a source checkout; update it with git, then reinstall")
    raise Refused("unmanaged-install", f"this install is managed by pip; run: python3 -m pip install --upgrade "
                                       f"'git+{SOURCE}@{tag}'")


def _run(args, **kw) -> None:
    proc = subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=600, **kw)
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
        raise Failed("install-failed", f"{Path(args[0]).name} exited {proc.returncode}: " + " | ".join(tail))


def clone(tag: str, dest: Path) -> None:
    _run(["git", "-c", "advice.detachedHead=false", "clone", "--quiet", "--depth", "1", "--branch", tag,
          SOURCE + ".git", str(dest)], env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})


def _checkout_version(src: Path):
    try:
        text = (src / "src" / "ultra_inference_optimization" / "__init__.py").read_text()
    except OSError:
        return None
    m = re.search(r'^__version__ = "([^"]+)"', text, re.M)
    return m.group(1) if m else None


def _install_shim(plan: dict) -> None:
    """Clone the exact tag into a temp dir, verify its version, then let its install.sh swap the shim."""
    with tempfile.TemporaryDirectory(prefix="uio-update-") as td:  # removed on success and on failure
        src = Path(td) / "src"
        clone(plan["tag"], src)
        got = _checkout_version(src)
        if got != plan["version"]:
            raise Failed("verify-failed", f"the {plan['tag']} checkout reports version {got}, not {plan['version']}")
        _run(["sh", str(src / "install.sh")], cwd=td,
             env={**os.environ, "UIO_INSTALL_ROOT": plan["root"], "UIO_BIN_DIR": plan["bin"]})


def _replace(target: Path, data: bytes, mode: int) -> None:
    tmp = target.with_name(target.name + f".tmp.{os.getpid()}")
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode), "wb") as fh:
        fh.write(data)
    os.chmod(tmp, mode)
    os.replace(tmp, target)


def write_shim(bindir: str, launcher: Path) -> None:
    """The same shim install.sh writes, replaced atomically."""
    q = lambda s: str(s).replace("'", "'\\''")  # noqa: E731
    body = (f"#!/bin/sh\n{SHIM_MARKER}\n# version {launcher.parent.name}\n"
            f"exec '{q(sys.executable)}' -I '{q(launcher)}' \"$@\"\n")
    _replace(Path(bindir) / TOOL, body.encode(), 0o755)


def prune_releases(root, keep) -> None:
    """install.sh release folders under `root` other than `keep` are removed (never more than one rollback)."""
    for d in Path(root).iterdir():
        if d.name not in keep and version_tuple(d.name) and (d / "uio_launcher.py").is_file():
            shutil.rmtree(d, ignore_errors=True)


def run_install(plan: dict) -> None:
    """Install `plan`. On failure the running version is put back; on success it is kept as the one rollback."""
    prev = {"kind": plan["kind"], "tag": "v" + __version__, "version": __version__}
    if plan["kind"] == "shim":
        prev.update(dir=plan["dir"], root=plan["root"], bin=plan["bin"])
        shim = Path(plan["bin"]) / TOOL
        shim_before = shim.read_bytes()
    try:
        if plan["kind"] == "shim":
            _install_shim(plan)
        else:
            _run(plan["args"])
    except Exception as e:
        cause = _cause(e)
        try:
            if plan["kind"] == "shim":
                _replace(shim, shim_before, 0o755)
                if plan["version"] != __version__:  # a half-written new release folder is debris
                    shutil.rmtree(Path(plan["root"]) / plan["version"], ignore_errors=True)
            else:  # a failed `uv tool install --force` can remove the old tool environment: reinstall this tag
                _run([shutil.which(plan["kind"]) or plan["kind"]] + _manager_command(plan["kind"], prev["tag"])[1:])
        except Exception as r:
            cause += f"; restoring {prev['tag']} also failed: {_cause(r)}"
        raise Failed("install-failed", cause[:600]) from e
    st = load_state()
    st.update(previous={**prev, "to": plan["version"], "at": time.time()}, autoFailed=None,
              shownAt=0.0, shownVersion=None)
    save_state(st)
    if plan["kind"] == "shim":
        prune_releases(plan["root"], {plan["version"], __version__})


def confirm_new_version_ran() -> None:
    """After the new version ran successfully once (or after 7 days at most) the rollback folder is removed."""
    if not (uio_home() / STATE).exists():
        return
    st = load_state()
    prev = st["previous"]
    if not isinstance(prev, dict) or prev.get("kind") != "shim":
        return  # a package-manager rollback is only a recorded tag: there is no copy on disk
    running = Path(__file__).resolve().parents[1]
    ran = prev.get("to") == __version__ and running.parent == Path(prev["root"]).resolve()
    if not ran and time.time() - float(prev.get("at") or 0) < KEEP_ROLLBACK:
        return
    old = Path(prev["dir"])
    if old.resolve() != running:
        shutil.rmtree(old, ignore_errors=True)
    st["previous"] = None
    save_state(st)


def rollback() -> dict:
    """Put back the version that ran before the last update and skip the version that was left."""
    st = load_state()
    prev = st["previous"]
    if not isinstance(prev, dict) or not version_tuple(prev.get("tag")):
        raise Refused("no-rollback", "no previous install is recorded")
    launcher = Path(prev.get("dir") or "") / "uio_launcher.py"
    if prev["kind"] == "shim" and launcher.is_file():
        write_shim(prev["bin"], launcher)
    else:  # a package manager, or a shim whose old folder is gone: reinstall the recorded tag
        run_install(install_plan({"tag": prev["tag"], "version": prev["version"]}))
    st = load_state()
    st.update(previous=None, skip=prev["to"], autoFailed=None)
    save_state(st)
    return {"status": "rolled-back", "version": prev["version"], "skipped": prev["to"]}


# ---- `uio update` --------------------------------------------------------------------------------------------------
def _prompts() -> bool:
    try:
        return sys.stdin.isatty() and sys.stderr.isatty()
    except (AttributeError, ValueError):
        return False


def ask(question: str, default: bool) -> bool:
    print(f"{question} {'[Y/n]' if default else '[y/N]'} ", end="", file=sys.stderr, flush=True)
    line = sys.stdin.readline()  # the question goes to stderr: stdout stays one JSON object
    if not line:
        return False  # end of input (Ctrl-D) is never a yes
    answer = line.strip().lower()
    return default if not answer else answer.startswith("y")


def first_lines(notes: str, count: int = 6) -> list:
    lines = (_CONTROL.sub("", x).strip() for x in str(notes or "").splitlines())
    return [x[:200] for x in lines if x][:count]


def settings(st: dict) -> dict:
    out = {"current": __version__, "installAutomatically": st["auto"], "checkAutomatically": st["checkAutomatically"]}
    if isinstance(st.get("autoFailed"), dict):
        out["lastAutomaticInstallFailure"] = st["autoFailed"]
    return out


def update_command(a, fetch=None) -> dict:
    st = load_state()
    if a.auto:
        if a.auto != "status":
            st.update(auto=a.auto == "on", askedAuto=True)
            save_state(st)
        return {"status": "settings", **settings(st)}
    if a.check_auto:
        if a.check_auto != "status":
            st["checkAutomatically"] = a.check_auto == "on"
            save_state(st)
        return {"status": "settings", **settings(st)}
    if a.clear_state:
        clear_state()
        return {"status": "state-cleared"}
    if a.rollback:
        return rollback()
    if a.skip and available(st):  # the version from the notice, without another request
        st.update(skip=st["latest"])
        save_state(st)
        return {"status": "skipped", "version": st["latest"]}

    st, rel = check(fetch)
    base = {"current": __version__, "latest": st["latest"]}
    if rel is None:
        raise Report({"error": "check-failed", "message": st["failure"], **base}, 1)
    newer = is_newer(rel["version"])
    if a.skip:
        if not newer:
            return {"status": "up-to-date", **base}
        st["skip"] = rel["version"]
        save_state(st)
        return {"status": "skipped", "version": rel["version"]}
    if not newer:
        return {"status": "up-to-date", **base}
    notes = first_lines(rel["notes"])
    if a.check:
        raise Report({"status": "update-available", **base, "skipped": st["skip"] == rel["version"],
                      "notes": notes}, 10)
    plan = install_plan(rel)  # an install another tool owns is refused here, naming the command to run
    print("\n".join([f"{TOOL} {__version__} -> {rel['version']}"] + notes), file=sys.stderr)
    if _prompts() and not ask(f"Install {rel['version']} now?", True):
        return {"status": "declined", **base}
    run_install(plan)
    st = load_state()
    if _prompts() and not st["auto"] and not st["askedAuto"]:
        st.update(auto=ask("Install future updates automatically?", False), askedAuto=True)
        save_state(st)
    return {"status": "installed", **base, "ran": plan["display"], "installAutomatically": st["auto"]}
