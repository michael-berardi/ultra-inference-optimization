"""Installer tests: build a local release dir and run install.sh against it with a temporary HOME."""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION = "0.2.0"


def make_release(release: Path, work: Path) -> None:
    build = work / "build"
    shutil.copytree(ROOT / "src", build / "src")
    shutil.copy2(ROOT / "pyproject.toml", build)
    shutil.copy2(ROOT / "README.md", build)
    env = dict(os.environ, SOURCE_DATE_EPOCH="0", PIP_DISABLE_PIP_VERSION_CHECK="1")
    built = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps",
                            "--wheel-dir", str(release), str(build)],
                           capture_output=True, text=True, env=env, timeout=180)
    if built.returncode:
        raise AssertionError(f"wheel build failed:\n{built.stdout}\n{built.stderr}")
    shutil.copy2(ROOT / "install.sh", release / "install.sh")
    shutil.copy2(ROOT / "install.ps1", release / "install.ps1")
    lines = []
    for path in sorted(p for p in release.iterdir() if p.is_file() and p.name != "SHA256SUMS"):
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    (release / "SHA256SUMS").write_text("".join(lines))


class InstallerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="uio-installer-test-")
        cls.work = Path(cls._tmp.name)
        cls.release = cls.work / "release"
        cls.release.mkdir()
        make_release(cls.release, cls.work)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def run_installer(self, backend, foreign_shim=None, bad_sums=False):
        home = Path(self._tmp.name) / f"home-{backend}-{len(list(self.work.glob('home-*')))}"
        bindir = home / ".local/bin"
        bindir.mkdir(parents=True)
        if foreign_shim is not None:
            (bindir / "uio").write_text(foreign_shim)
        release = self.release
        if bad_sums:
            release = Path(self._tmp.name) / "bad-release"
            if not release.exists():
                shutil.copytree(self.release, release)
            sums = release / "SHA256SUMS"
            text = sums.read_text()
            wheel_line = next(line for line in text.splitlines() if ".whl" in line)
            sums.write_text(text.replace(wheel_line.split()[0], "0" * 64))
        env = dict(os.environ)
        env.update(HOME=str(home), UIO_RELEASE_BASE=str(release), UIO_INSTALL_ROOT=str(home / ".local/share/uio"),
                   UIO_BIN_DIR=str(bindir), UIO_INSTALLER=backend, UV_CACHE_DIR=str(home / "uv-cache"),
                   UV_NO_CONFIG="1", PIPX_CACHE_HOME=str(home / "pipx-cache"))
        for key in ("UIO_VERSION", "UIO_INSTALL_ROOT_OVERRIDE"):
            env.pop(key, None)
        proc = subprocess.run(["sh", str(release / "install.sh")], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=180)
        return proc, home, bindir

    def assert_installed(self, proc, bindir):
        self.assertEqual(proc.returncode, 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        got = subprocess.run([str(bindir / "uio"), "--version"], capture_output=True, text=True, timeout=30)
        self.assertEqual(got.returncode, 0, got.stderr)
        self.assertEqual(got.stdout.strip(), f"uio {VERSION}")
        self.assertIn("# uio-shim: managed by ultra-inference-optimization install.sh",
                      (bindir / "uio").read_text())

    def test_venv_install_from_local_release(self):
        proc, _, bindir = self.run_installer("venv")
        self.assert_installed(proc, bindir)

    @unittest.skipUnless(shutil.which("uv"), "uv is not installed")
    def test_uv_install_from_local_release(self):
        proc, _, bindir = self.run_installer("uv")
        self.assert_installed(proc, bindir)

    @unittest.skipUnless(shutil.which("pipx"), "pipx is not installed")
    def test_pipx_install_from_local_release(self):
        proc, _, bindir = self.run_installer("pipx")
        self.assert_installed(proc, bindir)

    def test_foreign_shim_is_left_alone(self):
        foreign = "#!/bin/sh\necho foreign\n"
        proc, _, bindir = self.run_installer("venv", foreign_shim=foreign)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("was not installed by this script", proc.stderr)
        self.assertEqual((bindir / "uio").read_text(), foreign)

    def test_bad_checksum_installs_nothing(self):
        proc, _, bindir = self.run_installer("venv", bad_sums=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("checksum mismatch", proc.stderr)
        self.assertFalse((bindir / "uio").exists())

    def test_checkout_installer_still_works(self):
        home = Path(self._tmp.name) / "checkout-home"
        bindir = home / ".local/bin"
        env = dict(os.environ)
        env.update(HOME=str(home), UIO_INSTALL_ROOT=str(home / "uio"), UIO_BIN_DIR=str(bindir))
        env.pop("UIO_RELEASE_BASE", None)
        env.pop("UIO_INSTALLER", None)
        proc = subprocess.run(["sh", str(ROOT / "install.sh")], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=180)
        self.assert_installed(proc, bindir)


if __name__ == "__main__":
    unittest.main()
