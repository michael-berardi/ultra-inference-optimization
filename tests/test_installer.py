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
OWNER_MARKER = "# uio-installed: managed by ultra-inference-optimization install.sh\n"


def make_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


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

    def mutated_release(self, name, mutate):
        dest = Path(self._tmp.name) / name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self.release, dest)
        sums = dest / "SHA256SUMS"
        mutate(sums)
        return dest

    def rename_wheel_in_sums(self, name, old, new):
        def mutate(sums):
            text = sums.read_text()
            line = next(line for line in text.splitlines() if ".whl" in line)
            sums.write_text(text.replace(line, line.replace(old, new, 1)))
        return self.mutated_release(name, mutate)

    def run_installer(self, backend, foreign_shim=None, bad_sums=False, setup_home=None,
                      release_dir=None, path_prepend=None, path=None):
        home = Path(self._tmp.name) / f"home-{backend}-{len(list(self.work.glob('home-*')))}"
        bindir = home / ".local/bin"
        bindir.mkdir(parents=True)
        if foreign_shim is not None:
            (bindir / "uio").write_text(foreign_shim)
        if setup_home is not None:
            setup_home(home)
        release = release_dir or self.release
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
        if path_prepend is not None:
            env["PATH"] = str(path_prepend) + os.pathsep + env.get("PATH", "")
        if path is not None:
            env["PATH"] = str(path)
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

    def test_shell_name_on_stdin_never_enters_checkout_mode(self):
        cwd = Path(self._tmp.name) / "shell-cwd"
        (cwd / "src/ultra_inference_optimization").mkdir(parents=True)
        (cwd / "sh").write_text("#!/bin/sh\nexit 0\n")
        shutil.copy2(ROOT / "pyproject.toml", cwd / "pyproject.toml")
        shutil.copy2(ROOT / "src/ultra_inference_optimization/__init__.py",
                     cwd / "src/ultra_inference_optimization/__init__.py")
        fakebin = cwd / "fakebin"
        make_executable(fakebin / "curl", 'echo "offline release gate" >&2\nexit 1\n')
        home = cwd / "home"
        bindir = home / "bin"
        bindir.mkdir(parents=True)
        env = dict(os.environ)
        env.update(HOME=str(home), UIO_INSTALL_ROOT=str(home / "uio"), UIO_BIN_DIR=str(bindir),
                   PATH=str(fakebin) + os.pathsep + env.get("PATH", ""))
        for key in ("UIO_RELEASE_BASE", "UIO_VERSION", "UIO_INSTALLER"):
            env.pop(key, None)
        proc = subprocess.run(["sh"], input=(ROOT / "install.sh").read_text(), cwd=cwd, env=env,
                              capture_output=True, text=True, timeout=180)
        self.assertNotEqual(proc.returncode, 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("SHA256SUMS", proc.stderr)
        self.assertFalse((home / "uio" / VERSION / "uio_launcher.py").exists())

    def test_venv_without_marker_is_left_alone(self):
        def setup(home):
            venv = home / ".local/share/uio/venv"
            (venv / "bin").mkdir(parents=True)
            make_executable(venv / "bin/uio", "#!/bin/sh\necho OLD\n")

        proc, home, _ = self.run_installer("venv", setup_home=setup)
        self.assertNotEqual(proc.returncode, 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("was not installed by this script", proc.stderr)
        root = home / ".local/share/uio"
        self.assertEqual((root / "venv/bin/uio").read_text(), "#!/bin/sh\necho OLD\n")
        self.assertFalse((root / ".uio-installed").exists())

    def test_managed_paths_without_uio_are_left_alone(self):
        for name in ("venv", "venv.old"):
            with self.subTest(name=name):
                def setup(home, name=name):
                    root = home / ".local/share/uio"
                    root.mkdir(parents=True, exist_ok=True)
                    (root / ".uio-installed").write_text(OWNER_MARKER)
                    path = root / name
                    path.mkdir(parents=True)
                    (path / "keep").write_text("keep\n")

                proc, home, _ = self.run_installer("venv", setup_home=setup)
                self.assertNotEqual(proc.returncode, 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
                self.assertIn("was not installed by this script", proc.stderr)
                self.assertEqual((home / ".local/share/uio" / name / "keep").read_text(), "keep\n")

    def test_failed_swap_restores_venv_old(self):
        fakebin = Path(self._tmp.name) / "fake-chmod-bin"
        make_executable(fakebin / "chmod", 'echo "fake chmod failure" >&2\nexit 1\n')

        def setup(home):
            root = home / ".local/share/uio"
            root.mkdir(parents=True, exist_ok=True)
            (root / ".uio-installed").write_text(OWNER_MARKER)
            make_executable(root / "venv/bin/uio", "#!/bin/sh\necho OLD\n")

        proc, home, _ = self.run_installer("venv", setup_home=setup, path_prepend=fakebin)
        self.assertNotEqual(proc.returncode, 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("fake chmod failure", proc.stderr)
        root = home / ".local/share/uio"
        self.assertEqual((root / "venv/bin/uio").read_text(), "#!/bin/sh\necho OLD\n")
        self.assertFalse((root / "venv.old").exists())

    def test_wheel_name_with_slash_is_rejected(self):
        release = self.rename_wheel_in_sums("slash-release", "-py3-none-any.whl", "-../../-py3-none-any.whl")
        proc, _, bindir = self.run_installer("venv", release_dir=release)
        self.assertNotEqual(proc.returncode, 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("unsafe wheel name", proc.stderr)
        self.assertFalse((bindir / "uio").exists())

    def test_non_numeric_version_is_rejected(self):
        release = self.rename_wheel_in_sums("bad-version-release", "-0.2.0-", "-v0.2.0-")
        proc, _, bindir = self.run_installer("venv", release_dir=release)
        self.assertNotEqual(proc.returncode, 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("bad version", proc.stderr)
        self.assertFalse((bindir / "uio").exists())

    def test_auto_pipx_failure_falls_back_to_venv(self):
        bin_dir = Path(self._tmp.name) / "no-uv-path"
        bin_dir.mkdir()
        for cmd in ("python3", "mktemp", "awk", "sed", "grep", "cut", "cat", "chmod", "mv",
                    "rm", "mkdir", "cp", "find", "shasum", "sha256sum"):
            src = shutil.which(cmd)
            if src:
                (bin_dir / cmd).symlink_to(src)
        make_executable(bin_dir / "pipx", 'echo "fake pipx failure" >&2\nexit 7\n')
        proc, home, bindir = self.run_installer("auto", path=str(bin_dir) + os.pathsep + "/usr/bin:/bin")
        self.assertEqual(proc.returncode, 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("falling back to the private venv backend", proc.stdout + proc.stderr)
        self.assert_installed(proc, bindir)
        self.assertTrue((home / ".local/share/uio/venv/bin/uio").exists())


if __name__ == "__main__":
    unittest.main()


class BuildReleaseOutDirTests(unittest.TestCase):
    def test_dot_segments_in_out_dir_are_refused(self):
        for out in ("dist/..", "dist/x/..", "dist/./x", "."):
            proc = subprocess.run(["sh", str(ROOT / "scripts/build-release.sh"), out],
                                  cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0, out)
            self.assertTrue((ROOT / "pyproject.toml").exists(), out)
