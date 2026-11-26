#!/bin/sh
# Install uio into ~/.local/share/uio/<version> and a venv-free shim at ~/.local/bin/uio.
# Offline: copies this checkout, fetches nothing. Refuses to overwrite a uio it did not install.
set -eu

PINNED='DO_NOT_TRACK=1 uv tool install "git+https://github.com/michael-berardi/ultra-inference-optimization@v0.2.0"'
not_checkout() {
  cat >&2 <<MSG
uio install: this script must be run from a checkout of the repository (sh install.sh), not piped from a URL or run
from another directory: it copies the source tree that sits next to it and would otherwise install whatever is in the
current directory. To install without a checkout use the pinned release:
  $PINNED
  (or: pipx install with the same URL)
MSG
  exit 2
}
case "$(basename "$0")" in sh|-sh|bash|-bash|dash|zsh|-zsh|ash|ksh) not_checkout ;; esac
[ -f "$0" ] || not_checkout
here=$(cd "$(dirname "$0")" && pwd)
[ -f "$here/pyproject.toml" ] && [ -f "$here/src/ultra_inference_optimization/__init__.py" ] \
  && grep -q '^name = "ultra-inference-optimization"' "$here/pyproject.toml" || not_checkout
version=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$here/src/ultra_inference_optimization/__init__.py")
[ -n "$version" ] || { echo "uio install: cannot read version" >&2; exit 3; }

root=${UIO_INSTALL_ROOT:-$HOME/.local/share/uio}
bindir=${UIO_BIN_DIR:-$HOME/.local/bin}
dest="$root/$version"
shim="$bindir/uio"
marker="# uio-shim: managed by ultra-inference-optimization install.sh"

py=$(command -v python3 2>/dev/null || true)
case "$py" in /*) ;; *) echo "uio install: python3 (3.10+) is required (an absolute path to it is needed)" >&2; exit 3 ;; esac
"$py" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
  || { echo "uio install: Python 3.10 or newer is required" >&2; exit 3; }

if [ -e "$shim" ] || [ -L "$shim" ]; then
  if ! grep -qF "$marker" "$shim" 2>/dev/null; then
    cat >&2 <<MSG
uio install: $shim exists and was not installed by this script, so it was left alone.
To fix: move it aside (mv "$shim" "$shim.orig"), or install elsewhere with UIO_BIN_DIR=/some/dir/on/PATH sh install.sh
MSG
    exit 2
  fi
fi

mkdir -p "$dest" "$bindir"
rm -rf "$dest/ultra_inference_optimization"
cp -R "$here/src/ultra_inference_optimization" "$dest/ultra_inference_optimization"
find "$dest" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

# The launcher is a fixed Python file (no text from this script is interpolated into code); the shim only names two
# absolute paths, each single-quoted with embedded quotes escaped.
cat > "$dest/uio_launcher.py" <<'LAUNCHER'
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultra_inference_optimization.cli import main  # noqa: E402

sys.exit(main())
LAUNCHER

sq() { printf "%s" "$1" | sed "s/'/'\\\\''/g"; }
tmp="$shim.tmp.$$"
{
  printf '#!/bin/sh\n%s\n# version %s\n' "$marker" "$version"
  printf "exec '%s' -I '%s' \"\$@\"\n" "$(sq "$py")" "$(sq "$dest/uio_launcher.py")"
} > "$tmp"
chmod 755 "$tmp"
mv -f "$tmp" "$shim"

echo "installed uio $version -> $dest"
echo "shim: $shim"
case ":$PATH:" in *":$bindir:"*) ;; *) echo "note: $bindir is not on PATH" ;; esac
