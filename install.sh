#!/bin/sh
# Install uio into ~/.local/share/uio/<version> and a venv-free shim at ~/.local/bin/uio.
# Offline: copies this checkout, fetches nothing. Refuses to overwrite a uio it did not install.
set -eu

here=$(cd "$(dirname "$0")" && pwd)
version=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$here/src/ultra_inference_optimization/__init__.py")
[ -n "$version" ] || { echo "uio install: cannot read version" >&2; exit 3; }

root=${UIO_INSTALL_ROOT:-$HOME/.local/share/uio}
bindir=${UIO_BIN_DIR:-$HOME/.local/bin}
dest="$root/$version"
shim="$bindir/uio"
marker="# uio-shim: managed by ultra-inference-optimization install.sh"

command -v python3 >/dev/null 2>&1 || { echo "uio install: python3 (3.10+) is required" >&2; exit 3; }
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
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

tmp="$shim.tmp.$$"
cat > "$tmp" <<SHIM
#!/bin/sh
$marker
# version $version
exec python3 -I -c 'import sys; sys.path.insert(0, "$dest"); from ultra_inference_optimization.cli import main; sys.exit(main())' "\$@"
SHIM
chmod 755 "$tmp"
mv -f "$tmp" "$shim"

echo "installed uio $version -> $dest"
echo "shim: $shim"
case ":$PATH:" in *":$bindir:"*) ;; *) echo "note: $bindir is not on PATH" ;; esac
