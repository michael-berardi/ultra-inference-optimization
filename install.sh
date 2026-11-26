#!/bin/sh
# uio installer for macOS and Linux:
#   curl -fsSL https://github.com/michael-berardi/ultra-inference-optimization/releases/latest/download/install.sh | sh
# Release mode (the default when this script is piped): download the wheel and SHA256SUMS, verify the wheel's
# SHA-256 before installing anything, then use uv or pipx when available, otherwise a private venv under
# ~/.local/share/uio. A managed shim is placed in ~/.local/bin; a foreign shim is never overwritten.
# Checkout mode: when this file sits in a repository checkout, it keeps working offline by copying the source
# tree instead of downloading a wheel. Both modes write the same shim marker.
# Environment: UIO_INSTALL_ROOT (default ~/.local/share/uio), UIO_BIN_DIR (default ~/.local/bin),
# UIO_VERSION (pin a release, e.g. 0.2.0), UIO_RELEASE_BASE (download base override, used by tests),
# UIO_INSTALLER (uv|pipx|venv, override the backend; auto is the default).
set -eu

REPO="github.com/michael-berardi/ultra-inference-optimization"
MARK='# uio-shim: managed by ultra-inference-optimization install.sh'
TMP=""
STAGE=""

abs() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s/%s' "$(pwd)" "$1" ;; esac; }
HOME_DIR=$(abs "${UIO_INSTALL_ROOT:-$HOME/.local/share/uio}")
BIN_DIR=$(abs "${UIO_BIN_DIR:-$HOME/.local/bin}")
SHIM="$BIN_DIR/uio"
if [ -n "${UIO_RELEASE_BASE:-}" ]; then
  case "$UIO_RELEASE_BASE" in
    /*|./*|../*) BASE=$(abs "$UIO_RELEASE_BASE") ;;
    *) BASE=$UIO_RELEASE_BASE ;;
  esac
elif [ -n "${UIO_VERSION:-}" ]; then
  BASE="https://$REPO/releases/download/v${UIO_VERSION#v}"
else
  BASE="https://$REPO/releases/latest/download"
fi
BASE=${BASE%/}

say() { printf '%s\n' "$*"; }
die() { printf 'uio install: %s\n' "$*" >&2; exit 1; }
refuse() { printf 'uio install: %s\n' "$*" >&2; exit 2; }
trap 'rm -rf "$TMP" "$STAGE"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

fetch() { # fetch <url-or-local-file> <destination>
  if [ -z "${1:-}" ]; then die "empty download source"; fi
  case "$1" in
    http://*|https://*)
      if command -v curl >/dev/null 2>&1; then
        curl -fsSL --retry 3 -o "$2" "$1" || die "download failed: $1"
      elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$2" "$1" || die "download failed: $1"
      else
        die "needs curl or wget (macOS ships curl; Debian/Ubuntu: apt-get install curl)"
      fi ;;
    file://*) cp -f "${1#file://}" "$2" || die "cannot read $1" ;;
    *) cp -f "$1" "$2" || die "cannot read $1" ;;
  esac
}

if command -v sha256sum >/dev/null 2>&1; then sha256() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
else die "needs sha256sum or shasum"; fi

expect() { # expect <file> <name> <sums-file>
  want=$(awk -v n="$2" '{ f=$2; sub(/^\*/, "", f); if (f == n) print $1 }' "$3")
  [ -n "$want" ] || die "no checksum for $2 in SHA256SUMS"
  [ "$(sha256 "$1")" = "$want" ] || die "checksum mismatch for $2; nothing was installed"
}

need_python() {
  command -v python3 >/dev/null 2>&1 \
    || die "needs Python 3.10 or newer (macOS: brew install python@3.12; Debian/Ubuntu: apt-get install python3 python3-venv)"
  PY=$(command -v python3)
  case "$PY" in /*) ;; *) die "python3 must resolve to an absolute path" ;; esac
  "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
    || die "needs Python 3.10 or newer; found an older python3"
}

check_shim() {
  if [ -e "$SHIM" ] || [ -L "$SHIM" ]; then
    if [ -L "$SHIM" ] || ! grep -qF "$MARK" "$SHIM" 2>/dev/null; then
      refuse "$SHIM exists and was not installed by this script; it was left alone (move it aside or set UIO_BIN_DIR)"
    fi
  fi
}

sq() { printf "%s" "$1" | sed "s/'/'\\\\''/g"; }

write_shim() { # write_shim <absolute-command-path> [absolute-python-for-isolated-launcher]
  target=$1
  shift
  tmp="$SHIM.tmp.$$"
  {
    printf '#!/bin/sh\n%s\n' "$MARK"
    if [ "$#" -gt 0 ]; then
      printf "exec '%s' -I '%s' \"\$@\"\n" "$(sq "$1")" "$(sq "$target")"
    else
      printf "exec '%s' \"\$@\"\n" "$(sq "$target")"
    fi
  } > "$tmp"
  chmod 755 "$tmp" || die "cannot mark the shim executable"
  mv -f "$tmp" "$SHIM" || die "cannot install the shim at $SHIM"
  got=$("$SHIM" --version 2>&1) || die "the installed command does not run: $got"
  [ "$got" = "uio $VERSION" ] || die "the installed command reports '$got', not uio $VERSION"
  say "uio $VERSION installed ($SHIM)."
  case ":$PATH:" in *":$BIN_DIR:"*) ;; *) say "Add $BIN_DIR to your PATH:  export PATH=\"$BIN_DIR:\$PATH\"" ;; esac
}

checkout_install() {
  [ -f "$0" ] || die "checkout mode needs install.sh as a real file"
  here=$(cd "$(dirname "$0")" && pwd)
  [ -f "$here/pyproject.toml" ] && [ -f "$here/src/ultra_inference_optimization/__init__.py" ] \
    && grep -q '^name = "ultra-inference-optimization"' "$here/pyproject.toml" \
    || die "install.sh is not in a uio checkout"
  VERSION=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$here/src/ultra_inference_optimization/__init__.py" | head -n 1)
  [ -n "$VERSION" ] || die "cannot read the checkout version"

  need_python
  check_shim
  dest="$HOME_DIR/$VERSION"
  mkdir -p "$dest" "$BIN_DIR" || die "cannot create $HOME_DIR or $BIN_DIR"
  rm -rf "$dest/ultra_inference_optimization"
  cp -R "$here/src/ultra_inference_optimization" "$dest/ultra_inference_optimization"
  find "$dest" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
  cat > "$dest/uio_launcher.py" <<'LAUNCHER'
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultra_inference_optimization.cli import main  # noqa: E402

sys.exit(main())
LAUNCHER
  write_shim "$dest/uio_launcher.py" "$PY"
}

install_uv() {
  UV_TOOLS="$HOME_DIR/uv-tools"
  UV_BIN="$HOME_DIR/uv-bin"
  mkdir -p "$UV_TOOLS" "$UV_BIN" || die "cannot create the uv tool folders"
  UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME_DIR/uv-cache}" UV_NO_CONFIG=1 \
  UV_TOOL_DIR="$UV_TOOLS" UV_TOOL_BIN_DIR="$UV_BIN" \
    uv tool install --no-index --no-python-downloads --python "$PY" --force --quiet "$1" \
    || die "uv could not install the wheel"
  [ -x "$UV_BIN/uio" ] || die "uv did not expose a uio command"
  write_shim "$UV_BIN/uio"
}

install_pipx() {
  PIPX_HOME_DIR="$HOME_DIR/pipx"
  PIPX_BIN="$HOME_DIR/pipx-bin"
  mkdir -p "$PIPX_HOME_DIR" "$PIPX_BIN" || die "cannot create the pipx tool folders"
  PIPX_HOME="$PIPX_HOME_DIR" PIPX_BIN_DIR="$PIPX_BIN" PIPX_DEFAULT_BACKEND=pip \
  PIPX_CACHE_HOME="${PIPX_CACHE_HOME:-$HOME_DIR/pipx-cache}" \
    pipx install --force --skip-maintenance --python "$PY" "$1" \
    || die "pipx could not install the wheel"
  [ -x "$PIPX_BIN/uio" ] || die "pipx did not expose a uio command"
  write_shim "$PIPX_BIN/uio"
}

install_venv() {
  VENV="$HOME_DIR/venv"
  STAGE="$HOME_DIR/.venv.new.$$"
  rm -rf "$STAGE"
  "$PY" -c 'import ensurepip' >/dev/null 2>&1 \
    || die "this python3 cannot create venvs (ensurepip is missing); on Debian/Ubuntu install python3-venv"
  "$PY" -m venv "$STAGE" || die "cannot create the private venv"
  "$STAGE/bin/python3" -m pip install --no-index --no-deps --disable-pip-version-check --quiet "$1" \
    || die "pip could not install the wheel; nothing was changed"
  got=$("$STAGE/bin/uio" --version 2>&1) || die "the new copy does not start: $got"
  [ "$got" = "uio $VERSION" ] || die "the new copy reports '$got', not uio $VERSION; nothing was changed"

  rm -rf "$VENV.old"
  [ ! -d "$VENV" ] || mv "$VENV" "$VENV.old" || die "cannot move the old install aside"
  # A venv embeds its creation path in its scripts; rewrite it after the rename, on the same physical path.
  OLD_PATH=$STAGE
  PAT=$(printf '%s' "$OLD_PATH" | sed 's/[][\\.*^$|]/\\&/g')
  REP=$(printf '%s' "$VENV" | sed 's/[\\&|]/\\&/g')
  mv "$STAGE" "$VENV" || { [ ! -d "$VENV.old" ] || mv "$VENV.old" "$VENV"; die "cannot move the new venv into place"; }
  STAGE=""
  for f in "$VENV/bin/"*; do
    [ -f "$f" ] && [ ! -L "$f" ] || continue
    if grep -qF "$OLD_PATH" "$f" 2>/dev/null; then
      sed "s|$PAT|$REP|g" "$f" > "$TMP/fix" || die "cannot finalize $f"
      cat "$TMP/fix" > "$f" || die "cannot finalize $f"
    fi
  done
  rm -rf "$VENV.old"
  write_shim "$VENV/bin/uio"
}

release_install() {
  need_python
  check_shim
  TMP=$(mktemp -d "${TMPDIR:-/tmp}/uio-install.XXXXXX") || die "cannot create a temporary folder"

  say "Downloading uio..."
  fetch "$BASE/SHA256SUMS" "$TMP/SHA256SUMS"
  WHEEL=$(awk '{ f=$2; sub(/^\*/, "", f); if (f ~ /^ultra_inference_optimization-.*-py3-none-any\.whl$/) { print f; exit } }' "$TMP/SHA256SUMS")
  [ -n "$WHEEL" ] || die "no uio wheel is listed in SHA256SUMS; this is not a uio release"
  VERSION=${WHEEL#ultra_inference_optimization-}
  VERSION=${VERSION%%-*}
  if [ -n "${UIO_VERSION:-}" ] && [ "$VERSION" != "${UIO_VERSION#v}" ]; then
    die "the release at $BASE has version $VERSION, not the requested ${UIO_VERSION#v}"
  fi
  fetch "$BASE/$WHEEL" "$TMP/$WHEEL"
  expect "$TMP/$WHEEL" "$WHEEL" "$TMP/SHA256SUMS"

  mkdir -p "$HOME_DIR" "$BIN_DIR" || die "cannot create $HOME_DIR or $BIN_DIR"
  HOME_DIR=$(cd "$HOME_DIR" && pwd -P)
  backend=${UIO_INSTALLER:-auto}
  case "$backend" in
    auto)
      if command -v uv >/dev/null 2>&1; then backend=uv
      elif command -v pipx >/dev/null 2>&1; then backend=pipx
      else backend=venv; fi ;;
    uv|pipx|venv) ;;
    *) die "UIO_INSTALLER must be uv, pipx or venv" ;;
  esac
  say "Installing uio $VERSION with $backend into $HOME_DIR ..."
  case "$backend" in
    uv) install_uv "$TMP/$WHEEL" ;;
    pipx) install_pipx "$TMP/$WHEEL" ;;
    venv) install_venv "$TMP/$WHEEL" ;;
  esac
}

mode=release
if [ -z "${UIO_RELEASE_BASE:-}" ] && [ -f "${0:-}" ]; then
  here=$(cd "$(dirname "$0")" && pwd) || here=""
  if [ -n "$here" ] && [ -f "$here/pyproject.toml" ] \
     && [ -f "$here/src/ultra_inference_optimization/__init__.py" ]; then
    mode=checkout
  fi
fi
case "$mode" in
  checkout) checkout_install ;;
  release) release_install ;;
esac
