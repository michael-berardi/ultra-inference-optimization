#!/bin/sh
# Builds the download assets for HEAD into <out> (default dist/release):
#   ultra_inference_optimization-<version>-py3-none-any.whl  the verified release wheel
#   install.sh, install.ps1                                  the installers
#   SHA256SUMS                                               checksums of all three files above
# Usage: scripts/build-release.sh [out-dir]   (run from a clean checkout of the release tag)
set -eu
cd "$(dirname "$0")/.."
OUT="${1:-dist/release}"
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "build-release: the working tree has changes; build from a clean checkout" >&2; exit 1; }
root=$(pwd -P)
case "$OUT" in /*) ;; *) OUT="$root/$OUT" ;; esac
OUT=${OUT%/}
case "$OUT" in */..|*/../*|*/.|*/./*) echo "build-release: out directory must not contain . or .." >&2; exit 2 ;; esac
case "$OUT" in
  "$root"/dist/?*) ;;
  *) echo "build-release: out directory must be a non-empty path under $root/dist" >&2; exit 2 ;;
esac
command -v sha256sum >/dev/null 2>&1 || command -v shasum >/dev/null 2>&1 \
  || { echo "build-release: needs sha256sum or shasum" >&2; exit 1; }

commit=$(git rev-parse HEAD)
rm -rf "$OUT"; mkdir -p "$OUT"
stage=$(mktemp -d "${TMPDIR:-/tmp}/uio-build.XXXXXX"); trap 'rm -rf "$stage"' EXIT
git archive --format=tar HEAD | tar -x -C "$stage"

# Build from the committed tree. Build isolation fetches only setuptools; uio has no runtime dependencies.
SOURCE_DATE_EPOCH=0 PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip wheel --no-deps --wheel-dir "$OUT" "$stage" >/dev/null \
  || { echo "build-release: pip could not build the wheel" >&2; exit 1; }
wheel=$(find "$OUT" -maxdepth 1 -type f -name 'ultra_inference_optimization-*-py3-none-any.whl' -print | head -n 1)
[ -n "$wheel" ] || { echo "build-release: no wheel was built" >&2; exit 1; }
wheel=${wheel##*/}
cp install.sh install.ps1 "$OUT/"
( cd "$OUT" && if command -v sha256sum >/dev/null 2>&1; then
    sha256sum install.ps1 install.sh "$wheel"
  else
    shasum -a 256 install.ps1 install.sh "$wheel"
  fi > SHA256SUMS )
echo "built $OUT for $commit"
ls -l "$OUT"
