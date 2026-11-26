# Privacy

`uio` has no telemetry, analytics or crash reporting, no model download, no upload of benchmark results and no runtime
dependencies. It ships no UI, web page, font or remote script.

## Network

- `uio bench` sends the benchmark prompts to the `--base-url` you supply (your prompts if you passed `--panel FILE`,
  otherwise prompts generated locally). `uio tune` sends them to the `llama-server` it starts itself, bound to
  `127.0.0.1`. If you point `--base-url` at a remote server, that server's operator receives the prompts and sees your
  connection.
- `uio climb` runs the shell commands you give it; whatever they send is your own configuration.
- The update check is one anonymous `GET https://api.github.com/repos/<owner>/<repo>/releases/latest`, this project's
  public release record, at most once per 24 hours per machine. It runs in a background process started after a
  command finishes. The request carries the fixed `User-Agent: uio-update-check` and standard HTTP headers only: no
  id, installed version, machine data or usage data. GitHub sees the IP address, as with any web request.
- The check does not run when stdout or stderr is not a terminal, when `CI` is set, in agent sessions (`UIO_AGENT=1`
  or `AGENT=1`), with `UIO_NO_UPDATE_CHECK=1`, or after `uio update --check-auto off`.
- `uio update` (run by you, or by the daily check when you turned automatic installs on) makes the same request and
  then fetches the release from GitHub: the tag through uv, pipx or git, or the release's `SHA256SUMS`, `install.sh`
  and wheel for a one-line install. Apart from that, the installed tool never fetches code.
- The install commands in the README are run by you. The one-line installer downloads `SHA256SUMS` and the wheel (and,
  on Windows, `install.sh`) from the GitHub release; the uv and pipx commands fetch the pinned tag. All of them set
  `DO_NOT_TRACK=1` for uv and pipx.

## Local files

Everything lives in `~/.cache/uio` (or `$UIO_HOME`):

- Run folders (`runs/`) with plans, ledgers, configs and server logs; a lock file holding a PID; a child-PID file; and a
  `last/` record holding profile paths and metrics.
- `update.json`, one small JSON file: when the last check ran and its failure cause, the latest version seen, the
  skipped version, the two switches, when the notice was last shown, and the version to roll back to.
- `update.log`: only written by automatic installs; one file of at most 256 KB.

Files are 0600 and directories newly created by `uio` are 0700; `~/.cache/uio` itself and everything below it are
tightened to 0700 even if they already existed, and files left by older versions are tightened on the next write.
Directories that already exist outside `~/.cache/uio`, including an explicit `--workdir` and a profile's parent, are
never chmodded and keep their own permissions.

## Retention and removal

- The newest ten run folders are kept; older ones are deleted at the start of the next `tune` or `climb` (only for the
  default workdir; an explicit `--workdir` is left alone).
- With `--apply`, a 0600 backup of the profile is written next to the profile.
- `uio update --clear-state` removes `update.json` and `update.log` (run it before uninstalling).
- `rm -rf ~/.cache/uio` removes everything `uio` stores; also delete any `*.bak-*` files next to profiles you applied to.
- The installer itself writes only `~/.local/share/uio` (or `UIO_INSTALL_ROOT`) and the `~/.local/bin/uio` shim; remove
  both to uninstall it.
