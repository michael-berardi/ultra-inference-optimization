# Privacy

`uio` has no telemetry, analytics or crash reporting, and no runtime dependencies.

## Network

- `uio bench` sends benchmark prompts to the `--base-url` you supply; `uio tune` sends them to the `llama-server` it
  starts on `127.0.0.1`. `uio climb` runs the commands you give it.
- The update check is one anonymous `GET https://api.github.com/repos/<owner>/<repo>/releases/latest`, this project's
  public release record, at most once per 24 hours per machine. It runs in a background process started after a
  command finishes. The request carries the fixed `User-Agent: uio-update-check` and standard HTTP headers only: no
  id, installed version, machine data or usage data. GitHub sees the IP address, as with any web request.
- The check does not run when stdout or stderr is not a terminal, when `CI` is set, in agent sessions (`UIO_AGENT=1`
  or `AGENT=1`), with `UIO_NO_UPDATE_CHECK=1`, or after `uio update --check-auto off`.
- `uio update` (run by you, or by the daily check when you turned automatic installs on) makes the same request and
  then fetches the release tag from GitHub through uv, pipx or git.

## Local files

- `~/.cache/uio/update.json` (or `$UIO_HOME/update.json`), one small mode-0600 JSON file: when the last check ran
  and its failure cause, the latest version seen, the skipped version, the two switches, when the notice was last
  shown, and the version to roll back to.
- `~/.cache/uio/update.log`: only written by automatic installs; one file of at most 256 KB.
- `uio update --clear-state` removes both (run it before uninstalling).
- `uio tune` and `uio climb` keep run folders and a lock file in the same directory; `rm -rf ~/.cache/uio` removes
  everything `uio` stores.
