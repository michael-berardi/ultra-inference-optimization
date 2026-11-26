<!-- update-contract: v1 -->
# Updates

`uio` tells you when a newer stable release exists and installs it in one step. Automatic installs are off until
you turn them on.

## Notice

After a command finishes in an interactive terminal, when a newer release is known, stderr shows (second line dim):

```
uio 0.3.0 available (you have 0.2.0) · run: uio update
turn on automatic updates: uio update --auto on · skip: uio update --skip
```

The notice is never part of the JSON (or `--human`) result on stdout. It appears at most once per 24 hours per
version and never for a skipped version; a newer release notifies again.

## Commands

| command | what it does |
| --- | --- |
| `uio update` | checks now, shows `current -> new` and the first lines of the release notes, asks `Install <new> now? [Y/n]` on a terminal and installs; after the first install it asks once `Install future updates automatically? [y/N]` (the answer is remembered). Without a terminal it installs without questions. |
| `uio update --check` | report only: exit 0 up to date, 10 update available, 1 check failed (`message` names the cause) |
| `uio update --skip` | no notice for this version until a newer one exists |
| `uio update --auto on\|off\|status` | install updates automatically (off by default); stored in the state file, so it survives updates |
| `uio update --check-auto on\|off\|status` | check for updates automatically (on by default) |
| `uio update --rollback` | put back the version that ran before the last update, and skip the version you left |
| `uio update --clear-state` | remove the update state and log (run it before uninstalling) |

## The check

- One anonymous GET of `https://api.github.com/repos/<owner>/<repo>/releases/latest`; owner and repository are the
  package's declared source (`[project.urls] Source`). Fixed `User-Agent: uio-update-check`; no id, version, machine
  or usage data.
- At most once per 24 hours per machine. The result is cached in `~/.cache/uio/update.json` (or `$UIO_HOME`), one
  0600 file written atomically.
- It runs in a detached background process started after a command finishes, with a 4-second timeout; the command
  never waits for it. A failure is recorded in the cache with its cause and not retried before the next day; it is
  never printed. `uio update --check` checks again and shows the cause.
- Off automatically when stdout or stderr is not a terminal, when `CI` is set, in agent sessions (`UIO_AGENT=1` or
  `AGENT=1`), with `UIO_NO_UPDATE_CHECK=1`, or after `uio update --check-auto off`. An agent host that wants the
  notice can run `uio update --check` and show it in its own status line.
- Only a published, stable `x.y.z` release strictly newer than the running version counts. Drafts, prereleases and
  unparseable tags never do.

## Install kinds

| installed with | `uio update` runs |
| --- | --- |
| `uv tool install "git+…@vX.Y.Z"` | `uv tool install --force git+https://github.com/<owner>/<repo>@<new tag>` |
| `pipx install "git+…@vX.Y.Z"` | `pipx install --force git+https://github.com/<owner>/<repo>@<new tag>` |
| `curl -fsSL …/releases/latest/download/install.sh \| sh` (uv, pipx or private venv under `~/.local/share/uio`) | downloads the new release's `SHA256SUMS` and `install.sh` into a temporary directory, checks `install.sh` against `SHA256SUMS`, and runs it with the same install root, bin directory and backend; it verifies the wheel before installing |
| `sh install.sh` from a checkout | clones the new tag into a temporary directory and runs its `install.sh` with the same install root and bin directory |
| pip, or a source checkout | nothing: it refuses and names what to run (`pip install --upgrade …`, or update the checkout with git) |

## Safety

- Each release carries the wheel, `install.sh`, `install.ps1` and `SHA256SUMS`. A release-installer update runs the new
  `install.sh` only when its SHA-256 matches the release's `SHA256SUMS`, and that script installs the wheel only when the
  wheel matches too; a mismatch installs nothing. The shim is replaced atomically at the end.
- uv and pipx git installs rest on the exact tag fetched from GitHub over HTTPS. For checkout `install.sh` installs, the
  cloned tag's `__version__` must equal the release version before its `install.sh` runs.
- uv and pipx own their environments; `uio` never writes into them. A failed `uv tool install --force` can remove the
  old environment, so after any failed uv or pipx install `uio` reinstalls the running version's tag.
- A failed checkout `install.sh` update puts the previous shim back and removes the half-written release folder. A
  failed release-installer update leaves the running version in place (the private venv is swapped only after the new
  one starts), and if the shim no longer runs it, `uio` reinstalls the running version's release.
- One rollback is kept: the previous tag for uv, pipx and the release installer (a record only, no files), or the
  previous release folder for checkout `install.sh` installs. That folder is removed after the new version has run
  successfully once, or after 7 days at most (`--rollback` then reinstalls the previous tag). Older release folders are
  removed when an update installs.
- The temporary clone or download folder is removed on success and on failure.
- Running work is not touched: the new version applies from the next start, and an automatic install waits while a
  `uio tune` holds `~/.cache/uio/lock`.

## Automatic installs

Off by default; turn them on with `uio update --auto on` or by answering `y` after an update. When on, the daily
background check installs a new release at low priority and writes one line per attempt to
`~/.cache/uio/update.log` (one file, at most 256 KB). The new version applies from the next start. A failed automatic
install leaves the current version working, records the cause (`uio update --auto status`) and brings the notice
back for that version; it is not retried until a newer release appears.

## Uninstall

Run `uio update --clear-state`, then `uv tool uninstall ultra-inference-optimization` (or `pipx uninstall
ultra-inference-optimization`; for `install.sh`, remove `~/.local/share/uio` and the `~/.local/bin/uio` shim).

## Test override

`UIO_UPDATE_API=<http(s) URL>` points the check at another release record, for tests and manual proofs.
