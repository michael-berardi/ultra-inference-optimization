# Security policy

## Supported versions

Security fixes are provided for the latest published release of `ultra-inference-optimization` (`uio`).

## Report a vulnerability

Do not open a public issue for an undisclosed vulnerability. Use GitHub's private vulnerability reporting for this repository. Include the affected version, Python version, operating system, reproduction steps, impact, and any minimal fixture needed to reproduce the issue.

Please allow time for a fix and coordinated disclosure before publishing details.

## Security boundaries

- `uio` runs locally using only the Python 3.10+ standard library at runtime. It sends no telemetry. Besides the server URL you supply, it makes one anonymous release check at most once a day, only after a command run in an interactive terminal and never in CI or agent sessions (see `PRIVACY.md`).
- `uio update` installs only a published stable release: through uv or pipx when they own the install (it never writes into their environments), through the tag's `install.sh` after checking the tag's version for checkout installs, or, for one-line installs, through the release's `install.sh` only when it matches the release's `SHA256SUMS` (that script installs the wheel only when it matches too). A failed install puts the running version back; one rollback is kept. Automatic installs are off by default.
- The one-line installer (`install.sh`, `install.ps1`) verifies the wheel (and, on Windows, `install.sh`) against the release's `SHA256SUMS`, fetched over HTTPS, before installing and never overwrites a `uio` command it did not install.
- `uio tune --apply` modifies only `extraArgs` in the profile JSON you specify, after creating a unique timestamped backup with mode `0600` and verifying the file has not changed during the run.
- `uio climb` executes the `--metric-cmd` and `--apply-cmd` shell commands you provide. Do not pass untrusted parameter values or command templates.
- `uio ceiling` treats GGUF files as untrusted binary headers and rejects malformed or oversized fields before reading tensor data.
- `install.sh` must be run from a local checkout and refuses to overwrite an existing `uio` binary it did not install.
