# Security policy

## Supported versions

Security fixes are provided for the latest published release of `ultra-inference-optimization` (`uio`).

## Report a vulnerability

Do not open a public issue for an undisclosed vulnerability. Use GitHub's private vulnerability reporting for this repository. Include the affected version, Python version, operating system, reproduction steps, impact, and any minimal fixture needed to reproduce the issue.

Please allow time for a fix and coordinated disclosure before publishing details.

## Security boundaries

- `uio` runs locally using only the Python 3.10+ standard library at runtime. It sends no telemetry, performs no update checks, and makes no network requests except to the local server URL you supply.
- `uio tune --apply` modifies only `extraArgs` in the profile JSON you specify, after creating a unique timestamped backup with mode `0600` and verifying the file has not changed during the run.
- `uio climb` executes the `--metric-cmd` and `--apply-cmd` shell commands you provide. Do not pass untrusted parameter values or command templates.
- `uio ceiling` treats GGUF files as untrusted binary headers and rejects malformed or oversized fields before reading tensor data.
- `install.sh` must be run from a local checkout and refuses to overwrite an existing `uio` binary it did not install.
