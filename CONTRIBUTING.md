# Contributing

Contributions are welcome when they keep `ultra-inference-optimization` (`uio`) small, honest about measurement, and dependency-free at runtime. Read [README.md](README.md), [docs/METHOD.md](docs/METHOD.md), and [SECURITY.md](SECURITY.md) first.

## Rules

- Keep the runtime pure Python 3.10+ standard library with zero third-party dependencies.
- Keep measurements honest: count tokens from server `usage`, keep prompts cold with unique prefixes, and never weaken statistical or correctness gates.
- Do not embed model IDs, publishers, quantization defaults, personal paths, or private operational details in source, docs, or tests. `NeutralityTests` checks the whole tree.
- Do not add GitHub Actions; project checks run locally.

## Verify a change

Run the full test suite and single-file checks from the repository root:

```sh
python3 -m unittest discover -s tests
python3 -m unittest tests/test_bench.py
```

When changing `install.sh`, CLI contracts, or statistical/gate logic, add focused unit tests covering both the happy path and refusal/failure paths.

## Pull requests

Describe the user-visible problem, the chosen fix, and the exact verification commands run. Never commit credentials, local profiles, GGUF models, or run directories.

By contributing, you agree that your contribution is licensed under this repository's MIT License ([LICENSE](LICENSE)).
