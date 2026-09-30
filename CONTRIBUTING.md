# Contributing

Changes must preserve the distinction between author-disclosed behavior,
inferred reproduction choices, and diagnostic/oracle experiments.

Before opening a pull request:

```bash
ruff check .
pytest
```

Do not commit videos, datasets, checkpoints, credentials, server addresses
with passwords, or third-party code without a compatible licence. New
baseline entries must declare their protocol, supervision, implementation
status, checkpoint provenance and whether their reported source split is
comparable to either fair table.
