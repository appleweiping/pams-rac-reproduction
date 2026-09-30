## Summary

Describe the implementation or experiment change and its evidence class:
`disclosed`, `recovered`, `inferred`, or `diagnostic`.

## Validation

- [ ] `ruff check .`
- [ ] `pytest`
- [ ] No dataset, checkpoint, credential, or private server detail is committed
- [ ] Protocol/configuration changes produce a new fingerprint
- [ ] Result claims include per-video predictions and an immutable run manifest
- [ ] Oracle diagnostics are excluded from fair tables

## Claim status

- [ ] implementation only
- [ ] partial reproduction
- [ ] verified reproduction (attach gate evidence)
