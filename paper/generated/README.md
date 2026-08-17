# Generated evidence bindings

`evidence_values.tex` must be produced by
`paper/scripts/generate_evidence_tex.py`. The generator accepts values only when
`paper/evidence/results_manifest.json` is `frozen-eligible`, every referenced
JSON/CSV byte hash matches, the protocol and statistical scope are complete,
and experiment-audit, result-to-claim, and paper-claim audit all pass.

The generated file records the manifest hash and every source-input hash. It may
declare values through `\DeclareResult`, protocol fields through
`\DeclareProtocolField`, prose claims through `\DeclareResultClaim`, and exactly
one controlled abstract sentence through `\DeclareAbstractResult`. A companion
provenance receipt can be requested with `--receipt`.

This pre-results package intentionally has no `evidence_values.tex`. In draft
mode the LaTeX fallbacks render conspicuous pending text. In submission mode a
missing, stale, hand-edited, or unbound generated file is a hard failure.
