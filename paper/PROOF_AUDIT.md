# Proof audit

## Verdict

`NOT_APPLICABLE` — reason code `NO_FORMAL_PROOF_OBLIGATION`.

The audited five-page manuscript has no theorem, lemma, proposition, corollary, proof, proof sketch, convergence result, optimality claim, formal bound, or other proof-bearing statement. The four numbered equations define a proposal-level signal path; implementation-sensitive semantics are explicitly marked `SYNC-REQUIRED`, and empirical claims are withheld. No proof-related blocker or unsupported formal guarantee was found.

Audited PDF SHA-256: `873c83da0dbf4b24a6e1199f3dcd980e17d0b032580e0591f88a7c545ec67a45` (required hash matched). Pages 1–5 were freshly rendered and visually reviewed; all equations were legible and unobscured.

## Formula and guarantee findings

- **Equation (1):** a per-identity typing/dependency specification. It does not prove implementation invariants; the manuscript correctly leaves mask propagation, window geometry, and lifecycle behavior source-pending and test-gated.
- **Equation (2):** no recovery, bias, consistency, or confidence-calibration guarantee is claimed. Its support, out-of-window indexing, units/domains, centering choice, and `Phi` remain open and are explicitly synchronization-gated. This is a non-blocking precision gap, not a hidden theorem.
- **Equation (3):** the simplex constraint follows from a defined softmax, subject to a defined nonzero temperature. The sentence that soft weights "retain uncertainty" at tempo transitions is not guaranteed by softmax and should be read as a capability/hypothesis; the planned collapse controls make the empirical status clear.
- **Equation (4):** reconstruct-before-decode structurally avoids independent per-window decoding, and the text expressly denies any one-peak guarantee. Because of the added `epsilon`, the displayed overlap-add is not an exact constant-preserving/perfect-reconstruction normalization; the paper does not claim that stronger property.

Other potentially guarantee-like statements are adequately scoped: parameter count, not runtime or memory, is independent of track count for the fixed shared expert bank; robustness and stationary-track behavior are hypotheses; auxiliary losses are only a synchronization contract; and the planned time-warp diagnostic remains artifact- and protocol-gated.

The detailed reviewer trace is at `paper/.aris/traces/proof-audit-icassp/final/reviewer.md`. Review independence is `same-family`, so acceptance status is `provisional`.
