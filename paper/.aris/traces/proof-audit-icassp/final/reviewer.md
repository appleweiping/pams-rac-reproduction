# Fresh ARIS proof audit

## Verdict

`NOT_APPLICABLE` (`NO_FORMAL_PROOF_OBLIGATION`).

The five-page manuscript contains no theorem, lemma, proposition, corollary, proof environment, proof sketch, asymptotic result, convergence statement, optimality claim, or formal error bound. Its four numbered equations are proposal-level signal-path definitions, and the paper repeatedly marks implementation-sensitive semantics as `SYNC-REQUIRED` while withholding empirical claims. There is therefore no formal proof obligation to accept or reject.

The audited PDF SHA-256 is `873c83da0dbf4b24a6e1199f3dcd980e17d0b032580e0591f88a7c545ec67a45`, matching the required final input.

## Review procedure and independence

I read the current `main.tex`, `preamble.tex`, `math_commands.tex`, and every current `sections/*.tex` file in full. I rendered the current PDF afresh and visually reviewed pages 1 through 5. I also searched both source and freshly extracted PDF text for formal-proof vocabulary and guarantee-like wording. I did not inspect prior proof audits, audit history, or logs. All four displayed equations are legible in the rendered PDF; no formula is clipped, obscured, or visually corrupted.

## Equation-by-equation precision audit

1. **Equation (1), masked identity-indexed representation.** This is a typing and dependency specification, not a theorem. Written as a per-identity mapping, it supports the structural statement that the displayed encoder call has no other identity as an argument. The stronger implementation invariants (masks at every transformation, no cross-identity windows, buffer lifecycle) are explicitly source-pending and assigned future tests. No proof-backed performance or correctness guarantee is asserted.

2. **Equation (2), local masked periodic evidence.** The displayed autocorrelation and Fourier score are regularized definitions, not estimators with a proved bias, consistency, or recovery guarantee. The summation support, the definition of `q(t+tau)` outside a window, lag/frequency units, admissible domains, and the map `Phi` remain open; the prose explicitly marks these items `SYNC-REQUIRED`, requires out-of-range masks and minimum lag-pair support, and flags pair-dependent centering or an irregular-sample alternative. Calling the expressions "non-circular" and "normalized" is therefore a specification-level description, not an exact theorem. Non-blocking precision note: until those domains and extensions are written, Equation (2) is not a fully closed mathematical definition.

3. **Equation (3), soft routing.** For defined finite logits and nonzero routing temperature, softmax gives nonnegative weights summing to one, so the simplex constraint is sound. The manuscript does not state the temperature domain, but marks temperature and implementation details `SYNC-REQUIRED`. Non-blocking wording caveat: "Soft weights retain uncertainty at tempo transitions" is not entailed by softmax; logits can be arbitrarily concentrated at a transition. It should be read as a representational capability or hypothesis, not a guarantee. The paper's collapse and hard/uniform-routing controls appropriately treat this behavior as empirical.

4. **Equation (4), normalized overlap-add and one decode.** The ordering really does aggregate continuous window responses before a single trajectory-level decoder, so it removes the specific act of independently decoding every overlapping window. The text correctly limits this structural claim and explicitly says the order "does not guarantee one peak." Non-blocking precision note: because the denominator is `sum weights + epsilon`, the operation does not exactly preserve a constant response at finite positive coverage; thus "normalized" cannot be interpreted as an exact perfect-reconstruction guarantee. The manuscript makes no such exact-amplitude claim, and it conditions decoding on aligned timestamps and positive coverage.

## Hidden guarantee audit

- The abstract's claim that the order removes "one structural source" of boundary double-counting is appropriately narrow and is consistent with Equation (4); it does not promise complete elimination of double-counting.
- The statement that shared experts keep parameter count independent of the number of tracks is correct for the fixed shared expert bank as specified. The manuscript explicitly withholds runtime and memory claims.
- "Fast/medium/slow" specialization, robustness under tempo changes, stationary-track preservation, period-confidence validity, and mechanism isolation are hypotheses or planned empirical checks, not established guarantees.
- The planned time-warp and paired analysis do not create a proof obligation; causal interpretation remains conditional on the frozen warp, support, interpolation, and exclusion contracts that the manuscript itself requires.
- The proposed auxiliary losses are explicitly only a synchronization contract and are to be deleted if absent, so they are not represented as an implemented or theoretically justified objective.

## Decision

No blocking mathematical misstatement or unsupported formal guarantee was found. The precision caveats above should remain visible when source-bound definitions are finalized, but they do not convert this proposal-level, pre-results manuscript into a proof-bearing paper. Acceptance status is provisional because the review is same-family and the implementation-sensitive interfaces are intentionally unresolved.
