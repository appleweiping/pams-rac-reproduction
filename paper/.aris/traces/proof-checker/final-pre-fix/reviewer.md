# Fresh proof audit — pre-fix final-freeze attempt

- Agent: `/root/final_proof_audit_freeze`
- Route: fresh zero-context, same-family, `gpt-5.6-sol`, ultra
- Audited source freeze: `3e549abf1a9df6e2a857941537ef7100a7172b9ad61c176019ed62ab38e63388`
- Audited PDF: `a34d0067c43cec5a5831f3656728fd2bffec18243a1a0a3f197d926d071e2889`

## Verdict

`FAIL` — `FALSE_COVERAGE_INVARIANCE_CLAIM`

No formal theorem, lemma, proposition, corollary, or proof environments exist,
and all 14 displayed equations are ordinary method definitions or derivations.
However, the prose attached to Eq. (11) made a false exact algebraic claim about
invariance to window coverage.

## Finding PROOF-001 (high)

`paper/sections/3_method.tex:197-204` defined Eq. (11) with denominator
`sum(a_u m_iℓu) + epsilon`, then stated that normalization removed dependence on
the number of covering windows. For `K` identical valid covering windows with
`a_u = r_iℓu = m_iℓu = m_it = 1`, Eq. (11) gives `R_it = K/(K+epsilon)`, which
varies with `K` whenever `epsilon > 0`. Even at zero stabilizer, changing the
responses being averaged changes the weighted average, so the unqualified
claim was too broad. `paper/sections/4_experiments.tex:84-88` already required
the experiment to expose stabilizer-induced coverage dependence.

Required resolution: qualify the prose as removing direct unnormalized
coverage-count scaling under stated conditions and retain the explicit
stabilizer-bias audit.

## Scope

The reviewer read all 10 manuscript LaTeX files and inspected all 11 PDF pages.
It found zero theorem/proof environments, 14 ordinary displayed equations, and
one hidden formal claim requiring resolution. Source/PDF consistency passed for
this scope. No files were edited by the reviewer.
