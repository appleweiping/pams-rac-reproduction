---
audit_skill: citation-audit
verdict: PASS
reason_code: ALL_16_FINAL_CONTEXTS_SUPPORTED
review_independence: fresh_zero_context_same_family
acceptance_status: provisional
agent_id: /root/user_revision_citation_audit
generated_at: 2026-08-15T01:30:00+08:00
---

# Final citation audit

The citation audit passes on the exact final artifact.

| Artifact | SHA-256 |
|---|---|
| `paper/main.pdf` | `2976344081abbceb84c4114068bd7b37100ae7aa8438d3112b5227bee442731e` |
| `paper/references.bib` | `c84cb6df17ac4133f7c3115e1aa4dcad82e9e850f1ebdc5e6979cbc1cfd917b3` |
| `paper/sections/1_introduction.tex` | `7ce1329707dae209f745c200492253356e975448d9b32053b9f97d48bdcf3b3e` |
| `paper/sections/4_experiments.tex` | `cd6508a60e3e9cac767ccfcec53e019a52525f58cebc2b1559819139abe0ecb5` |

The active source contains 16 distinct citation keys and the bibliography
contains exactly the same 16 entries. Missing entries, unused entries,
duplicate keys, duplicate DOIs, and unresolved citation markers are all zero.
All references were checked against DBLP, Crossref, CVF, IEEE, Springer,
OpenReview, or the official publisher/author page rather than generated from
memory. All 16 current contexts support the sentence in which they occur.

The final set covers classic repetition counting, modern learned single-person
RAC, pose/low-label approaches, both direct multi-person baselines, local
spectral/irregular-sampling and overlap-add foundations, sparse experts, and
tracking metrics. For a four-page ICASSP paper this set is sufficient and
focused; adding unrelated entries would reduce rather than improve quality.

The replacement of unsupported preregistration wording in the experiments
section changes no citation claim. References [1]--[16] are visible across
pages 4--5, and page 5 contains references only.

`CITATION_AUDIT_VERDICT: PASS`

Because the reviewer is from the same model family and the Claude overlay did
not respond, this PASS remains provisional under the ARIS assurance policy.
