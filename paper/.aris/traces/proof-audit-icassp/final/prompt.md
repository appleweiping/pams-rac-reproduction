# Fresh proof-audit prompt

Act as a fresh, zero-context ARIS proof-audit reviewer for the final ICASSP manuscript in `D:/Project/rac-paper-writing-icassp27`.

- Audit only the current `paper/main.pdf`, `paper/main.tex`, `paper/preamble.tex`, `paper/math_commands.tex`, and `paper/sections/*.tex`.
- Require `paper/main.pdf` SHA-256 to equal `873c83da0dbf4b24a6e1199f3dcd980e17d0b032580e0591f88a7c545ec67a45`.
- Read every in-scope source file and visually inspect every PDF page.
- Search for theorem, lemma, proposition, corollary, proof, and any implicit theoretical guarantee.
- Audit the precision claims surrounding all four displayed equations.
- Return `NOT_APPLICABLE` when there is no formal proof obligation, while still reporting any mathematical or guarantee overstatement.
- Do not inspect old proof audits, history, or logs. Do not modify manuscript source.
- Write the final trace and the public Markdown/JSON proof-audit reports with current input hashes.

Reviewer identity: `/root/icassp_proof_audit`; model `gpt-5.6-sol`; reasoning `xhigh`; independence `same-family`.
