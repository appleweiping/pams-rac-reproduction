# Citation Audit — ICASSP 2027 Pre-results Draft

**Verdict: PASS (same-family / provisional).** Two fresh zero-context reviewers checked all 16 active bibliography keys against primary or official sources and against the frozen five-page PDF. All 16 entries are `KEEP`; all active citation contexts are supported; no `FIX`, `REPLACE`, `REMOVE`, `WEAK`, or `WRONG` item remains.

## Frozen inputs

- PDF SHA-256: `873c83da0dbf4b24a6e1199f3dcd980e17d0b032580e0591f88a7c545ec67a45`
- Bibliography SHA-256: `322a5d16b0078e8b3436608d4c6158b67c5c8922b0dcae94537ea456c9f6f646`
- Introduction SHA-256: `6823651162f59e5f5263949f2d308f594b667cb3cba0044433c39031ae04db9d`
- Experiments SHA-256: `6052c816e94f7d7f50b560d397495a37863b05d240ed5e418e4b6ec4d3aa131b`

## Scope notes

- PAMS supports the cited pose-based, self-supervised period-adaptive starting objective; it does not provide the proposed Track-PAMS or MultiRep protocol.
- Griffin--Lim is cited for weighted overlap-add machinery, not for the proposed identity-indexed counting method.
- “Identity continuity” is a defensible paraphrase of MultiCounter+'s tracking and spatiotemporal-consistency mechanism, not its exact terminology.
- The official MultiCounter record supports `Period-mAP`, `Period-AP50`, `Period-AP75`, `AvgMAE`, and `AvgOBO`; the manuscript uses those canonical names.

The full independent prompts, responses, source links, input hashes, and per-key rulings are archived in `.aris/traces/citation-audit-icassp/group-a-postfix/` and `.aris/traces/citation-audit-icassp/group-b-final/`. Because both reviewers are from the executor's model family and the Claude overlay did not respond, this audit cannot raise the paper above the provisional ceiling.
