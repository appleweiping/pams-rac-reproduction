# Fresh experiment-integrity review

**Verdict: FAIL**  
**Reason:** `NO_MULTI_PERSON_REAL_GT_EVIDENCE`

Formal UCFRep dev84 scoring uses official digest-bound real targets and all 17
checked-in evaluations recompute; 47 focused tests passed. The evidence remains
single-person development evidence. No MultiRep bundle or tracked proposed
multi-person router/normalized-overlap-add implementation exists, so zero local
artifacts are ICASSP-main-table eligible. Test105 is unscored: no predictions,
evaluation, or metrics exist, although an older development CLI deserialized
the full manifest. Detailed A--F findings and claim impacts are frozen in
`EXPERIMENT_AUDIT.md` and `EXPERIMENT_AUDIT.json`.

