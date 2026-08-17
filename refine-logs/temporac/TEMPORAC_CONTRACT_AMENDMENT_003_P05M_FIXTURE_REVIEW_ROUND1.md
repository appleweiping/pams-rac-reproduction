# TempoRAC Amendment 003 P05M Fixture Review — Round 1

## Verdict

`REVISE`

Status: `REVISE_SAME_FAMILY_PROVISIONAL_ZERO_AUTHORITY`.

This is the faithful historical record of the fresh zero-context first review. The exact Round 1 amendment bytes have since been superseded at their canonical paths. This review grants no implementation, S0, gate, capability, launch, data, server, test, training, result, Git, paper, or claim authority.

## Exact reviewed snapshot

| Input | SHA-256 | State |
|---|---|---|
| `TEMPORAC_CONTRACT_AMENDMENT_003_P05M_FIXTURE.md` | `15520072a9e716f148659991cda00aa01d35e070f312ecccb560c687709b7366` | superseded Round 1 bytes |
| `TEMPORAC_CONTRACT_AMENDMENT_003_P05M_FIXTURE.json` | `2b0f6bb8e34b3ad271b60508ab9a08971ed86639b25075c72beb18c3364416c8` | superseded Round 1 bytes |
| `refine-logs/temporac/FINAL_PROPOSAL.md` | `562325dedab0637421f62dec5ad149b3c884f67b47c4b525c8fb54a57bb1233c` | reviewed proposal snapshot |
| `refine-logs/EXPERIMENT_PLAN.md` | `3fd1986871b32674d74b395fbfc9720712d9fba0c8583c24c1d5bddb87086eb6` | reviewed plan snapshot |
| `refine-logs/EXPERIMENT_TRACKER.md` | `c7e2f32c3c81b66884abb13760a671ecc7cdd314fd6c994a4ed4c46cb754a63b` | reviewed tracker snapshot |
| `src/pams/temporac/metrics.py` | `1dadf08e89330bed2ab773afda05bffcb3fe2252b0d3e23c0bcc18c2c3b049dd` | reviewed code |
| `src/pams/temporac/evaluator.py` | `81037186896309b093cd184b6e63542653f5a7e1ab063b78a9e63558369a47d2` | reviewed code |
| `src/pams/temporac/contract.py` | `b71255a35cce1701d2741064064b53ba819b742873a348f1852f147149e36467` | closure dependency |
| `src/pams/temporac/types.py` | `4d2ba91a19b9eddf529d91a366d29072dedcdc5977ba5daf7a1720484259181b` | closure dependency |
| `src/pams/temporac/hashio.py` | `2d81e415b45e3d1b612f998b6a8386a967dfdfcfaac31bc914b2c95cbaeeab3a` | final stable Round 1 reviewed bytes |
| `src/pams/temporac/receipts.py` | `0ac800d96f536a41a67775bc8e8e778fe14c312c02f642f7cde78d210244016d` | final stable Round 1 reviewed bytes |
| `src/pams/temporac/trusted_packer.py` | `40d6e3d5be1b26b7ba0f0f7aadf8da7496c95c4e829face5c7e9faaee2cdf94e` | closure dependency |
| `src/pams/temporac/gates.py` | `b6c9d7432443d7ad36738a81700746709fd571d475514aa6f1bbcd39be744b4a` | reviewed gate code |
| `src/pams/temporac/fixtures.py` | `a8d10df7c8e3cbcb5684198192d3a553921da0f6aa92f03e54448ee182671406` | reviewed fixture code |

The earlier pre-format values `hashio.py=f6f8e4e6f891c510c561b507d6826bae3cddb12ee9c34d4bade584f98e7fb0bf` and `receipts.py=5f30581bdf31be4f98a891f9631f0ef7eda9ddcd58c2893f064ab277c3d879bd` were an intermediate formatter snapshot, not the Round 1 adjudication binding. The stable Round 1 binding is the `2d81…` / `0ac8…` pair in the table.

## Independent fixture witnesses

The fixture arithmetic itself was coherent:

- fixture input: `1264779925401d8efeecddefbe7245eca892543e923ec7c19ade08286fa062c9`
- materialized 288 rows: `759b344fbc7db6ea78b041c84e23ff0fc59733985775be18f6d5c235e8408dbf`
- component order: `8583d714619e7b1bddc755bac95b91a801a8c97951e382fbfd7f875a70f066f2`
- normal expected output: `50f8376d0ee1e245a8521c7831d6d23635ac8cc74694ceeef1f028c8c48e6622`
- attack oracle: `2e62a7b830f453e0505cdeabb766f8a46c0ab2523321d3b1f933bf7fd2cc5f30`
- reference source: `b23bf344a10f84a583c38e2326efd5ca70d4c4462fbf76715cd32b9afd20922f`
- PCG64 seed: `3783362605739069799`; 10,000 draw-index witness: `bcfccfc96c2781162e3c973dea8f88478b1912553acc730aa52772947ed8aef1`
- route raw `<f8[10000]>`: `8a325a51517a7a7669f4f8d4bdfa4d103204383b5d769e19cad409aba5f9e132`
- clean raw `<f8[10000]>`: `91d1fa29d36ad2e4e41733141d841a756c986b3670617f651315026a7157bdae`
- draw-pair root: `49b83785c436d2abd5cbcda617d142922f27ee21a4e17a39f730135add15cf3e`
- selected comparator counts: global `3567`, uniform `4133`, capacity-control `2300`
- route lower/upper: `2c9b6cb2c926abbf` / `e7a28b2ebae8c23f`; clean upper: `692fa1bd84f6a2bf`
- F21 route/clean point bytes: `deddddddddddbd3f` / `565555555555b5bf`; strongest comparator: `uniform`
- all ordered attacks A00–A16 produced their exact required global failure codes.

The then-current production path did not reproduce the frozen component order. Its route/clean raw hashes were `9d13330237fed79f53a93db19edee3d7a66049333f269936afa325712ee507cd` / `79afebeb7a9b2bd3ba23b45432483eebc6846cd34c3d7c0cced6c37cf19b26b9`, with comparator counts `3636/4145/2219` rather than `3567/4133/2300`.

## Blocking findings

### R1-B1 — historical hashes self-locked P2

The Round 1 amendment treated the implementation hashes inspected while drafting as execution requirements. P2 therefore required a corrected implementation to retain hashes of the known-nonconforming implementation. Any real fix changed those hashes and failed P2; retaining them preserved the defect. Historical provenance and fresh candidate authority were not separated.

Required correction: keep old hashes only in a historical audit object, create a separately hashed fresh candidate index, and make P2 compare execution only with that freshly reviewed candidate.

### R1-B2 — the evaluator bundle was not an executable dependency closure

The two-file evaluator bundle digest `dd2a724d4a91088bbd7c2eaf8c71cebd67497502ce618b00982cbf2c5010047e` covered only `metrics.py` and `evaluator.py`, although those files execute local dependencies including contract, types, canonical hashing, receipts, and trusted-packer code. An omitted dependency could change behavior without changing the claimed evaluator code hash.

Required correction: define an ordered raw-byte executable closure, exact preimage and runtime/import boundary, and bind the fresh candidate members rather than historical bytes.

### R1-B3 — the six-key receipt could self-certify; K7 consumed too early

`count_metric_fixture_receipt` accepted caller-provided fixture/expected/attack bytes and a caller-provided evaluator digest, then emitted `status=PASS` without running production, reference, or any attack. A concrete six-key PASS receipt had SHA-256 `a850b50bd40780eee86e0476b9ba03f774ca1ba59b5445f03c0a5f671f55eb27` even while production raw-draw hashes disagreed with the frozen oracle. `OneUseEvaluator.run` also set `_consumed=True` before validating that receipt and its S0/G5a bindings.

Required correction: demote the six-key object to a non-authoritative inner summary; require external runner evidence from fresh production/reference processes and all 17 observations; perform the entire K7 prestart verification before constructing or consuming the one-use evaluator.

## Adjudication

The numerical fixture and non-expansion boundary passed, but the authority chain did not. The Round 1 amendment could not safely authorize P2, S0, G5a, or K7.

No claim, threshold, population, seed, job, GPU-hour allocation, disk class, evidence block, or F23 scientific decision was expanded by this review. Same-family review caps every finding at provisional and authorizes nothing.
