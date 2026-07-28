# Designated server runbook

The public repository contains no server address, username, password or SSH
key. Infrastructure coordinates are supplied privately at execution time.
Password settings are not changed by this project.

## Access precondition

The target SSH daemon must accept the operator's registered public key. A
network proxy cannot replace SSH authentication. Until key authentication is
available, all benchmark stages remain `resource-blocked`.

Private infrastructure coordinates and key paths are never written to the
repository, command logs, or run manifests.

## Current execution status

The operator has recorded a designated-server component smoke for the current
pose preprocessing revision: the strict 526-video manifest, frozen 337/84/105
split, one real-video MediaPipe extraction, exact cache-resume check,
all 421 official training-pool pose caches, synthetic gates, and the CUDA test
suite are summarized under `results/server-smoke`. The path-free 421-cache
audit and clean identity ledger are published there; raw videos, pose arrays,
and server logs are not. Thirteen training videos are all-invalid and remain
present as zero-valued masked samples under the frozen failure policy. This has
produced no UCFRep metric. Encoder/SSHead training, sealed evaluation, baselines,
and executable Table 2 variants remain incomplete.

## Read-only inventory

Before installing or training, record:

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
nvidia-smi topo -m
python3 --version
df -h
free -h
uname -a
```

The resulting non-secret properties enter the run manifest. The inventory
selects only physical microbatch size and whether CountLLM-Lite is eligible;
it does not change frozen model or evaluation parameters.

## Environment

The reviewed source checkout is mounted read-only into a pinned container.
The image does not bake in project source, so the Git SHA in the checkout is
the code that runs:

```bash
export PAMS_EXPECTED_ROOT=/absolute/approved/path/pams-rac
mkdir -p "$PAMS_EXPECTED_ROOT"
git clone https://github.com/appleweiping/pams-rac-reproduction.git \
  "$PAMS_EXPECTED_ROOT/repo"
cd "$PAMS_EXPECTED_ROOT/repo"
git checkout <reviewed-commit>
bash scripts/server/build_image.sh
bash scripts/server/run_gpu1.sh --sealed -- \
  python -m pytest -q -p no:cacheprovider
bash scripts/server/run_gpu1.sh --sealed -- \
  python -m ruff check --no-cache .
```

`docker/server/Dockerfile` pins the official PyTorch CUDA base by digest, the
Ubuntu archive at snapshot `20260727T000000Z`, exact requested Debian package
versions, and the complete Python dependency set. `PAMS_BASE_IMAGE` may name an
already-mirrored repository only when it ends in that same immutable digest.
`PAMS_APT_MIRROR`, `PAMS_PULL_BASE=0`, and `PAMS_CLEAR_PROXY=1` are explicit
build-time escape hatches for restricted networks; their actual private
values must not contain credentials. Custom-mirror build logs remain private
infrastructure artifacts and must be redacted before any publication.
`PAMS_APT_SNAPSHOT` cannot move in a formal build.

The launcher exposes only one configured GPU (GPU 1 by default), mounts
source and raw data read-only, and keeps caches, checkpoints, logs, runs, and
artifacts under the designated data root. Supply that private absolute path
at runtime as `PAMS_EXPECTED_ROOT`; the launcher refuses an unset, relative,
or mismatching root. Both build and launch require a clean checkout. The
launcher verifies the image's source-revision label and a deterministic hash
of the Dockerfile/lock/wrapper inputs against that checkout, then injects the
immutable Docker image ID and both labels into every run manifest.
Every invocation also holds a per-device project `flock`. GPU optimizer and
evaluation commands must add `--wait-for-idle-gpu`; this requires three
consecutive 30-second samples below the frozen memory, utilization, and
heavy-process thresholds, and exits 75 after 12 hours. This project-local
lock serializes PAMS launchers only. The idle samples reduce accidental
contention with unrelated jobs, but they are not a server-wide scheduler and
cannot prevent another user from starting work after the final sample.
`--sealed` disables container network access and forces model-hub offline
mode. Record `docker image inspect`, `pip freeze`, `pip check`, the driver,
and GPU properties in the environment artifact.

Before a formal checkpoint is trained, retain the exact image in a public
container registry. Record both the registry manifest digest and Docker image
ID; checkpoints bind the latter, while users pull the former. Rebuilding from
the same Dockerfile is not a substitute because Ubuntu package mirrors can
change. A release is incomplete unless its exact image remains pullable by
digest (or an SHA-256-verified OCI archive is published).

## Data and pose

The launcher maps all dataset state outside the Git checkout:

```bash
export PAMS_UCFREP_VIDEO_ROOT=/pams/data/extracted
export PAMS_ANNOTATIONS=/pams/data/annotations/ucf526_annotations.zip
export PAMS_POSE_CACHE_DIR=/pams/pose-cache
export PAMS_RUNS_DIR=/pams/runs
```

Prepare and validate the canonical manifest:

```bash
bash scripts/server/run_gpu1.sh --sealed -- \
  pams data prepare-ucfrep "$PAMS_UCFREP_VIDEO_ROOT" \
    --annotations "$PAMS_ANNOTATIONS" \
    --output "$PAMS_RUNS_DIR/manifests/ucfrep_526.json" \
    --split-dir "$PAMS_RUNS_DIR/splits"
bash scripts/server/run_gpu1.sh --sealed -- \
  pams data validate "$PAMS_RUNS_DIR/manifests/ucfrep_526.json"
bash scripts/server/run_gpu1.sh --sealed -- \
  pams data split "$PAMS_RUNS_DIR/manifests/ucfrep_526.json" \
    --output "$PAMS_RUNS_DIR/manifests/ucfrep_337_84_105.json"
bash scripts/server/run_gpu1.sh --sealed -- \
  pams pose extract "$PAMS_RUNS_DIR/manifests/ucfrep_526.json" \
    "$PAMS_POSE_CACHE_DIR" --config configs/pams.yaml \
    --split train --skip-existing \
    --failure-ledger "$PAMS_RUNS_DIR/pose-train-failures.json"
```

Any decode/pose failure is recorded. A run lacking all 105 sealed predictions
is incomplete rather than evaluated on a smaller denominator.

## Synthetic safety preflight

Run and retain both label-free diagnostics before a GPU optimizer smoke:

```bash
bash scripts/server/run_gpu1.sh --sealed -- \
  pams synthetic counter-gate \
    --output /pams/artifacts/counter-sign-phase-gate.json
bash scripts/server/run_gpu1.sh --sealed -- \
  pams synthetic period-counter-gate \
    --output /pams/artifacts/period-counter-sign-phase-gate.json
bash scripts/server/run_gpu1.sh --sealed -- \
  pams synthetic sshead-collapse \
    --output /pams/artifacts/sshead-collapse-diagnostic.json
```

Both counter commands must pass. The first isolates multi-expert counting
with an exact synthetic period; the second also exercises FFT/autocorrelation
period estimation. Neither validates the encoder or Period Head. The collapse
command is an adverse diagnostic and
exits nonzero while the inferred loss retains its known constant-stream dead
point or near-collapse gradient spike. Its failure must be published, not
suppressed. Actual SSHead training independently checks finite values and
aborts before an optimizer step if its observed stream/gradient reaches those
conditions.

## GPU encoder integration smoke

Before training, audit the canonical 421-cache pool against a clean
`--skip-existing` identity ledger:

```bash
bash scripts/server/run_gpu1.sh --sealed -- \
  python scripts/server/audit_pose_cache.py \
    /pams/runs/manifests/ucfrep_526.json \
    /pams/pose-cache \
    configs/pams.yaml \
    /pams/runs/audits/pose421-identity-ledger.json
```

The audit rejects missing or extra caches, source-video or pose-fingerprint
mismatches, malformed NPZ arrays, and non-finite values. It emits only a
path-free aggregate JSON summary.

After that audit passes, exercise one complete physical
batch/KMeans/PAMS-TCC epoch and one real-cache inferred-SSHead epoch with the
dedicated two-stage smoke configuration:

```bash
bash scripts/server/run_gpu1.sh --sealed --wait-for-idle-gpu -- \
  pams train encoder \
    /pams/runs/manifests/ucfrep_526.json \
    /pams/pose-cache \
    /pams/checkpoints/smoke/two-stage-1epoch/encoder \
    --config configs/smoke/pams_two_stage_1epoch.yaml \
    --device cuda:0 --microbatch-size 32

bash scripts/server/run_gpu1.sh --sealed --wait-for-idle-gpu -- \
  pams train sshead \
    /pams/checkpoints/smoke/two-stage-1epoch/encoder/encoder.pt \
    /pams/runs/manifests/ucfrep_526.json \
    /pams/pose-cache \
    /pams/checkpoints/smoke/two-stage-1epoch/sshead \
    --encoder-progress \
      /pams/checkpoints/smoke/two-stage-1epoch/encoder/logs/encoder.jsonl \
    --config configs/smoke/pams_two_stage_1epoch.yaml \
    --device cuda:0 --microbatch-size 8
```

Inside the one-GPU container, `cuda:0` is the host GPU selected by the
launcher. Both checkpoints are `smoke_only` and cannot enter any benchmark
table. Validate both completion receipts without remaps and retain their
terminal progress/checkpoint hashes before formal training.

## Training order

For seeds 42, 2026 and 3407:

1. copy `configs/pams.yaml` and change only `seed`;
2. train the encoder checkpoint;
3. retain that checkpoint unchanged as `PAMS-Literal`;
4. freeze the encoder and train `PAMS-SSHead`;
5. store logs, checkpoint hashes and immutable manifests;
6. run development diagnostics only after configuration freeze.

An interrupted run resumes into a new output directory. Pass the prior,
unchanged files with `--resume`, `--resume-checkpoint <old.pt>`, and
`--resume-progress <old.jsonl>`; never point a resumed run at an output
directory referenced by an existing completion receipt.

At completion, every input or registry receipt outside the run directory is
copied, with its already-captured SHA-256 enforced, into
`inputs/receipt-artifacts/`. The schema-v3 completion receipt binds those
portable snapshots instead of host-specific paths. Moving the complete run
tree and running `pams data validate-run` therefore requires no artifact
remaps; the shared sealed-attempt registry remains the authoritative
one-attempt ledger.

After all assumptions and checkpoints are frozen, retrain on all 421 training
videos and permit the evaluator one sealed 105-video pass per seed. With the
337/84/105 manifest, sealed evaluation requires `--include-dev`; the
421/105 manifest already denotes the full training pool.

### Formal 421-video commands

The following template uses only the generic host-root variable and fixed
container mount points. It contains no infrastructure coordinate or secret.
Run it from the clean reviewed checkout after the current pose extraction
finishes with zero failures. The generated seed configs live outside the Git
checkout, so creating them does not dirty the reviewed source. Run all three
blocks in the same host shell because the later blocks reuse the variables
declared by the first:

```bash
set -euo pipefail

readonly PAMS_HOST_CONFIG_DIR="$PAMS_EXPECTED_ROOT/runs/configs"
readonly PAMS_CONTAINER_CONFIG_DIR=/pams/runs/configs
readonly PAMS_FINAL_MANIFEST=/pams/runs/manifests/ucfrep_337_84_105.json
readonly PAMS_FINAL_POSE_CACHE=/pams/pose-cache
readonly PAMS_ATTEMPT_REGISTRY=/pams/runs/sealed-test-attempts/ucfrep_526
PAMS_SSHEAD_MICROBATCH="${PAMS_SSHEAD_MICROBATCH:-8}"

test "$(grep -c '^seed:' configs/pams.yaml)" -eq 1
mkdir -p "$PAMS_HOST_CONFIG_DIR"
for seed in 42 2026 3407; do
  sed -E "s/^seed: [0-9]+$/seed: ${seed}/" configs/pams.yaml \
    > "$PAMS_HOST_CONFIG_DIR/pams-seed-${seed}.yaml"
  bash scripts/server/run_gpu1.sh --sealed -- \
    pams config validate "$PAMS_CONTAINER_CONFIG_DIR/pams-seed-${seed}.yaml"
done
```

Train all three terminal 150-epoch encoders and inferred 30-epoch SSHeads
before reading the sealed test labels. Encoder training requires one physical
batch of 32. The SSHead microbatch may be any positive divisor of 32 selected
from the hardware inventory; gradient accumulation preserves its effective
batch of 32.

```bash
set -euo pipefail

for seed in 42 2026 3407; do
  config="$PAMS_CONTAINER_CONFIG_DIR/pams-seed-${seed}.yaml"
  encoder_run="/pams/checkpoints/ucfrep_526/seed-${seed}/encoder150"
  sshead_run="/pams/checkpoints/ucfrep_526/seed-${seed}/sshead30"

  bash scripts/server/run_gpu1.sh --sealed --wait-for-idle-gpu -- \
    pams train encoder \
      "$PAMS_FINAL_MANIFEST" \
      "$PAMS_FINAL_POSE_CACHE" \
      "$encoder_run" \
      --config "$config" \
      --device cuda \
      --epochs 150 \
      --microbatch-size 32 \
      --include-dev

  bash scripts/server/run_gpu1.sh --sealed --wait-for-idle-gpu -- \
    pams train sshead \
      "$encoder_run/encoder.pt" \
      "$PAMS_FINAL_MANIFEST" \
      "$PAMS_FINAL_POSE_CACHE" \
      "$sshead_run" \
      --encoder-progress "$encoder_run/logs/encoder.jsonl" \
      --config "$config" \
      --device cuda \
      --epochs 30 \
      --microbatch-size "$PAMS_SSHEAD_MICROBATCH" \
      --include-dev
done
```

Do not prepare the sealed-test pose inputs until all six terminal checkpoints,
progress logs, hashes, and completion receipts have been reviewed and frozen.
The pose process consumes only the count-free/action-field-free sidecar:

```bash
set -euo pipefail

readonly PAMS_TEST_POSE_INPUTS=/pams/runs/manifests/ucfrep_526_test_pose_inputs.json
readonly PAMS_TEST_POSE_LEDGER=/pams/runs/pose-test-identity-ledger.json

bash scripts/server/run_gpu1.sh --sealed -- \
  pams pose extract \
    "$PAMS_TEST_POSE_INPUTS" \
    "$PAMS_FINAL_POSE_CACHE" \
    --config configs/pams.yaml \
    --label-free-manifest \
    --skip-existing \
    --failure-ledger "$PAMS_TEST_POSE_LEDGER"

# A second identity-only resume must report 105 exact skips and zero failures.
bash scripts/server/run_gpu1.sh --sealed -- \
  pams pose extract \
    "$PAMS_TEST_POSE_INPUTS" \
    "$PAMS_FINAL_POSE_CACHE" \
    --config configs/pams.yaml \
    --label-free-manifest \
    --skip-existing \
    --failure-ledger "$PAMS_TEST_POSE_LEDGER"

bash scripts/server/run_gpu1.sh --sealed -- \
  python scripts/server/audit_pose_cache.py \
    "$PAMS_TEST_POSE_INPUTS" \
    "$PAMS_FINAL_POSE_CACHE" \
    configs/pams.yaml \
    "$PAMS_TEST_POSE_LEDGER" \
    --split test \
    --label-free-manifest \
    --allow-extra-caches
```

The test audit verifies all 105 source/cache hashes and reports the 421
already-audited training caches as allowed out-of-scope extras. It never loads
the labelled dataset manifest.

Only after that audit passes may the following commands consume the sole legal
sealed attempt for each method/seed pair.

```bash
set -euo pipefail

for seed in 42 2026 3407; do
  config="$PAMS_CONTAINER_CONFIG_DIR/pams-seed-${seed}.yaml"
  encoder_run="/pams/checkpoints/ucfrep_526/seed-${seed}/encoder150"
  sshead_run="/pams/checkpoints/ucfrep_526/seed-${seed}/sshead30"
  literal_output="/pams/artifacts/ucfrep_526/seed-${seed}/pams-literal-test"
  sshead_output="/pams/artifacts/ucfrep_526/seed-${seed}/pams-sshead-test"

  bash scripts/server/run_gpu1.sh --sealed --wait-for-idle-gpu -- \
    pams evaluate checkpoint \
      "$encoder_run/encoder.pt" \
      "$PAMS_FINAL_MANIFEST" \
      "$PAMS_FINAL_POSE_CACHE" \
      "$literal_output" \
      --variant literal \
      --checkpoint-progress "$encoder_run/logs/encoder.jsonl" \
      --config "$config" \
      --split test \
      --device cuda \
      --include-dev \
      --method-id pams-literal \
      --sealed-attempt-registry "$PAMS_ATTEMPT_REGISTRY"

  bash scripts/server/run_gpu1.sh --sealed --wait-for-idle-gpu -- \
    pams evaluate checkpoint \
      "$sshead_run/sshead.pt" \
      "$PAMS_FINAL_MANIFEST" \
      "$PAMS_FINAL_POSE_CACHE" \
      "$sshead_output" \
      --variant sshead \
      --checkpoint-progress "$sshead_run/logs/sshead.jsonl" \
      --upstream-encoder-checkpoint "$encoder_run/encoder.pt" \
      --upstream-encoder-progress "$encoder_run/logs/encoder.jsonl" \
      --config "$config" \
      --split test \
      --device cuda \
      --include-dev \
      --method-id pams-sshead \
      --sealed-attempt-registry "$PAMS_ATTEMPT_REGISTRY"
done
```

`configs/ablations/table2.yaml` is currently a preregistration ledger, not an
executable ablation configuration. The five non-full variants remain blocked
until their switches, per-row config identity, checkpoint provenance and
sealed method IDs are implemented. Do not manufacture Table 2 rows by
relabelling a full-model checkpoint.

For `split=test`, `pams evaluate checkpoint`
derive the only legal attempt registry from
`$PAMS_RUN_ROOT/sealed-test-attempts/<protocol>`. The launcher fixes
`PAMS_RUN_ROOT=/pams/runs`; an explicitly supplied
`--sealed-attempt-registry` must resolve to that same path. Reservation is
atomic and keyed by frozen method plus experiment seed, so changing output
directories or checkpoints cannot create a second legal attempt. Test
evaluation also refuses a dirty Git worktree. A failed label-free preflight
(cache, checkpoint, prediction IDs, or finite prediction values) occurs
before the attempt is consumed.

`pams report metrics` contains the same atomic machinery for future baseline
adapters, but its production baseline allowlist is currently empty. A method
is admitted only after runnable parity and a method-specific provenance gate;
an arbitrary prediction JSON cannot claim a paper method name.
UCFRep-pose-110 sealed scoring is likewise disabled until its exact 89/21
official annotation identity is frozen.

## Publication

Copy only environment locks, manifests, logs, metrics and per-video prediction
CSVs into a result PR. Upload weights as GitHub Release assets after hashing.
Push the exact formal image to the public registry and record its immutable
registry digest, Docker image ID, package inventory, and source revision.
Run the repository verification gate; publish `v1.0-verified` only when every
gate condition passes, otherwise publish a clearly labelled partial release.
