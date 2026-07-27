# Designated server runbook

The public repository contains no server address, username, password or SSH
key. Infrastructure coordinates are supplied privately at execution time.
Password settings are not changed by this project.

## Access precondition

The target SSH daemon must accept the operator's registered public key. A
network proxy cannot replace SSH authentication. Until key authentication is
available, all benchmark stages remain `resource-blocked`.

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

```bash
git clone https://github.com/appleweiping/pams-rac-reproduction.git
cd pams-rac-reproduction
git checkout <reviewed-commit>
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,pose]"
ruff check .
pytest
```

CUDA/PyTorch wheels must match the inventoried driver. Record the final
`pip freeze`, not credentials or private package-index URLs.

## Data and pose

Set local paths outside the Git checkout:

```bash
export PAMS_UCFREP_VIDEO_ROOT=/data/ucf101
export PAMS_POSE_CACHE_DIR=/data/pams/pose-cache
export PAMS_RUNS_DIR=/data/pams/runs
```

Prepare and validate the canonical manifest:

```bash
pams data prepare-ucfrep "$PAMS_UCFREP_VIDEO_ROOT" \
  --output "$PAMS_RUNS_DIR/manifests/ucfrep_526.json" \
  --hash-videos
pams data validate "$PAMS_RUNS_DIR/manifests/ucfrep_526.json"
pams data split "$PAMS_RUNS_DIR/manifests/ucfrep_526.json" \
  --output "$PAMS_RUNS_DIR/manifests/ucfrep_337_84_105.json"
pams pose extract "$PAMS_RUNS_DIR/manifests/ucfrep_337_84_105.json" \
  "$PAMS_POSE_CACHE_DIR" --config configs/pams.yaml
```

Any decode/pose failure is recorded. A run lacking all 105 sealed predictions
is incomplete rather than evaluated on a smaller denominator.

## Training order

For seeds 42, 2026 and 3407:

1. copy `configs/pams.yaml` and change only `seed`;
2. train the encoder checkpoint;
3. retain that checkpoint unchanged as `PAMS-Literal`;
4. freeze the encoder and train `PAMS-SSHead`;
5. store logs, checkpoint hashes and immutable manifests;
6. run development diagnostics only after configuration freeze.

After all assumptions and checkpoints are frozen, retrain on all 421 training
videos and permit the evaluator one sealed 105-video pass per seed. Generate
the Table 2 variants from `configs/ablations/table2.yaml`; never tune a failed
variant on the test output.

## Publication

Copy only environment locks, manifests, logs, metrics and per-video prediction
CSVs into a result PR. Upload weights as GitHub Release assets after hashing.
Run the repository verification gate; publish `v1.0-verified` only when every
gate condition passes, otherwise publish a clearly labelled partial release.
