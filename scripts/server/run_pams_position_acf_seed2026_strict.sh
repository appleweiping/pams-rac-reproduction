#!/usr/bin/env bash
set -Eeuo pipefail

# Strict seed-2026 dev84 execution for the independently inferred projected-
# position vector-ACF readout. The sealed test105 split is never evaluated.

readonly ROOT_INPUT="${PAMS_ROOT:-/media/lenovo/data2/pams-rac}"
readonly SOURCE_CHECKOUT_INPUT="${PAMS_SOURCE_CHECKOUT:?PAMS_SOURCE_CHECKOUT is required}"
readonly SOURCE_REVISION="${PAMS_SOURCE_REVISION:?PAMS_SOURCE_REVISION is required}"
readonly ATTEMPT_ID="${PAMS_ATTEMPT_ID:?PAMS_ATTEMPT_ID is required}"
readonly READOUT_CONFIG_SHA256="${PAMS_READOUT_CONFIG_SHA256:-7ef845025be5d4b3bd7e99583bb8df3514bc32b448214f67db9f2bb797bf64e0}"
readonly GPU_DEVICE="${PAMS_GPU_DEVICE:-0}"
readonly IMAGE_ID="${PAMS_IMAGE_ID:-sha256:a6d10131c321c323c00bb85408cb8333503ecbc5d780d5059a73152d54df6005}"
readonly ENVIRONMENT_SHA256="${PAMS_ENVIRONMENT_SHA256:-1ccfd265d829e5160b8af39156193f63622ac73eaefec00b0f9dab40106c3508}"
readonly V8_RUN_ROOT_INPUT="${PAMS_V8_RUN_ROOT:-${ROOT_INPUT}/runs/pams-v8-seed2026/pams-v8-s2026-07192de-20260730t0251z}"

readonly EXPECTED_IMAGE_ID="sha256:a6d10131c321c323c00bb85408cb8333503ecbc5d780d5059a73152d54df6005"
readonly EXPECTED_ENVIRONMENT_SHA256="1ccfd265d829e5160b8af39156193f63622ac73eaefec00b0f9dab40106c3508"
readonly EXPECTED_READOUT_CONFIG_SHA256="7ef845025be5d4b3bd7e99583bb8df3514bc32b448214f67db9f2bb797bf64e0"
readonly READOUT_CONFIG_SEMANTIC_SHA256="b9c03953e66ef0146d607947c53d1e0d54466e3b938c9c314c6abf644c336773"
readonly EXPERIMENT_CONFIG_RELATIVE="configs/experiments/pams_longest_contiguous_track_v8.yaml"
readonly READOUT_CONFIG_RELATIVE="configs/readouts/projected_position_acf_direct_v1.yaml"
readonly EXPERIMENT_CONFIG_SHA256="eb4072a195608757cd2542855b33fb000c987f96c94b83a1e448ab78f97e9374"
readonly ENCODER_SHA256="6817fe468d76c3fa2182dd831695d6fe3d6da208a2b7f8dfa90cffa6872e8053"
readonly ENCODER_PROGRESS_SHA256="aadc8f9bf067b3489efcd80db43c42cf09dba3477c2976bbc16571ff31684622"
readonly ENCODER_COMPLETION_SHA256="f4815e1961ba883c23ef479ddcdef44aea5cad8a50d54e17b8e6a4cccc613f3d"
readonly ENCODER_COMPLETION_NAME="20260730T025521Z-e6e6a2f657aa.completed.json"
readonly TRAIN_POSE_SET_SHA256="f32d718ae922120778f535a6a4f467edba79cafeba90373b3a6a55bf11be6ee2"
readonly DEV_POSE_SET_SHA256="681b0390ff252df9cf5b1e7c41bfcb6ecb8a0c5b4ef341087cacddc1bf5a3de1"
readonly TRAIN_INPUT_SHA256="e238d307b4ed37b2f6c19eefdf7fa8812ec3d231cd2a7091be38e1c279f04207"
readonly TRAIN_COMMIT_SHA256="ce39b1c9038bb1506354de010e46a23053b307b852222332f13df3f79d799044"
readonly DEV_INPUT_SHA256="f21c49569805e4aea326977f6d7ce2964f503fad129a17e95214594364ca4ae7"
readonly DEV_COMMIT_SHA256="a2bb2db86abef64e16a8fe1ac196857ac8e77cc9cd461c568b2193401a8cd169"
readonly TEST_ID_INPUT_SHA256="9eb2a057246b77fc7a1121ba4c9d27bf55177ad2fcd5dbb87237396a4a6015ca"
readonly TEST_ID_COMMIT_SHA256="232d3db7ce09018716594a1f1b332eec47b37410a8d3068e662b026ee2dc9c66"
readonly DEV_TARGET_SHA256="1ab3bc010ccb5d869af7662586594c973e53841838643e1e9b8cc1787b0efab6"
readonly LOCAL_FREQUENCY_EVALUATION_SHA256="b6add8af9c229cd467be6bba282f267d0d70651a1a39b46ba1443016689add97"
readonly LOCAL_FREQUENCY_NMAE="0.5290583435903633"
readonly LOCAL_FREQUENCY_OBO="0.3333333333333333"

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

for value_name in SOURCE_REVISION; do
  value="${!value_name}"
  [[ "$value" =~ ^[0-9a-f]{40}$ ]] \
    || fail "PAMS_${value_name} must be a full lowercase Git SHA"
done
for value_name in READOUT_CONFIG_SHA256 ENVIRONMENT_SHA256; do
  value="${!value_name}"
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] \
    || fail "PAMS_${value_name} must be a lowercase SHA-256"
done
[[ "$IMAGE_ID" =~ ^sha256:[0-9a-f]{64}$ ]] \
  || fail "PAMS_IMAGE_ID must be an immutable image ID"
[[ "$IMAGE_ID" == "$EXPECTED_IMAGE_ID" ]] \
  || fail "position-ACF must use the exact frozen-v8 container environment"
[[ "$ENVIRONMENT_SHA256" == "$EXPECTED_ENVIRONMENT_SHA256" ]] \
  || fail "position-ACF environment fingerprint differs from frozen v8"
[[ "$READOUT_CONFIG_SHA256" == "$EXPECTED_READOUT_CONFIG_SHA256" ]] \
  || fail "position-ACF readout config differs from its frozen bytes"
[[ "$ATTEMPT_ID" =~ ^[a-z0-9][a-z0-9._-]{0,47}$ ]] \
  || fail "PAMS_ATTEMPT_ID must be a lowercase safe slug of at most 48 characters"
[[ "$ATTEMPT_ID" != *".."* ]] || fail "PAMS_ATTEMPT_ID must not contain '..'"
[[ "$GPU_DEVICE" =~ ^[0-9]+$ ]] \
  || fail "PAMS_GPU_DEVICE must be a non-negative integer"
for mount_path in \
  "$ROOT_INPUT" \
  "$SOURCE_CHECKOUT_INPUT" \
  "$V8_RUN_ROOT_INPUT"; do
  [[ "$mount_path" == /* ]] || fail "all host input roots must be absolute"
  [[ "$mount_path" != *","* ]] || fail "Docker bind paths must not contain commas"
done

readonly ROOT="$(realpath -e -- "$ROOT_INPUT")"
readonly SOURCE_CHECKOUT="$(realpath -e -- "$SOURCE_CHECKOUT_INPUT")"
readonly V8_RUN_ROOT="$(realpath -e -- "$V8_RUN_ROOT_INPUT")"
readonly FIREWALL="${ROOT}/runs/firewall/ucfrep-337-84-105-375d8da"
readonly RUN_PARENT="${ROOT}/runs/pams-position-acf-seed2026"
readonly RUN_ROOT="${RUN_PARENT}/${ATTEMPT_ID}"
readonly SOURCE_VIEW="${RUN_ROOT}/source"
readonly SOURCE_RECEIPT="${RUN_ROOT}/source-export.receipt.json"
readonly INPUT_ROOT="${RUN_ROOT}/inputs"
readonly TRAIN_POSE_VIEW="${INPUT_ROOT}/pose-train337"
readonly DEV_POSE_VIEW="${INPUT_ROOT}/pose-dev84"
readonly SYNTHETIC_STAGE="${RUN_ROOT}/stages/synthetic-k2-7"
readonly TRAIN_GATE_STAGE="${RUN_ROOT}/stages/train337-distribution"
readonly PREDICT_STAGE="${RUN_ROOT}/stages/dev-predict"
readonly SCORE_STAGE="${RUN_ROOT}/stages/dev-score"
readonly AUDIT_ROOT="${RUN_ROOT}/audit"
readonly LOG_ROOT="${RUN_ROOT}/logs"
readonly GATE_SCRIPT="${AUDIT_ROOT}/position-acf-predev-gate.py"
readonly LOCK_ROOT="${ROOT}/.pams-gpu-locks"
readonly LOCK_PATH="${LOCK_ROOT}/gpu${GPU_DEVICE}.lock"

readonly ENCODER_ROOT="${V8_RUN_ROOT}/stages/encoder/run"
readonly ENCODER_CHECKPOINT="${ENCODER_ROOT}/encoder.pt"
readonly ENCODER_PROGRESS="${ENCODER_ROOT}/logs/encoder.jsonl"
readonly ENCODER_COMPLETION="${ENCODER_ROOT}/manifests/${ENCODER_COMPLETION_NAME}"
readonly POSE_POOL="${V8_RUN_ROOT}/inputs/pose-train337-dev84"
readonly TRAIN_INPUT="${FIREWALL}/train.inputs.json"
readonly TRAIN_COMMIT="${FIREWALL}/train.inputs.commitment.json"
readonly DEV_INPUT="${FIREWALL}/dev.inputs.json"
readonly DEV_COMMIT="${FIREWALL}/dev.inputs.commitment.json"
readonly TEST_ID_INPUT="${FIREWALL}/test-identity.inputs.json"
readonly TEST_ID_COMMIT="${FIREWALL}/test-identity.inputs.commitment.json"
readonly DEV_TARGET="${FIREWALL}/dev.targets.json"

readonly SYNTHETIC_NAME="pacfsyn-${ATTEMPT_ID}"
readonly TRAIN_GATE_NAME="pacftrain-${ATTEMPT_ID}"
readonly PREDICT_NAME="pacfpred-${ATTEMPT_ID}"
readonly SCORE_NAME="pacfscore-${ATTEMPT_ID}"

RUN_RESERVED=0
CURRENT_STAGE="preflight"
ACTIVE_CONTAINER=""
SOURCE_RECEIPT_SHA256=""
LAST_CONTAINER_EXIT=255

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

assert_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  test -f "$path" || fail "missing frozen input: $path"
  actual="$(sha256_of "$path")"
  [[ "$actual" == "$expected" ]] \
    || fail "SHA-256 mismatch for $path: expected=$expected actual=$actual"
}

write_status() {
  local state="$1"
  local stage="$2"
  local exit_code="$3"
  local temporary="${RUN_ROOT}/status.json.tmp"
  printf \
    '{"schema_version":1,"status":"%s","stage":"%s","exit_code":%s,"updated_utc":"%s"}\n' \
    "$state" \
    "$stage" \
    "$exit_code" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "$temporary"
  mv -- "$temporary" "${RUN_ROOT}/status.json"
}

freeze_failure_manifest() {
  local manifest="${AUDIT_ROOT}/failed-artifact-sha256.txt"
  test -d "$AUDIT_ROOT" || return 0
  (
    cd "$RUN_ROOT"
    find . \
      -type f \
      ! -path './audit/failed-artifact-sha256.txt' \
      ! -name '*.tmp' \
      -print0 \
      | sort -z \
      | xargs -0 -r sha256sum
  ) > "$manifest"
  chmod 0444 "$manifest"
}

on_exit() {
  local exit_code="$?"
  trap - EXIT HUP INT TERM
  set +e
  if [[ -n "$ACTIVE_CONTAINER" ]]; then
    docker inspect "$ACTIVE_CONTAINER" \
      > "${AUDIT_ROOT}/${ACTIVE_CONTAINER}.error.inspect.json" 2>/dev/null
    if [[ "$(
      docker inspect "$ACTIVE_CONTAINER" \
        --format '{{.State.Running}}' 2>/dev/null
    )" == "true" ]]; then
      docker stop --time 10 "$ACTIVE_CONTAINER" >/dev/null 2>&1
    fi
  fi
  if [[ "$exit_code" -ne 0 && "$RUN_RESERVED" -eq 1 ]]; then
    write_status "failed" "$CURRENT_STAGE" "$exit_code" >/dev/null 2>&1
    freeze_failure_manifest >/dev/null 2>&1
    chmod -R a-w "$RUN_ROOT" >/dev/null 2>&1
  fi
  exit "$exit_code"
}

trap on_exit EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

copy_pose_view() {
  local view_name="$1"
  local manifest="$2"
  local destination="$3"
  local expected_count="$4"
  local receipt="${AUDIT_ROOT}/${view_name}.sha256.tsv"
  local count=0
  local video_id
  local digest
  local source
  local target
  local source_sha256

  mkdir -- "$destination"
  : > "$receipt"
  while IFS= read -r video_id; do
    [[ -n "$video_id" ]] || fail "${view_name} contains an empty video_id"
    [[ "$video_id" != *$'\n'* && "$video_id" != *$'\r'* && "$video_id" != *$'\t'* ]] \
      || fail "${view_name} contains a control character in video_id"
    digest="$(printf '%s' "$video_id" | sha256sum | awk '{print $1}')"
    source="${POSE_POOL}/${digest}.npz"
    target="${destination}/${digest}.npz"
    test -f "$source" || fail "missing exact-v8 pose cache for ${video_id}"
    test ! -e "$target" || fail "duplicate pose cache for ${video_id}"
    source_sha256="$(sha256_of "$source")"
    cp --reflink=auto --preserve=mode,timestamps -- "$source" "$target"
    [[ ! "$source" -ef "$target" ]] \
      || fail "${view_name} cache must be an independent file"
    [[ "$(sha256_of "$target")" == "$source_sha256" ]] \
      || fail "${view_name} pose copy hash mismatch for ${video_id}"
    printf '%s\t%s\t%s\n' "$video_id" "$digest" "$source_sha256" >> "$receipt"
    ((count += 1))
  done < <(jq -r '.records[].video_id' "$manifest")
  [[ "$count" -eq "$expected_count" ]] \
    || fail "${view_name} expected ${expected_count} caches, copied ${count}"
  [[ "$(find "$destination" -maxdepth 1 -type f -name '*.npz' | wc -l)" -eq "$expected_count" ]] \
    || fail "${view_name} has an unexpected NPZ count"
  [[ "$(find "$destination" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq "$expected_count" ]] \
    || fail "${view_name} contains a non-NPZ entry"
  [[ -z "$(find "$destination" -mindepth 1 -type l -print -quit)" ]] \
    || fail "${view_name} contains a symlink"
  chmod 0444 "$destination"/*.npz "$receipt"
  chmod 0555 "$destination"
}

write_gate_receipt() {
  local stage="$1"
  local gate_path="$2"
  local receipt_path="$3"
  local inspect_path="$4"
  test ! -e "$receipt_path" || fail "refusing to overwrite gate receipt"
  GATE_STAGE="$stage" \
  GATE_PATH="$gate_path" \
  GATE_RECEIPT="$receipt_path" \
  GATE_INSPECT="$inspect_path" \
  GATE_SCRIPT="$GATE_SCRIPT" \
  GATE_SOURCE_RECEIPT="$SOURCE_RECEIPT" \
  GATE_READOUT_CONFIG="${SOURCE_VIEW}/${READOUT_CONFIG_RELATIVE}" \
  GATE_EXPERIMENT_CONFIG="${SOURCE_VIEW}/${EXPERIMENT_CONFIG_RELATIVE}" \
  PAMS_EXPECTED_READOUT_CONFIG_SEMANTIC_SHA256="$READOUT_CONFIG_SEMANTIC_SHA256" \
  python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


gate = Path(os.environ["GATE_PATH"])
inspect = Path(os.environ["GATE_INSPECT"])
payload = {
    "schema_version": 1,
    "artifact_type": "pams_position_acf_predev_gate_receipt",
    "stage": os.environ["GATE_STAGE"],
    "gate_file": gate.name,
    "gate_sha256": digest(gate),
    "gate_bytes": gate.stat().st_size,
    "gate_script_sha256": digest(Path(os.environ["GATE_SCRIPT"])),
    "source_export_receipt_sha256": digest(
        Path(os.environ["GATE_SOURCE_RECEIPT"])
    ),
    "readout_config_sha256": digest(
        Path(os.environ["GATE_READOUT_CONFIG"])
    ),
    "readout_config_semantic_sha256": os.environ[
        "PAMS_EXPECTED_READOUT_CONFIG_SEMANTIC_SHA256"
    ],
    "experiment_config_sha256": digest(
        Path(os.environ["GATE_EXPERIMENT_CONFIG"])
    ),
    "container_pre_run_inspect_sha256": digest(inspect),
    "test105_evaluation_authorized": False,
}
Path(os.environ["GATE_RECEIPT"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

validate_gate_bundle() {
  local expected_stage="$1"
  local gate_path="$2"
  local receipt_path="$3"
  EXPECTED_STAGE="$expected_stage" \
  GATE_PATH="$gate_path" \
  RECEIPT_PATH="$receipt_path" \
  python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

gate_path = Path(os.environ["GATE_PATH"])
receipt_path = Path(os.environ["RECEIPT_PATH"])
gate_bytes = gate_path.read_bytes()
gate = json.loads(gate_bytes)
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
assert gate["schema_version"] == 1
assert gate["stage"] == os.environ["EXPECTED_STAGE"]
assert gate["status"] == "passed"
assert gate["eligible_for_paper_table"] is False
assert gate["test105_evaluation_authorized"] is False
assert gate["dev_inputs_mounted"] is False
assert gate["dev_targets_mounted"] is False
assert gate["test105_media_mounted"] is False
assert gate["test105_pose_mounted"] is False
assert gate["test105_labels_mounted"] is False
assert gate["gates"] and all(gate["gates"].values())
assert receipt["stage"] == os.environ["EXPECTED_STAGE"]
assert receipt["gate_sha256"] == hashlib.sha256(gate_bytes).hexdigest()
assert receipt["gate_bytes"] == len(gate_bytes)
assert receipt["test105_evaluation_authorized"] is False
PY
}

validate_prediction_bundle() {
  local predictions="$1"
  local receipt="$2"
  PREDICTIONS="$predictions" \
  RECEIPT="$receipt" \
  EXPECTED_READOUT_SHA256="$READOUT_CONFIG_SHA256" \
  python3 - <<'PY'
import hashlib
import json
import math
import os
from pathlib import Path

predictions_path = Path(os.environ["PREDICTIONS"])
receipt_path = Path(os.environ["RECEIPT"])
prediction_bytes = predictions_path.read_bytes()
payload = json.loads(prediction_bytes)
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
assert payload["protocol"] == "ucfrep_526"
assert payload["split"] == "dev"
assert payload["method_key"] == "pams-position-acf-direct-inferred-v1"
assert payload["eligible_for_paper_table"] is False
assert payload["test_evaluation_authorized"] is False
assert payload["record_total"] == 84
assert len(payload["records"]) == 84
assert len({row["video_id"] for row in payload["records"]}) == 84
assert payload["readout_config_file_sha256"] == os.environ[
    "EXPECTED_READOUT_SHA256"
]
assert payload["readout_config_fingerprint"] == (
    "b9c03953e66ef0146d607947c53d1e0d54466e3b938c9c314c6abf644c336773"
)
for row in payload["records"]:
    assert math.isfinite(float(row["period_frames"]))
    assert math.isfinite(float(row["confidence"]))
    spectrum = row["allowed_spectrum"]
    assert len(spectrum) == 63
    assert [item["bin_index"] for item in spectrum] == list(range(2, 65))
    if row["position_selected_bin"] is None:
        assert float(row["confidence"]) == 0.0
    else:
        selected = spectrum[row["position_selected_bin"] - 2]
        assert selected["period_frames"] == row["period_frames"]
        assert selected["power_share"] == row["confidence"]
    assert all(
        forbidden not in row
        for forbidden in (
            "target",
            "target_count",
            "ground_truth",
            "action",
            "label",
        )
    )
assert receipt["prediction_file"] == predictions_path.name
assert receipt["prediction_sha256"] == hashlib.sha256(prediction_bytes).hexdigest()
assert receipt["prediction_bytes"] == len(prediction_bytes)
assert receipt["readout_config_file_sha256"] == os.environ[
    "EXPECTED_READOUT_SHA256"
]
assert receipt["readout_config_fingerprint"] == payload[
    "readout_config_fingerprint"
]
assert receipt["test_evaluation_authorized"] is False
PY
}

validate_score_bundle() {
  local evaluation="$1"
  local receipt="$2"
  EVALUATION="$evaluation" \
  RECEIPT="$receipt" \
  EXPECTED_READOUT_SHA256="$READOUT_CONFIG_SHA256" \
  python3 - <<'PY'
import hashlib
import json
import math
import os
from pathlib import Path

evaluation_path = Path(os.environ["EVALUATION"])
receipt_path = Path(os.environ["RECEIPT"])
evaluation_bytes = evaluation_path.read_bytes()
evaluation = json.loads(evaluation_bytes)
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
assert evaluation["protocol"] == "ucfrep_526"
assert evaluation["split"] == "dev"
assert evaluation["method_key"] == "pams-position-acf-direct-inferred-v1"
assert evaluation["eligible_for_paper_table"] is False
assert evaluation["test_evaluation_authorized"] is False
assert evaluation["readout_config_file_sha256"] == os.environ[
    "EXPECTED_READOUT_SHA256"
]
report = evaluation["report"]
assert report["sample_count"] == 84
assert report["bootstrap_samples"] == 10_000
assert report["bootstrap_seed"] == 2026
assert len(report["per_video"]) == 84
assert math.isfinite(float(report["nmae"]))
assert math.isfinite(float(report["obo"]))
assert receipt["evaluation_file"] == evaluation_path.name
assert receipt["evaluation_sha256"] == hashlib.sha256(
    evaluation_bytes
).hexdigest()
assert receipt["evaluation_bytes"] == len(evaluation_bytes)
assert receipt["test_evaluation_authorized"] is False
assert (
    receipt["readout_config_file_sha256"]
    == evaluation["readout_config_file_sha256"]
)
PY
}

verify_container() {
  local container_name="$1"
  local stage="$2"
  local inspect_path="${AUDIT_ROOT}/${container_name}.pre-run.inspect.json"
  docker inspect "$container_name" > "$inspect_path"
  VERIFY_STAGE="$stage" \
  VERIFY_INSPECT="$inspect_path" \
  VERIFY_SOURCE_VIEW="$SOURCE_VIEW" \
  VERIFY_SOURCE_RECEIPT="$SOURCE_RECEIPT" \
  VERIFY_EXPERIMENT_CONFIG="${SOURCE_VIEW}/${EXPERIMENT_CONFIG_RELATIVE}" \
  VERIFY_READOUT_CONFIG="${SOURCE_VIEW}/${READOUT_CONFIG_RELATIVE}" \
  VERIFY_GATE_SCRIPT="$GATE_SCRIPT" \
  VERIFY_ENCODER_CHECKPOINT="$ENCODER_CHECKPOINT" \
  VERIFY_ENCODER_PROGRESS="$ENCODER_PROGRESS" \
  VERIFY_ENCODER_COMPLETION="$ENCODER_COMPLETION" \
  VERIFY_TRAIN_INPUT="$TRAIN_INPUT" \
  VERIFY_TRAIN_COMMIT="$TRAIN_COMMIT" \
  VERIFY_DEV_INPUT="$DEV_INPUT" \
  VERIFY_DEV_COMMIT="$DEV_COMMIT" \
  VERIFY_TEST_ID_INPUT="$TEST_ID_INPUT" \
  VERIFY_TEST_ID_COMMIT="$TEST_ID_COMMIT" \
  VERIFY_DEV_TARGET="$DEV_TARGET" \
  VERIFY_TRAIN_POSE="$TRAIN_POSE_VIEW" \
  VERIFY_DEV_POSE="$DEV_POSE_VIEW" \
  VERIFY_SYNTHETIC_STAGE="$SYNTHETIC_STAGE" \
  VERIFY_TRAIN_GATE_STAGE="$TRAIN_GATE_STAGE" \
  VERIFY_PREDICT_STAGE="$PREDICT_STAGE" \
  VERIFY_SCORE_STAGE="$SCORE_STAGE" \
  VERIFY_IMAGE_ID="$IMAGE_ID" \
  VERIFY_GPU_DEVICE="$GPU_DEVICE" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

stage = os.environ["VERIFY_STAGE"]
item = json.loads(
    Path(os.environ["VERIFY_INSPECT"]).read_text(encoding="utf-8")
)[0]
config = item["Config"]
host = item["HostConfig"]
mounts = {
    mount["Destination"]: (mount["Source"], bool(mount["RW"]))
    for mount in item["Mounts"]
}

assert config["Image"] == os.environ["VERIFY_IMAGE_ID"]
assert config["User"] == "1000:1000"
assert host["NetworkMode"] == "none"
assert host["ReadonlyRootfs"] is True
assert "ALL" in (host.get("CapDrop") or [])
assert "no-new-privileges:true" in (host.get("SecurityOpt") or [])

base = {
    "/workspace": (os.environ["VERIFY_SOURCE_VIEW"], False),
    "/pams/source-export-receipt.json": (
        os.environ["VERIFY_SOURCE_RECEIPT"],
        False,
    ),
}
configs = {
    "/pams/input/experiment-config.yaml": (
        os.environ["VERIFY_EXPERIMENT_CONFIG"],
        False,
    ),
    "/pams/input/readout-config.yaml": (
        os.environ["VERIFY_READOUT_CONFIG"],
        False,
    ),
}
encoder = {
    "/pams/encoder/encoder.pt": (
        os.environ["VERIFY_ENCODER_CHECKPOINT"],
        False,
    ),
    "/pams/encoder/encoder.jsonl": (
        os.environ["VERIFY_ENCODER_PROGRESS"],
        False,
    ),
    "/pams/encoder/encoder.completed.json": (
        os.environ["VERIFY_ENCODER_COMPLETION"],
        False,
    ),
}
if stage == "synthetic-k2-7":
    expected = {
        **base,
        **configs,
        **encoder,
        "/pams/gate.py": (os.environ["VERIFY_GATE_SCRIPT"], False),
        "/pams/output": (os.environ["VERIFY_SYNTHETIC_STAGE"], True),
    }
elif stage == "train337-distribution":
    expected = {
        **base,
        **configs,
        **encoder,
        "/pams/gate.py": (os.environ["VERIFY_GATE_SCRIPT"], False),
        "/pams/protocol/train.inputs.json": (
            os.environ["VERIFY_TRAIN_INPUT"],
            False,
        ),
        "/pams/protocol/train.inputs.commitment.json": (
            os.environ["VERIFY_TRAIN_COMMIT"],
            False,
        ),
        "/pams/pose-cache": (os.environ["VERIFY_TRAIN_POSE"], False),
        "/pams/output": (os.environ["VERIFY_TRAIN_GATE_STAGE"], True),
    }
elif stage == "dev-predict":
    expected = {
        **base,
        **configs,
        **encoder,
        "/pams/protocol/train.inputs.json": (
            os.environ["VERIFY_TRAIN_INPUT"],
            False,
        ),
        "/pams/protocol/train.inputs.commitment.json": (
            os.environ["VERIFY_TRAIN_COMMIT"],
            False,
        ),
        "/pams/protocol/dev.inputs.json": (
            os.environ["VERIFY_DEV_INPUT"],
            False,
        ),
        "/pams/protocol/dev.inputs.commitment.json": (
            os.environ["VERIFY_DEV_COMMIT"],
            False,
        ),
        "/pams/protocol/test-identity.inputs.json": (
            os.environ["VERIFY_TEST_ID_INPUT"],
            False,
        ),
        "/pams/protocol/test-identity.inputs.commitment.json": (
            os.environ["VERIFY_TEST_ID_COMMIT"],
            False,
        ),
        "/pams/pose-cache": (os.environ["VERIFY_DEV_POSE"], False),
        "/pams/output": (os.environ["VERIFY_PREDICT_STAGE"], True),
    }
elif stage == "dev-score":
    expected = {
        **base,
        "/pams/input/readout-config.yaml": (
            os.environ["VERIFY_READOUT_CONFIG"],
            False,
        ),
        "/pams/frozen/predictions.json": (
            f'{os.environ["VERIFY_PREDICT_STAGE"]}/predictions.json',
            False,
        ),
        "/pams/frozen/prediction.receipt.json": (
            f'{os.environ["VERIFY_PREDICT_STAGE"]}/prediction.receipt.json',
            False,
        ),
        "/pams/protocol/dev.targets.json": (
            os.environ["VERIFY_DEV_TARGET"],
            False,
        ),
        "/pams/output": (os.environ["VERIFY_SCORE_STAGE"], True),
    }
else:
    raise AssertionError(stage)
assert mounts == expected, (stage, mounts, expected)

all_text = "\0".join(
    list(mounts)
    + [source for source, _ in mounts.values()]
    + list(config.get("Cmd") or [])
)
if stage != "dev-score":
    assert "dev.targets" not in all_text
if stage in {"synthetic-k2-7", "train337-distribution"}:
    assert "dev.inputs" not in all_text
    assert "test-identity" not in all_text
if stage == "dev-predict":
    assert mounts["/pams/pose-cache"][0] == os.environ["VERIFY_DEV_POSE"]
    assert os.environ["VERIFY_TRAIN_POSE"] not in all_text
if stage == "dev-score":
    assert "/pams/pose-cache" not in mounts
    assert not any(path.startswith("/pams/encoder") for path in mounts)
    assert "test-identity" not in all_text
for forbidden in (
    "test.targets",
    "test.labels",
    "test105.targets",
    "test105.pose",
    "test105.media",
):
    assert forbidden not in all_text

device_requests = host.get("DeviceRequests") or []
if stage == "dev-score":
    assert not device_requests
else:
    assert len(device_requests) == 1
    assert device_requests[0].get("DeviceIDs") == [
        os.environ["VERIFY_GPU_DEVICE"]
    ]
PY
}

run_created_container() {
  local container_name="$1"
  local stage="$2"
  local stdout_path="${LOG_ROOT}/${container_name}.stdout.log"
  local stderr_path="${LOG_ROOT}/${container_name}.stderr.log"
  CURRENT_STAGE="$stage"
  write_status "running" "$stage" "null"
  ACTIVE_CONTAINER="$container_name"
  set +e
  docker start --attach "$container_name" > "$stdout_path" 2> "$stderr_path"
  local attach_exit="$?"
  set -e
  local still_running
  still_running="$(
    docker inspect "$container_name" --format '{{.State.Running}}'
  )"
  if [[ "$still_running" == "true" ]]; then
    docker stop --time 10 "$container_name" >/dev/null
  fi
  local container_exit
  container_exit="$(
    docker inspect "$container_name" --format '{{.State.ExitCode}}'
  )"
  docker inspect "$container_name" \
    > "${AUDIT_ROOT}/${container_name}.post-run.inspect.json"
  printf '%s\n' "$container_exit" \
    > "${AUDIT_ROOT}/${container_name}.exit-code.txt"
  ACTIVE_CONTAINER=""
  docker rm "$container_name" >/dev/null
  if [[ "$attach_exit" -ne 0 || "$still_running" == "true" ]]; then
    LAST_CONTAINER_EXIT=125
  else
    LAST_CONTAINER_EXIT="$container_exit"
  fi
}

[[ "$V8_RUN_ROOT" == "${ROOT}/runs/pams-v8-seed2026/"* ]] \
  || fail "PAMS_V8_RUN_ROOT must be under the frozen v8 run parent"
[[ ! -L "$V8_RUN_ROOT_INPUT" ]] || fail "v8 run root must not be a symlink"
test -d "$FIREWALL" || fail "missing frozen UCFRep firewall"
test -d "$POSE_POOL" || fail "missing exact-v8 pose cache pool"
[[ "$(find "$POSE_POOL" -maxdepth 1 -type f -name '*.npz' | wc -l)" -eq 421 ]] \
  || fail "exact-v8 pose pool does not contain 421 NPZ files"
[[ "$(find "$POSE_POOL" -mindepth 1 -maxdepth 1 -type f | wc -l)" -eq 421 ]] \
  || fail "exact-v8 pose pool contains unexpected files"
[[ -z "$(find "$POSE_POOL" -mindepth 1 -type l -print -quit)" ]] \
  || fail "exact-v8 pose pool contains a symlink"

assert_sha256 "$ENCODER_CHECKPOINT" "$ENCODER_SHA256"
assert_sha256 "$ENCODER_PROGRESS" "$ENCODER_PROGRESS_SHA256"
assert_sha256 "$ENCODER_COMPLETION" "$ENCODER_COMPLETION_SHA256"
assert_sha256 "$TRAIN_INPUT" "$TRAIN_INPUT_SHA256"
assert_sha256 "$TRAIN_COMMIT" "$TRAIN_COMMIT_SHA256"
assert_sha256 "$DEV_INPUT" "$DEV_INPUT_SHA256"
assert_sha256 "$DEV_COMMIT" "$DEV_COMMIT_SHA256"
assert_sha256 "$TEST_ID_INPUT" "$TEST_ID_INPUT_SHA256"
assert_sha256 "$TEST_ID_COMMIT" "$TEST_ID_COMMIT_SHA256"
# DEV_TARGET is intentionally not stat'ed, opened, or hashed before dev-score.

[[ "$(git -C "$SOURCE_CHECKOUT" rev-parse HEAD)" == "$SOURCE_REVISION" ]] \
  || fail "source checkout revision mismatch"
[[ -z "$(
  git -C "$SOURCE_CHECKOUT" status --porcelain=v1 --untracked-files=all
)" ]] || fail "source checkout is dirty"
readonly EXPERIMENT_CONFIG_SOURCE="${SOURCE_CHECKOUT}/${EXPERIMENT_CONFIG_RELATIVE}"
readonly READOUT_CONFIG_SOURCE="${SOURCE_CHECKOUT}/${READOUT_CONFIG_RELATIVE}"
assert_sha256 "$EXPERIMENT_CONFIG_SOURCE" "$EXPERIMENT_CONFIG_SHA256"
assert_sha256 "$READOUT_CONFIG_SOURCE" "$READOUT_CONFIG_SHA256"
source "${SOURCE_CHECKOUT}/scripts/server/environment_fingerprint.sh"
[[ "$(
  pams_environment_fingerprint "${SOURCE_CHECKOUT}/docker/server"
)" == "$ENVIRONMENT_SHA256" ]] || fail "source environment fingerprint mismatch"
[[ "$(docker image inspect "$IMAGE_ID" --format '{{.Id}}')" == "$IMAGE_ID" ]] \
  || fail "container image mismatch"
[[ "$(
  docker image inspect "$IMAGE_ID" \
    --format \
      '{{index .Config.Labels "org.opencontainers.image.pams.environment-sha256"}}'
)" == "$ENVIRONMENT_SHA256" ]] || fail "container environment label mismatch"

mkdir -p -- "$RUN_PARENT"
[[ ! -L "$RUN_PARENT" ]] || fail "run parent must not be a symlink"
readonly RUN_PARENT_RESOLVED="$(realpath -e -- "$RUN_PARENT")"
[[ "$RUN_PARENT_RESOLVED" == "${ROOT}/runs/pams-position-acf-seed2026" ]] \
  || fail "run parent resolves outside PAMS_ROOT"
mkdir -- "$RUN_ROOT" || fail "immutable run root exists: $RUN_ROOT"
RUN_RESERVED=1
mkdir -- \
  "$SOURCE_VIEW" \
  "$INPUT_ROOT" \
  "${RUN_ROOT}/stages" \
  "$SYNTHETIC_STAGE" \
  "$TRAIN_GATE_STAGE" \
  "$PREDICT_STAGE" \
  "$SCORE_STAGE" \
  "$AUDIT_ROOT" \
  "$LOG_ROOT"
write_status "preparing" "source-export" "null"

git -C "$SOURCE_CHECKOUT" archive \
  --format=tar \
  "$SOURCE_REVISION" \
  src \
  pyproject.toml \
  "$EXPERIMENT_CONFIG_RELATIVE" \
  "$READOUT_CONFIG_RELATIVE" \
  | tar -xf - -C "$SOURCE_VIEW"
test ! -e "${SOURCE_VIEW}/.git" || fail "source export unexpectedly contains .git"
test ! -e "${SOURCE_VIEW}/results" || fail "source export unexpectedly contains results"
test ! -e "${SOURCE_VIEW}/data" || fail "source export unexpectedly contains data"
test ! -e "${SOURCE_VIEW}/tests" || fail "source export unexpectedly contains tests"
[[ -z "$(find "$SOURCE_VIEW" -type l -print -quit)" ]] \
  || fail "source export contains a symlink"
assert_sha256 \
  "${SOURCE_VIEW}/${EXPERIMENT_CONFIG_RELATIVE}" \
  "$EXPERIMENT_CONFIG_SHA256"
assert_sha256 \
  "${SOURCE_VIEW}/${READOUT_CONFIG_RELATIVE}" \
  "$READOUT_CONFIG_SHA256"

SOURCE_EXPORT_ROOT="$SOURCE_VIEW" \
SOURCE_EXPORT_REVISION="$SOURCE_REVISION" \
SOURCE_EXPORT_RECEIPT="$SOURCE_RECEIPT" \
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["SOURCE_EXPORT_ROOT"]).resolve(strict=True)
output = Path(os.environ["SOURCE_EXPORT_RECEIPT"])
records = []
for path in sorted(
    root.rglob("*"),
    key=lambda item: item.relative_to(root).as_posix(),
):
    if path.is_symlink():
        raise RuntimeError(f"source export contains symlink: {path}")
    if path.is_dir():
        continue
    if not path.is_file():
        raise RuntimeError(f"source export contains non-file: {path}")
    data = path.read_bytes()
    records.append(
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
    )
if not records:
    raise RuntimeError("source export is empty")
payload = {
    "schema_version": 1,
    "source_revision": os.environ["SOURCE_EXPORT_REVISION"],
    "root": ".",
    "files": records,
}
output.write_text(
    json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY
SOURCE_RECEIPT_SHA256="$(sha256_of "$SOURCE_RECEIPT")"
readonly SOURCE_RECEIPT_SHA256
chmod -R a-w "$SOURCE_VIEW"
chmod 0444 "$SOURCE_RECEIPT"

CURRENT_STAGE="pose-views"
copy_pose_view "pose-train337" "$TRAIN_INPUT" "$TRAIN_POSE_VIEW" 337
copy_pose_view "pose-dev84" "$DEV_INPUT" "$DEV_POSE_VIEW" 84

SPLIT_TRAIN="$TRAIN_INPUT" \
SPLIT_DEV="$DEV_INPUT" \
SPLIT_TEST="$TEST_ID_INPUT" \
SPLIT_TEST_COMMITMENT="$TEST_ID_COMMIT" \
python3 - <<'PY'
import json
import os
from pathlib import Path


def load(name):
    return json.loads(Path(os.environ[name]).read_text(encoding="utf-8"))


train = load("SPLIT_TRAIN")
dev = load("SPLIT_DEV")
test = load("SPLIT_TEST")
test_commitment = load("SPLIT_TEST_COMMITMENT")
assert len(train["records"]) == 337
assert len(dev["records"]) == 84
assert len(test["records"]) == 105
sets = [
    {row["video_id"] for row in payload["records"]}
    for payload in (train, dev, test)
]
assert all(
    len(ids) == expected
    for ids, expected in zip(sets, (337, 84, 105), strict=True)
)
assert not (sets[0] & sets[1])
assert not (sets[0] & sets[2])
assert not (sets[1] & sets[2])
for row in test["records"]:
    assert set(row) == {"video_id", "video_path", "video_sha256"}
for payload in (test, test_commitment):
    text = json.dumps(payload, sort_keys=True).lower()
    for forbidden in ('"action"', '"count"', '"target"', '"label"'):
        assert forbidden not in text
PY

cat > "$GATE_SCRIPT" <<'PY'
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from pams.config import load_config
from pams.data import load_pose_cache_set, load_pose_input_manifest
from pams.metrics import round_count
from pams.period import (
    estimate_period_from_projected_position,
    projected_position_vector_acf_diagnostics,
)
from pams.training import (
    CheckpointProvenance,
    collate_pose_sequences,
    load_model_checkpoint,
)
from pams.types import PoseSequence
from pams.reproducibility import sha256_json


def digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_policy(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("readout config must be a mapping")
    if set(payload) != {
        "schema_version",
        "key",
        "classification",
        "eligible_for_paper_table",
        "test105_evaluation_authorized",
        "frozen_inputs",
        "readout",
        "predev_gates",
        "dev_stop_gate",
        "mount_policy",
    }:
        raise ValueError("readout config has unexpected top-level fields")
    if payload["schema_version"] != 1:
        raise ValueError("readout config schema mismatch")
    if payload["key"] != "pams-projected-position-acf-direct-v1":
        raise ValueError("readout key mismatch")
    if payload["eligible_for_paper_table"] is not False:
        raise ValueError("position-ACF may not be paper-table eligible")
    if payload["test105_evaluation_authorized"] is not False:
        raise ValueError("position-ACF may not authorize test105")
    if sha256_json(payload) != os.environ[
        "PAMS_EXPECTED_READOUT_CONFIG_SEMANTIC_SHA256"
    ]:
        raise ValueError("readout config semantic fingerprint mismatch")
    readout = payload["readout"]
    if readout["temporal_transform"] != "none":
        raise ValueError("projected-position ACF must not use velocity")
    if readout["selection"] != "maximum_allowed_non_dc_power_share":
        raise ValueError("unexpected projected-position ACF selector")
    return payload


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_encoder(
    *,
    config_path: Path,
    readout_path: Path,
    checkpoint_path: Path,
    device: torch.device,
):
    policy = load_policy(readout_path)
    frozen = policy["frozen_inputs"]
    checks = {
        "experiment config": (
            digest(config_path),
            frozen["experiment_config_sha256"],
        ),
        "encoder checkpoint": (
            digest(checkpoint_path),
            frozen["encoder_checkpoint_sha256"],
        ),
        "encoder progress": (
            digest("/pams/encoder/encoder.jsonl"),
            frozen["encoder_progress_sha256"],
        ),
        "encoder completion": (
            digest("/pams/encoder/encoder.completed.json"),
            frozen["encoder_completion_receipt_sha256"],
        ),
    }
    mismatches = [
        f"{name}: observed={actual}, expected={expected}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError("; ".join(mismatches))
    config = load_config(config_path)
    if config.fingerprint != frozen["experiment_config_fingerprint"]:
        raise ValueError("experiment config fingerprint mismatch")
    if config.pose_fingerprint != frozen["pose_fingerprint"]:
        raise ValueError("pose fingerprint mismatch")
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    provenance = CheckpointProvenance.from_mapping(raw["provenance"])
    model = load_model_checkpoint(
        checkpoint_path,
        config,
        device=device,
        expected_stage="encoder",
        expected_provenance=provenance,
    ).eval()
    return policy, config, model


def infer(
    sequences,
    *,
    model,
    minimum: int,
    maximum: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(sequences), 32):
            batch = collate_pose_sequences(sequences[start : start + 32]).to(device)
            _, projected = model.encoder.forward_with_pre_pe(
                batch.poses,
                batch.valid_mask,
            )
            periods, confidences = estimate_period_from_projected_position(
                projected,
                minimum=minimum,
                maximum=maximum,
                valid_mask=batch.valid_mask,
            )
            diagnostics = projected_position_vector_acf_diagnostics(
                projected,
                minimum=minimum,
                maximum=maximum,
                valid_mask=batch.valid_mask,
            )
            if len(diagnostics) != len(batch.video_ids):
                raise RuntimeError("position-ACF diagnostic batch length mismatch")
            valid_frames = batch.valid_mask.sum(dim=1)
            for index, (video_id, diagnostic) in enumerate(
                zip(batch.video_ids, diagnostics, strict=True)
            ):
                period = float(periods[index])
                confidence = float(confidences[index])
                valid = int(valid_frames[index])
                if period != float(diagnostic.selected_period):
                    raise RuntimeError("estimator and diagnostic period disagree")
                if confidence != float(diagnostic.confidence):
                    raise RuntimeError("estimator and diagnostic confidence disagree")
                raw_count = (
                    float(valid - 1) / period
                    if confidence > 0.0 and valid >= 2
                    else 0.0
                )
                allowed_bins = tuple(int(value) for value in diagnostic.allowed_bins)
                frequencies = tuple(float(value) for value in diagnostic.frequencies)
                candidate_periods = tuple(float(value) for value in diagnostic.periods)
                power_shares = tuple(float(value) for value in diagnostic.power_shares)
                if not (
                    len(allowed_bins)
                    == len(frequencies)
                    == len(candidate_periods)
                    == len(power_shares)
                ):
                    raise RuntimeError("position-ACF diagnostic vectors disagree")
                rows.append(
                    {
                        "video_id": video_id,
                        "valid_frames": valid,
                        "period_frames": period,
                        "confidence": confidence,
                        "raw_count": raw_count,
                        "rounded_count": round_count(raw_count),
                        "allowed_bins": list(allowed_bins),
                        "frequencies": list(frequencies),
                        "periods": list(candidate_periods),
                        "power_shares": list(power_shares),
                        "selected_bin": (
                            None
                            if diagnostic.selected_bin is None
                            else int(diagnostic.selected_bin)
                        ),
                        "selected_period": float(diagnostic.selected_period),
                    }
                )
    return rows


def synthetic_gate(policy, config, model, device):
    gate = policy["predev_gates"]["synthetic"]
    harmonic_orders = tuple(int(value) for value in gate["harmonic_orders"])
    if harmonic_orders != tuple(range(2, 8)):
        raise ValueError("synthetic harmonic-order gate must be exactly k=2..7")
    frames = int(gate["frames"])
    fundamental_period = float(gate["fundamental_period_frames"])
    fundamental_amplitude = float(gate["fundamental_amplitude"])
    harmonic_amplitude = float(gate["harmonic_amplitude"])
    if (
        frames != 256
        or fundamental_period != 64.0
        or fundamental_amplitude != 1.0
        or harmonic_amplitude != 0.75
    ):
        raise ValueError("synthetic harmonic-order gate constants drifted")
    time = np.arange(frames, dtype=np.float64)
    phase = 2.0 * np.pi * time / fundamental_period
    sequences = []
    for order in harmonic_orders:
        waveform = (
            fundamental_amplitude * np.sin(phase)
            + harmonic_amplitude * np.sin(float(order) * phase)
        )
        xyz = np.zeros((frames, 33, 3), dtype=np.float32)
        # Both components share one pose-space direction, so the frozen linear
        # projection cannot change their amplitude ratio.
        xyz[:, 0, 0] = waveform.astype(np.float32)
        sequences.append(
            PoseSequence(
                video_id=f"harmonic-order-{order}",
                fps=30.0,
                xyz=xyz,
                valid_mask=np.ones(frames, dtype=np.bool_),
            )
        )
    rows = infer(
        tuple(sequences),
        model=model,
        minimum=config.period.minimum,
        maximum=config.period.maximum,
        device=device,
    )
    for row, order in zip(rows, harmonic_orders, strict=True):
        row["harmonic_order"] = order
        row["expected_fundamental_period_frames"] = fundamental_period
    finite = all(
        math.isfinite(float(row[field]))
        for row in rows
        for field in ("period_frames", "confidence", "raw_count")
    )
    positive = all(float(row["confidence"]) > 0.0 for row in rows)
    fundamental_selection_fraction = sum(
        abs(float(row["period_frames"]) - fundamental_period) <= 1e-6
        for row in rows
    ) / len(rows)
    gates = {
        "record_total_exact": len(rows) == int(gate["required_records"]),
        "all_outputs_finite": finite if gate["require_finite_outputs"] else True,
        "all_confidences_positive": (
            positive if gate["require_positive_confidence"] else True
        ),
        "fundamental_selection_fraction_gte_frozen_minimum": (
            fundamental_selection_fraction
            >= float(gate["minimum_fundamental_selection_fraction"])
        ),
    }
    return {
        "summary": {
            "records": len(rows),
            "harmonic_orders": list(harmonic_orders),
            "fundamental_period_frames": fundamental_period,
            "fundamental_amplitude": fundamental_amplitude,
            "harmonic_amplitude": harmonic_amplitude,
            "fundamental_selection_fraction": fundamental_selection_fraction,
        },
        "rows": rows,
        "gates": gates,
    }


def train_gate(policy, config, model, device):
    gate = policy["predev_gates"]["train337_target_free_distribution"]
    manifest = load_pose_input_manifest("/pams/protocol/train.inputs.json")
    if manifest.split != "train" or len(manifest.records) != 337:
        raise ValueError("train337 identity sidecar mismatch")
    sequences, snapshot = load_pose_cache_set(
        manifest.records,
        cache_dir="/pams/pose-cache",
        pose_fingerprint=config.pose_fingerprint,
    )
    expected_pose_set = policy["frozen_inputs"]["train337_pose_cache_set_sha256"]
    if snapshot.fingerprint != expected_pose_set:
        raise ValueError("train337 pose-cache-set fingerprint mismatch")
    rows = infer(
        sequences,
        model=model,
        minimum=config.period.minimum,
        maximum=config.period.maximum,
        device=device,
    )
    finite = all(
        math.isfinite(float(row[field]))
        for row in rows
        for field in ("period_frames", "confidence", "raw_count")
    )
    positive = [
        row
        for row in rows
        if row["confidence"] > 0.0 and row["valid_frames"] >= 2
    ]
    denominator = max(1, len(positive))
    zero_fraction = (len(rows) - len(positive)) / len(rows)
    minimum_fraction = sum(
        row["period_frames"] <= config.period.minimum + 1e-6
        for row in positive
    ) / denominator
    maximum_fraction = sum(
        row["period_frames"] >= config.period.maximum - 1e-6
        for row in positive
    ) / denominator
    bins = [
        int(row["selected_bin"])
        for row in positive
        if row["selected_bin"] is not None
    ]
    positive_periods = [float(row["period_frames"]) for row in positive]
    bin_counts = Counter(bins)
    mode_fraction = max(bin_counts.values(), default=0) / denominator
    zero_guard = all(
        row["rounded_count"] == 0
        for row in rows
        if row["confidence"] <= 0.0 or row["valid_frames"] < 2
    )
    selected_bin_guard = all(
        row["selected_bin"] is not None for row in positive
    )
    gates = {
        "record_total_exact": len(rows) == int(gate["required_records"]),
        "all_outputs_finite": finite if gate["require_finite_outputs"] else True,
        "zero_confidence_fraction_lte_frozen_maximum": (
            zero_fraction <= float(gate["maximum_zero_confidence_fraction"])
        ),
        "minimum_period_fraction_lte_frozen_maximum": (
            minimum_fraction <= float(gate["maximum_minimum_period_fraction"])
        ),
        "maximum_period_fraction_lte_frozen_maximum": (
            maximum_fraction <= float(gate["maximum_maximum_period_fraction"])
        ),
        "selected_bin_mode_fraction_lte_frozen_maximum": (
            mode_fraction <= float(gate["maximum_selected_bin_mode_fraction"])
        ),
        "unique_selected_bins_gte_frozen_minimum": (
            len(bin_counts) >= int(gate["minimum_unique_selected_bins"])
        ),
        "positive_confidence_selected_bin_present": selected_bin_guard,
        "zero_confidence_zero_count": (
            zero_guard if gate["require_zero_confidence_zero_count"] else True
        ),
    }
    return {
        "summary": {
            "records": len(rows),
            "positive_confidence_records": len(positive),
            "zero_confidence_records": len(rows) - len(positive),
            "zero_confidence_fraction": zero_fraction,
            "minimum_period_fraction": minimum_fraction,
            "maximum_period_fraction": maximum_fraction,
            "selected_bin_mode_fraction": mode_fraction,
            "unique_selected_bins": len(bin_counts),
            "period_min": min(positive_periods) if positive_periods else None,
            "period_median": (
                float(np.median(positive_periods)) if positive_periods else None
            ),
            "period_max": max(positive_periods) if positive_periods else None,
            "confidence_mean": float(
                np.mean([float(row["confidence"]) for row in rows])
            ),
            "pose_cache_set_sha256": snapshot.fingerprint,
        },
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("synthetic", "train337"), required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--readout-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage = "synthetic-k2-7" if args.mode == "synthetic" else "train337-distribution"
    try:
        device = torch.device("cuda:0")
        policy, config, model = load_encoder(
            config_path=args.experiment_config,
            readout_path=args.readout_config,
            checkpoint_path=args.checkpoint,
            device=device,
        )
        result = (
            synthetic_gate(policy, config, model, device)
            if args.mode == "synthetic"
            else train_gate(policy, config, model, device)
        )
        gates = result.pop("gates")
        payload = {
            "schema_version": 1,
            "artifact_type": "pams_position_acf_predev_gate",
            "stage": stage,
            "status": "passed" if all(gates.values()) else "failed",
            "classification": policy["classification"],
            "method_key": policy["key"],
            "eligible_for_paper_table": False,
            "test105_evaluation_authorized": False,
            "source_revision": os.environ["PAMS_CONTAINER_SOURCE_REVISION"],
            "readout_config_sha256": digest(args.readout_config),
            "experiment_config_sha256": digest(args.experiment_config),
            "encoder_checkpoint_sha256": digest(args.checkpoint),
            "algorithm": (
                "encoder_input_projection_pre_pe -> projected-position "
                "masked vector ACF -> Hann-windowed allowed-band spectrum -> "
                "maximum power share -> half_up((valid_frames-1)/period)"
            ),
            "dev_inputs_mounted": False,
            "dev_targets_mounted": False,
            "test105_media_mounted": False,
            "test105_pose_mounted": False,
            "test105_labels_mounted": False,
            "gates": gates,
            **result,
        }
        write_exclusive(args.output, payload)
        return 0 if payload["status"] == "passed" else 3
    except Exception as exc:
        write_exclusive(
            args.output,
            {
                "schema_version": 1,
                "artifact_type": "pams_position_acf_predev_gate",
                "stage": stage,
                "status": "error",
                "eligible_for_paper_table": False,
                "test105_evaluation_authorized": False,
                "dev_inputs_mounted": False,
                "dev_targets_mounted": False,
                "test105_media_mounted": False,
                "test105_pose_mounted": False,
                "test105_labels_mounted": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "gates": {},
            },
        )
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
PY
chmod 0444 "$GATE_SCRIPT"

cat > "${RUN_ROOT}/attempt.reservation.json" <<EOF
{
  "schema_version": 1,
  "attempt_id": "${ATTEMPT_ID}",
  "reserved_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "dataset": "UCFRep",
  "split": "dev84",
  "seed": 2026,
  "classification": "inferred frozen-v8 projected-position vector-ACF diagnostic",
  "eligible_for_paper_table": false,
  "test105_evaluation_authorized": false,
  "source_revision": "${SOURCE_REVISION}",
  "source_export_receipt_sha256": "${SOURCE_RECEIPT_SHA256}",
  "container_image_id": "${IMAGE_ID}",
  "container_environment_sha256": "${ENVIRONMENT_SHA256}",
  "experiment_config_sha256": "${EXPERIMENT_CONFIG_SHA256}",
  "readout_config_sha256": "${READOUT_CONFIG_SHA256}",
  "readout_config_semantic_sha256": "${READOUT_CONFIG_SEMANTIC_SHA256}",
  "encoder_checkpoint_sha256": "${ENCODER_SHA256}",
  "train337_pose_cache_set_sha256": "${TRAIN_POSE_SET_SHA256}",
  "dev84_pose_cache_set_sha256": "${DEV_POSE_SET_SHA256}",
  "stage_order": [
    "synthetic-k2-7",
    "train337-target-free-distribution",
    "dev84-target-free-predict",
    "dev84-target-bearing-score",
    "dev84-stop-decision"
  ],
  "dev_predict_pose_scope": "dev84-only",
  "dev_predict_targets_mounted": false,
  "dev_score_pose_mounted": false,
  "dev_score_checkpoint_mounted": false,
  "test105_identity_sidecars_mounted_for_prediction": true,
  "test105_media_mounted": false,
  "test105_pose_mounted": false,
  "test105_labels_mounted": false,
  "dev_stop_reference": {
    "key": "pams-local-frequency-synthetic-v1",
    "evaluation_sha256": "${LOCAL_FREQUENCY_EVALUATION_SHA256}",
    "nmae": ${LOCAL_FREQUENCY_NMAE},
    "obo": ${LOCAL_FREQUENCY_OBO},
    "requires_strict_improvement_on_both": true
  }
}
EOF
chmod 0444 "${RUN_ROOT}/attempt.reservation.json"

{
  printf 'source_revision %s\n' "$SOURCE_REVISION"
  printf 'source_export_receipt_sha256 %s\n' "$SOURCE_RECEIPT_SHA256"
  printf 'image_id %s\n' "$IMAGE_ID"
  printf 'environment_sha256 %s\n' "$ENVIRONMENT_SHA256"
  printf 'experiment_config_sha256 %s\n' "$EXPERIMENT_CONFIG_SHA256"
  printf 'readout_config_sha256 %s\n' "$READOUT_CONFIG_SHA256"
  printf 'readout_config_semantic_sha256 %s\n' "$READOUT_CONFIG_SEMANTIC_SHA256"
  printf 'encoder_checkpoint_sha256 %s\n' "$ENCODER_SHA256"
  printf 'encoder_progress_sha256 %s\n' "$ENCODER_PROGRESS_SHA256"
  printf 'encoder_completion_sha256 %s\n' "$ENCODER_COMPLETION_SHA256"
  printf 'train_inputs_sha256 %s\n' "$TRAIN_INPUT_SHA256"
  printf 'train_commitment_sha256 %s\n' "$TRAIN_COMMIT_SHA256"
  printf 'dev_inputs_sha256 %s\n' "$DEV_INPUT_SHA256"
  printf 'dev_commitment_sha256 %s\n' "$DEV_COMMIT_SHA256"
  printf 'test_identity_inputs_sha256 %s\n' "$TEST_ID_INPUT_SHA256"
  printf 'test_identity_commitment_sha256 %s\n' "$TEST_ID_COMMIT_SHA256"
  printf 'train337_pose_cache_set_sha256 %s\n' "$TRAIN_POSE_SET_SHA256"
  printf 'dev84_pose_cache_set_sha256 %s\n' "$DEV_POSE_SET_SHA256"
  printf 'dev_targets_expected_sha256 %s\n' "$DEV_TARGET_SHA256"
} > "${AUDIT_ROOT}/frozen-bindings.txt"
chmod 0444 "${AUDIT_ROOT}/frozen-bindings.txt"

CURRENT_STAGE="hardware-audit"
nvidia-smi -q > "${AUDIT_ROOT}/nvidia-smi-q.txt"
lscpu > "${AUDIT_ROOT}/lscpu.txt"
docker version > "${AUDIT_ROOT}/docker-version.txt"
git --version > "${AUDIT_ROOT}/git-version.txt"
docker image inspect "$IMAGE_ID" > "${AUDIT_ROOT}/image.inspect.json"

for container_name in \
  "$SYNTHETIC_NAME" \
  "$TRAIN_GATE_NAME" \
  "$PREDICT_NAME" \
  "$SCORE_NAME"; do
  if docker inspect "$container_name" >/dev/null 2>&1; then
    fail "container name already exists: $container_name"
  fi
done

mkdir -p -- "$LOCK_ROOT"
[[ ! -L "$LOCK_ROOT" ]] || fail "GPU lock root must not be a symlink"
readonly LOCK_ROOT_RESOLVED="$(realpath -e -- "$LOCK_ROOT")"
[[ "$LOCK_ROOT_RESOLVED" == "${ROOT}/.pams-gpu-locks" ]] \
  || fail "GPU lock root resolves outside PAMS_ROOT"
[[ ! -L "$LOCK_PATH" ]] || fail "GPU lock file must not be a symlink"
exec 9>"$LOCK_PATH"
chmod 0600 "$LOCK_PATH"
if ! flock -n 9; then
  CURRENT_STAGE="gpu-lock"
  exit 75
fi

common_args=(
  --init
  --network none
  --read-only
  --shm-size 8g
  --pids-limit 4096
  --cpus 24
  --memory 96g
  --cap-drop ALL
  --security-opt no-new-privileges:true
  --user 1000:1000
  --workdir /workspace
  --env PYTHONPATH=/workspace/src
  --env PYTHONDONTWRITEBYTECODE=1
  --env HOME=/pams/home
  --env PAMS_AUDIT_MODE=formal
  --env "PAMS_CONTAINER_IMAGE_ID=${IMAGE_ID}"
  --env "PAMS_CONTAINER_ENVIRONMENT_SHA256=${ENVIRONMENT_SHA256}"
  --env "PAMS_CONTAINER_SOURCE_REVISION=${SOURCE_REVISION}"
  --env "PAMS_EXPECTED_READOUT_CONFIG_SEMANTIC_SHA256=${READOUT_CONFIG_SEMANTIC_SHA256}"
  --env PAMS_SOURCE_EXPORT_RECEIPT=/pams/source-export-receipt.json
  --env "PAMS_SOURCE_EXPORT_RECEIPT_SHA256=${SOURCE_RECEIPT_SHA256}"
  --tmpfs /tmp:rw,noexec,nosuid,nodev,uid=1000,gid=1000,size=2g
  --tmpfs /pams/tmp:rw,noexec,nosuid,nodev,uid=1000,gid=1000,size=8g
  --tmpfs /pams/cache:rw,noexec,nosuid,nodev,uid=1000,gid=1000,size=4g
  --tmpfs /pams/home:rw,noexec,nosuid,nodev,uid=1000,gid=1000,size=1g
)
source_args=(
  --mount "type=bind,src=${SOURCE_VIEW},dst=/workspace,readonly"
  --mount \
    "type=bind,src=${SOURCE_RECEIPT},dst=/pams/source-export-receipt.json,readonly"
)
config_args=(
  --mount \
    "type=bind,src=${SOURCE_VIEW}/${EXPERIMENT_CONFIG_RELATIVE},dst=/pams/input/experiment-config.yaml,readonly"
  --mount \
    "type=bind,src=${SOURCE_VIEW}/${READOUT_CONFIG_RELATIVE},dst=/pams/input/readout-config.yaml,readonly"
)
encoder_args=(
  --mount \
    "type=bind,src=${ENCODER_CHECKPOINT},dst=/pams/encoder/encoder.pt,readonly"
  --mount \
    "type=bind,src=${ENCODER_PROGRESS},dst=/pams/encoder/encoder.jsonl,readonly"
  --mount \
    "type=bind,src=${ENCODER_COMPLETION},dst=/pams/encoder/encoder.completed.json,readonly"
)
gpu_args=(
  --gpus "device=${GPU_DEVICE}"
  --env CUDA_VISIBLE_DEVICES=0
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8
)

CURRENT_STAGE="synthetic-k2-7-create"
docker create \
  --name "$SYNTHETIC_NAME" \
  "${common_args[@]}" \
  "${gpu_args[@]}" \
  "${source_args[@]}" \
  "${config_args[@]}" \
  "${encoder_args[@]}" \
  --mount "type=bind,src=${GATE_SCRIPT},dst=/pams/gate.py,readonly" \
  --mount "type=bind,src=${SYNTHETIC_STAGE},dst=/pams/output" \
  "$IMAGE_ID" \
  python /pams/gate.py \
    --mode synthetic \
    --experiment-config /pams/input/experiment-config.yaml \
    --readout-config /pams/input/readout-config.yaml \
    --checkpoint /pams/encoder/encoder.pt \
    --output /pams/output/gate.json \
  > "${AUDIT_ROOT}/${SYNTHETIC_NAME}.create-id.txt"
verify_container "$SYNTHETIC_NAME" "synthetic-k2-7"
run_created_container "$SYNTHETIC_NAME" "synthetic-k2-7"
test -s "${SYNTHETIC_STAGE}/gate.json" \
  || fail "synthetic gate did not retain a result"
write_gate_receipt \
  "synthetic-k2-7" \
  "${SYNTHETIC_STAGE}/gate.json" \
  "${SYNTHETIC_STAGE}/gate.receipt.json" \
  "${AUDIT_ROOT}/${SYNTHETIC_NAME}.pre-run.inspect.json"
chmod 0444 "${SYNTHETIC_STAGE}/gate.json" "${SYNTHETIC_STAGE}/gate.receipt.json"
[[ "$LAST_CONTAINER_EXIT" -eq 0 ]] || fail "synthetic k=2..7 gate failed"
validate_gate_bundle \
  "synthetic-k2-7" \
  "${SYNTHETIC_STAGE}/gate.json" \
  "${SYNTHETIC_STAGE}/gate.receipt.json"
chmod 0555 "$SYNTHETIC_STAGE"

CURRENT_STAGE="train337-distribution-create"
docker create \
  --name "$TRAIN_GATE_NAME" \
  "${common_args[@]}" \
  "${gpu_args[@]}" \
  "${source_args[@]}" \
  "${config_args[@]}" \
  "${encoder_args[@]}" \
  --mount "type=bind,src=${GATE_SCRIPT},dst=/pams/gate.py,readonly" \
  --mount \
    "type=bind,src=${TRAIN_INPUT},dst=/pams/protocol/train.inputs.json,readonly" \
  --mount \
    "type=bind,src=${TRAIN_COMMIT},dst=/pams/protocol/train.inputs.commitment.json,readonly" \
  --mount "type=bind,src=${TRAIN_POSE_VIEW},dst=/pams/pose-cache,readonly" \
  --mount "type=bind,src=${TRAIN_GATE_STAGE},dst=/pams/output" \
  "$IMAGE_ID" \
  python /pams/gate.py \
    --mode train337 \
    --experiment-config /pams/input/experiment-config.yaml \
    --readout-config /pams/input/readout-config.yaml \
    --checkpoint /pams/encoder/encoder.pt \
    --output /pams/output/gate.json \
  > "${AUDIT_ROOT}/${TRAIN_GATE_NAME}.create-id.txt"
verify_container "$TRAIN_GATE_NAME" "train337-distribution"
run_created_container "$TRAIN_GATE_NAME" "train337-distribution"
test -s "${TRAIN_GATE_STAGE}/gate.json" \
  || fail "train337 distribution gate did not retain a result"
write_gate_receipt \
  "train337-distribution" \
  "${TRAIN_GATE_STAGE}/gate.json" \
  "${TRAIN_GATE_STAGE}/gate.receipt.json" \
  "${AUDIT_ROOT}/${TRAIN_GATE_NAME}.pre-run.inspect.json"
chmod 0444 \
  "${TRAIN_GATE_STAGE}/gate.json" \
  "${TRAIN_GATE_STAGE}/gate.receipt.json"
[[ "$LAST_CONTAINER_EXIT" -eq 0 ]] || fail "train337 target-free distribution gate failed"
validate_gate_bundle \
  "train337-distribution" \
  "${TRAIN_GATE_STAGE}/gate.json" \
  "${TRAIN_GATE_STAGE}/gate.receipt.json"
chmod 0555 "$TRAIN_GATE_STAGE"

CURRENT_STAGE="dev-predict-create"
docker create \
  --name "$PREDICT_NAME" \
  "${common_args[@]}" \
  "${gpu_args[@]}" \
  "${source_args[@]}" \
  "${config_args[@]}" \
  "${encoder_args[@]}" \
  --mount \
    "type=bind,src=${TRAIN_INPUT},dst=/pams/protocol/train.inputs.json,readonly" \
  --mount \
    "type=bind,src=${TRAIN_COMMIT},dst=/pams/protocol/train.inputs.commitment.json,readonly" \
  --mount \
    "type=bind,src=${DEV_INPUT},dst=/pams/protocol/dev.inputs.json,readonly" \
  --mount \
    "type=bind,src=${DEV_COMMIT},dst=/pams/protocol/dev.inputs.commitment.json,readonly" \
  --mount \
    "type=bind,src=${TEST_ID_INPUT},dst=/pams/protocol/test-identity.inputs.json,readonly" \
  --mount \
    "type=bind,src=${TEST_ID_COMMIT},dst=/pams/protocol/test-identity.inputs.commitment.json,readonly" \
  --mount "type=bind,src=${DEV_POSE_VIEW},dst=/pams/pose-cache,readonly" \
  --mount "type=bind,src=${PREDICT_STAGE},dst=/pams/output" \
  "$IMAGE_ID" \
  python -m pams position-acf dev-predict \
    /pams/encoder/encoder.pt \
    /pams/protocol/train.inputs.json \
    /pams/pose-cache \
    /pams/output \
    --checkpoint-progress /pams/encoder/encoder.jsonl \
    --checkpoint-completion-receipt /pams/encoder/encoder.completed.json \
    --input-commitment /pams/protocol/train.inputs.commitment.json \
    --dev-inputs /pams/protocol/dev.inputs.json \
    --dev-input-commitment /pams/protocol/dev.inputs.commitment.json \
    --test-identity-inputs /pams/protocol/test-identity.inputs.json \
    --test-identity-commitment /pams/protocol/test-identity.inputs.commitment.json \
    --config /pams/input/experiment-config.yaml \
    --readout-config /pams/input/readout-config.yaml \
    --device cuda:0 \
  > "${AUDIT_ROOT}/${PREDICT_NAME}.create-id.txt"
verify_container "$PREDICT_NAME" "dev-predict"
run_created_container "$PREDICT_NAME" "dev-predict"
[[ "$LAST_CONTAINER_EXIT" -eq 0 ]] || fail "position-ACF dev prediction failed"
test -s "${PREDICT_STAGE}/predictions.json"
test -s "${PREDICT_STAGE}/prediction.receipt.json"
test -s "${PREDICT_STAGE}/inputs/dev-pose-cache-snapshot.json"
validate_prediction_bundle \
  "${PREDICT_STAGE}/predictions.json" \
  "${PREDICT_STAGE}/prediction.receipt.json"
sha256sum \
  "${PREDICT_STAGE}/predictions.json" \
  "${PREDICT_STAGE}/prediction.receipt.json" \
  "${PREDICT_STAGE}/inputs/dev-pose-cache-snapshot.json" \
  > "${AUDIT_ROOT}/frozen-prediction-sha256.txt"
chmod -R a-w "$PREDICT_STAGE"
flock -u 9

# This is the first target-bearing boundary. It is reached only after the
# prediction bytes, receipt, and pose snapshot have been frozen read-only.
CURRENT_STAGE="dev-score-target-boundary"
assert_sha256 "$DEV_TARGET" "$DEV_TARGET_SHA256"

CURRENT_STAGE="dev-score-create"
docker create \
  --name "$SCORE_NAME" \
  "${common_args[@]}" \
  --env CUDA_VISIBLE_DEVICES= \
  "${source_args[@]}" \
  --mount \
    "type=bind,src=${SOURCE_VIEW}/${READOUT_CONFIG_RELATIVE},dst=/pams/input/readout-config.yaml,readonly" \
  --mount \
    "type=bind,src=${PREDICT_STAGE}/predictions.json,dst=/pams/frozen/predictions.json,readonly" \
  --mount \
    "type=bind,src=${PREDICT_STAGE}/prediction.receipt.json,dst=/pams/frozen/prediction.receipt.json,readonly" \
  --mount \
    "type=bind,src=${DEV_TARGET},dst=/pams/protocol/dev.targets.json,readonly" \
  --mount "type=bind,src=${SCORE_STAGE},dst=/pams/output" \
  "$IMAGE_ID" \
  python -m pams position-acf dev-score \
    /pams/frozen/predictions.json \
    /pams/frozen/prediction.receipt.json \
    /pams/protocol/dev.targets.json \
    /pams/output \
    --readout-config /pams/input/readout-config.yaml \
  > "${AUDIT_ROOT}/${SCORE_NAME}.create-id.txt"
verify_container "$SCORE_NAME" "dev-score"
run_created_container "$SCORE_NAME" "dev-score"
[[ "$LAST_CONTAINER_EXIT" -eq 0 ]] || fail "position-ACF dev scoring failed"
test -s "${SCORE_STAGE}/evaluation.json"
test -s "${SCORE_STAGE}/evaluation.receipt.json"
validate_score_bundle \
  "${SCORE_STAGE}/evaluation.json" \
  "${SCORE_STAGE}/evaluation.receipt.json"
sha256sum \
  "${SCORE_STAGE}/evaluation.json" \
  "${SCORE_STAGE}/evaluation.receipt.json" \
  > "${AUDIT_ROOT}/dev-score-sha256.txt"
chmod -R a-w "$SCORE_STAGE"

CURRENT_STAGE="dev-stop-decision"
GATE_EVALUATION="${SCORE_STAGE}/evaluation.json" \
GATE_OUTPUT="${RUN_ROOT}/dev-stop-decision.json" \
GATE_REFERENCE_NMAE="$LOCAL_FREQUENCY_NMAE" \
GATE_REFERENCE_OBO="$LOCAL_FREQUENCY_OBO" \
GATE_REFERENCE_SHA256="$LOCAL_FREQUENCY_EVALUATION_SHA256" \
python3 - <<'PY'
import json
import math
import os
from pathlib import Path

evaluation = json.loads(
    Path(os.environ["GATE_EVALUATION"]).read_text(encoding="utf-8")
)
nmae = float(evaluation["report"]["nmae"])
obo = float(evaluation["report"]["obo"])
reference_nmae = float(os.environ["GATE_REFERENCE_NMAE"])
reference_obo = float(os.environ["GATE_REFERENCE_OBO"])
if not all(math.isfinite(value) for value in (nmae, obo)):
    raise RuntimeError("dev stop gate received non-finite metrics")
conditions = {
    "nmae_strictly_better_than_public_local_frequency": nmae < reference_nmae,
    "obo_strictly_better_than_public_local_frequency": obo > reference_obo,
}
passed = all(conditions.values())
payload = {
    "schema_version": 1,
    "artifact_type": "pams_position_acf_dev_stop_decision",
    "decision": (
        "continue-dev-only-analysis" if passed else "stop-position-acf-line"
    ),
    "passed": passed,
    "conditions": conditions,
    "position_acf_dev84": {"nmae": nmae, "obo": obo},
    "public_local_frequency_reference": {
        "evaluation_sha256": os.environ["GATE_REFERENCE_SHA256"],
        "nmae": reference_nmae,
        "obo": reference_obo,
    },
    "scope": "dev84-stop-decision-only",
    "authorizes_additional_seeds": False,
    "test105_evaluation_authorized": False,
}
Path(os.environ["GATE_OUTPUT"]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
chmod 0444 "${RUN_ROOT}/dev-stop-decision.json"

CURRENT_STAGE="final-audit"
write_status "completed" "dev-stop-decision" "0"
(
  cd "$RUN_ROOT"
  find . \
    -type f \
    ! -path './audit/artifact-sha256.txt' \
    ! -name '*.tmp' \
    -print0 \
    | sort -z \
    | xargs -0 -r sha256sum
) > "${AUDIT_ROOT}/artifact-sha256.txt"
chmod 0444 "${AUDIT_ROOT}/artifact-sha256.txt"
chmod -R a-w "$RUN_ROOT"
trap - EXIT HUP INT TERM
printf '%s\n' "$RUN_ROOT"
