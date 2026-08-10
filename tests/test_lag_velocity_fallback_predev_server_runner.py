from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import yaml

RUNNER = (
    Path(__file__).parents[1]
    / "scripts/server/run_pams_position_lag_velocity_fallback_predev_seed2026.sh"
)
READOUT = (
    Path(__file__).parents[1]
    / "configs/readouts/projected_position_lag_velocity_fallback_predev_v2.yaml"
)

READOUT_SHA256 = "b3c38341e442765cb709d8ca9c5ecfe541121ea61d1ae35c0ff2cb48b6b09a98"
READOUT_SEMANTIC_SHA256 = (
    "e17b4105f732be1bc4a40b153c0db28729fc0379a5f921ab0a26d3b2301b4233"
)


def _runner() -> str:
    return RUNNER.read_text(encoding="utf-8")


def _container_create_block(source: str) -> str:
    start = source.index("docker create \\\n")
    end = source.index("\nACTIVE_CONTAINER=", start)
    return source[start:end]


def _heredocs(source: str, marker: str) -> tuple[str, ...]:
    pattern = re.compile(
        rf"<<'{re.escape(marker)}'\n(.*?)\n{re.escape(marker)}",
        re.DOTALL,
    )
    return tuple(pattern.findall(source))


def test_frozen_readout_and_exact_v8_bindings_are_literal() -> None:
    source = _runner()

    assert f'READOUT_CONFIG_SHA256="{READOUT_SHA256}"' in source
    assert (
        f'READOUT_CONFIG_SEMANTIC_SHA256="{READOUT_SEMANTIC_SHA256}"'
        in source
    )
    assert (
        'ENCODER_SHA256="6817fe468d76c3fa2182dd831695d6fe3d6da208a2b7f8dfa90cffa6872e8053"'
        in source
    )
    assert (
        'EXPERIMENT_CONFIG_SHA256="eb4072a195608757cd2542855b33fb000c987f96c94b83a1e448ab78f97e9374"'
        in source
    )
    assert (
        'TRAIN_INPUT_SHA256="e238d307b4ed37b2f6c19eefdf7fa8812ec3d231cd2a7091be38e1c279f04207"'
        in source
    )
    assert (
        'TRAIN_POSE_SET_SHA256="f32d718ae922120778f535a6a4f467edba79cafeba90373b3a6a55bf11be6ee2"'
        in source
    )
    assert READOUT.is_file()
    raw = READOUT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == READOUT_SHA256
    payload = yaml.safe_load(raw)
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == READOUT_SEMANTIC_SHA256


def test_exactly_one_network_none_gpu_container_has_only_predev_mounts() -> None:
    source = _runner()
    block = _container_create_block(source)

    assert source.count("docker create \\\n") == 1
    assert "--network none" in block
    assert "--read-only" in block
    assert '--gpus "device=${GPU_DEVICE}"' in block
    assert "--cap-drop ALL" in block
    assert "--security-opt no-new-privileges:true" in block
    destinations = set(re.findall(r"dst=([^,\"]+)", block))
    assert destinations == {
        "/workspace",
        "/pams/source-export-receipt.json",
        "/pams/input/experiment-config.yaml",
        "/pams/input/readout-config.yaml",
        "/pams/encoder/encoder.pt",
        "/pams/protocol/train.inputs.json",
        "/pams/pose-cache",
        "/pams/predev.py",
        "/pams/output",
    }
    assert "src=${TRAIN_POSE_VIEW},dst=/pams/pose-cache,readonly" in block
    assert "src=${POSE_POOL},dst=/pams/pose-cache" not in block
    for forbidden in (
        "/pams/protocol/dev",
        "/pams/protocol/test",
        "dev.targets",
        "test.targets",
        "test-identity",
        "pose-dev",
        "pose-test",
    ):
        assert forbidden not in block


def test_source_export_and_pose_view_are_bounded_and_non_overwriting() -> None:
    source = _runner()

    assert 'mkdir -- "$RUN_ROOT" || fail "immutable run root exists' in source
    assert "git -C \"$SOURCE_CHECKOUT\" archive" in source
    archive = source[
        source.index('git -C "$SOURCE_CHECKOUT" archive') :
        source.index("SOURCE_EXPORT_ROOT=", source.index("git -C"))
    ]
    assert "\n  src \\" in archive
    assert "\n  pyproject.toml \\" in archive
    assert "\n  data \\" not in archive
    assert "\n  tests \\" not in archive
    assert 'done < <(jq -r \'.records[].video_id\' "$TRAIN_INPUT")' in source
    assert '[[ "$pose_count" -eq 337 ]]' in source
    assert "pose-train337.sha256.tsv" in source
    assert 'chmod 0555 "$TRAIN_POSE_VIEW"' in source
    assert 'with path.open("x", encoding="utf-8", newline="\\n")' in source


def test_container_gate_uses_only_lag_velocity_fallback_api_and_fixed_batches() -> None:
    source = _runner()
    container_python = _heredocs(source, "CONTAINERPY")

    assert len(container_python) == 1
    gate = container_python[0]
    assert "estimate_period_from_projected_position_lag_velocity_fallback" in gate
    assert "projected_position_lag_velocity_fallback_diagnostics" in gate
    assert "estimate_period_from_projected_position(" not in gate
    assert "estimate_period_from_projected_pose(" not in gate
    assert "forward_with_pre_pe" in gate
    assert "for start in range(0, len(items), 32):" in gate
    assert "items[start : start + 32]" in gate
    assert "batch_sizes == ([32] * 10 + [17])" in gate
    assert "diagnostic.selection_source" in gate
    assert '"projected-velocity-spectrum-fallback"' in gate
    assert '{str(index): 0 for index in range(4, 129)}' in gate


def test_synthetic_gates_then_train337_are_complete_and_target_free() -> None:
    source = _runner()
    gate = _heredocs(source, "CONTAINERPY")[0]

    assert "tuple(range(2, 8))" in gate
    assert "drift != 0.02" in gate
    assert "+ drift * time" in gate
    synthetic_index = gate.index(
        "synthetic = synthetic_gate(policy, model, config, device)"
    )
    sweep_index = gate.index(
        "synthetic_count_sweep = synthetic_count_sweep_gate(",
        synthetic_index,
    )
    train_index = gate.index("train337 = train337_gate(", sweep_index)
    assert synthetic_index < sweep_index < train_index
    assert "counts != tuple(range(2, 41))" in gate
    assert "drift != 1.0" in gate
    assert "batch_sizes == [32, 7]" in gate
    assert "maximum_relative_period_error_lte_frozen_maximum" in gate
    assert "recovered_fraction_gte_frozen_minimum" in gate
    for condition in (
        "fundamental_selection_fraction_gte_frozen_minimum",
        "zero_confidence_fraction_lte_frozen_maximum",
        "minimum_period_fraction_lte_frozen_maximum",
        "maximum_period_fraction_lte_frozen_maximum",
        "selected_period_mode_fraction_lte_frozen_maximum",
        "unique_selected_periods_gte_frozen_minimum",
        "zero_confidence_zero_count",
    ):
        assert condition in gate
    assert "load_pose_input_manifest" in gate
    assert "load_pose_cache_set" in gate
    assert "load_dev_target_manifest" not in gate
    for forbidden in ('row["action"]', 'row["target"]', 'row["label"]'):
        assert forbidden not in gate


def test_artifact_contains_provenance_hardware_histograms_and_mount_audit() -> None:
    gate = _heredocs(_runner(), "CONTAINERPY")[0]

    assert '"checkpoint_provenance": provenance.to_dict()' in gate
    assert '"runtime_provenance": {' in gate
    assert '"hardware": hardware_fingerprint()' in gate
    assert '"selected_period_histogram": complete_period_histogram(rows)' in gate
    assert '"finite_float_9_significant_digits_or_none"' in gate
    assert '"selection_source_counts": complete_selection_source_counts(rows)' in gate
    assert '"synthetic_count_2_to_40_with_linear_drift"' in gate
    assert '"mount_audit": {' in gate
    assert '"dev84_identity_mounted": False' in gate
    assert '"dev84_pose_mounted": False' in gate
    assert '"dev84_targets_mounted": False' in gate
    assert '"test105_identity_mounted": False' in gate
    assert '"test105_labels_mounted": False' in gate
    assert '"dev84_scoring_authorized": False' in gate
    assert '"test105_evaluation_authorized": False' in gate


def test_failure_paths_retain_artifact_receipt_hashes_and_logs() -> None:
    source = _runner()

    assert "trap on_exit EXIT" in source
    assert 'elif [[ "$CONTAINER_EXIT" -eq 3 ]]; then' in source
    assert 'elif [[ "$attach_exit" -ne "$CONTAINER_EXIT" ]]; then' in source
    assert "container did not retain a predev artifact" in source
    assert "container exited without retaining predev.json" in source
    assert "pams_position_lag_velocity_fallback_predev_receipt" in source
    assert '"predev_sha256": hashlib.sha256(artifact_bytes).hexdigest()' in source
    assert "container_pre_run_inspect_sha256" in source
    assert "container_post_run_inspect_sha256" in source
    assert "write_hash_manifest" in source
    assert "! -path './audit/artifact-sha256.txt'" in source
    assert '${CONTAINER_NAME}.stdout.log' in source
    assert '${CONTAINER_NAME}.stderr.log' in source
    assert 'chmod -R a-w "$RUN_ROOT"' in source
    assert 'CURRENT_STAGE="predev-gates-failed"' in source
    assert 'exit "$CONTAINER_EXIT"' in source
    assert "if [[ \"$CONTAINER_EXIT\" -eq 0 ]]; then" in source


def test_all_host_python_heredocs_parse_as_python38() -> None:
    snippets = _heredocs(_runner(), "HOSTPY")

    assert len(snippets) == 5
    for snippet in snippets:
        ast.parse(snippet, feature_version=(3, 8))


def test_embedded_container_program_is_syntax_valid() -> None:
    snippets = _heredocs(_runner(), "CONTAINERPY")

    assert len(snippets) == 1
    ast.parse(snippets[0], feature_version=(3, 8))
