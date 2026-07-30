from __future__ import annotations

import ast
import re
from pathlib import Path

RUNNER = (
    Path(__file__).parents[1]
    / "scripts/server/run_pams_position_lag_velocity_fallback_dev_seed2026.sh"
)


def _runner() -> str:
    return RUNNER.read_text(encoding="utf-8")


def _heredocs(source: str, marker: str) -> tuple[str, ...]:
    pattern = re.compile(
        rf"<<'{re.escape(marker)}'\n(.*?)\n{re.escape(marker)}",
        re.DOTALL,
    )
    return tuple(pattern.findall(source))


def _docker_blocks(source: str) -> tuple[str, ...]:
    starts = [match.start() for match in re.finditer(r"^docker create \\\n", source, re.M)]
    return tuple(source[start : source.find("\nrun_container ", start)] for start in starts)


def test_formal_predev_and_unfrozen_bindings_are_mandatory() -> None:
    source = _runner()

    for name in (
        "PAMS_SOURCE_REVISION",
        "PAMS_READOUT_CONFIG_SHA256",
        "PAMS_READOUT_CONFIG_SEMANTIC_SHA256",
        "PAMS_PREDEV_ARTIFACT",
        "PAMS_PREDEV_ARTIFACT_SHA256",
    ):
        assert f"${{{name}:?{name} is required}}" in source
    assert 'artifact_type": "pams_position_lag_velocity_fallback_predev"' in source
    assert '"status": "passed"' in source
    assert '"stop_decision": "eligible_to_build_separate_frozen_dev84_protocol"' in source
    assert '"synthetic_count_2_to_40_with_linear_drift"' in source
    assert 'EXPECTED_SOURCE_REVISION="470d49bd97eef5adad89142d4ffc017125505525"' in source
    assert (
        'EXPECTED_READOUT_CONFIG_SHA256="b3c38341e442765cb709d8ca9c5ecfe541121'
        'ea61d1ae35c0ff2cb48b6b09a98"' in source
    )
    assert (
        'EXPECTED_PREDEV_ARTIFACT_SHA256="f563cffa4235b34e091bf9b93af9123a93fa'
        '1d57ee29dc39f54f4feb60e0d928"' in source
    )
    assert "validated_before_dev_inputs_or_pose_access" in source
    predev_validation = source.index("formal predev binding mismatch")
    dev_identity_read = source.index('assert_sha256 "$DEV_INPUT"')
    pose_pool_read = source.index('test -d "$POSE_POOL"')
    assert predev_validation < dev_identity_read < pose_pool_read
    assert 'orchestrator_runner_sha256": "${RUNNER_SHA256}"' in source
    assert 'cp -- "$RUNNER_PATH" "${AUDIT_ROOT}/orchestrator-runner.sh"' in source


def test_prediction_and_score_are_two_network_none_containers() -> None:
    blocks = _docker_blocks(_runner())

    assert len(blocks) == 2
    prediction, score = blocks
    common = _runner()[
        _runner().index("common_args=(") : _runner().index(
            "\n)\n\nCURRENT_STAGE=", _runner().index("common_args=(")
        )
    ]
    assert "--network none" in common
    assert "--read-only" in common
    assert "--cap-drop ALL" in common
    assert '"${common_args[@]}"' in prediction
    assert '"${common_args[@]}"' in score
    assert '"PAMS_EXPERIMENT_CONFIG_SHA256=${EXPERIMENT_CONFIG_SHA256}"' in common
    assert "--gpus" in prediction
    assert "--gpus" not in score
    assert "CUDA_VISIBLE_DEVICES=" in score


def test_prediction_mounts_only_target_free_dev_inputs() -> None:
    prediction = _docker_blocks(_runner())[0]
    destinations = set(re.findall(r"dst=([^,\"]+)", prediction))

    assert destinations == {
        "/workspace",
        "/pams/input/readout-config.yaml",
        "/pams/input/experiment-config.yaml",
        "/pams/encoder/encoder.pt",
        "/pams/protocol/dev.inputs.json",
        "/pams/pose-cache",
        "/pams/predict.py",
        "/pams/output",
    }
    assert "dev.targets" not in prediction
    assert "PREDEV_ARTIFACT},dst=" not in prediction
    assert "POSE_POOL},dst=" not in prediction
    assert "src=${DEV_POSE_VIEW},dst=/pams/pose-cache,readonly" in prediction


def test_score_mounts_prediction_targets_and_metric_code_but_no_model_assets() -> None:
    score = _docker_blocks(_runner())[1]
    destinations = set(re.findall(r"dst=([^,\"]+)", score))

    assert destinations == {
        "/pams/metric-code/metrics.py",
        "/pams/frozen/predictions.json",
        "/pams/frozen/prediction.receipt.json",
        "/pams/protocol/dev.targets.json",
        "/pams/score.py",
        "/pams/output",
    }
    for forbidden in (
        "/workspace",
        "/pams/pose-cache",
        "/pams/encoder",
        "experiment-config",
        "readout-config",
    ):
        assert forbidden not in score


def test_target_boundary_occurs_after_prediction_freeze() -> None:
    source = _runner()

    frozen = source.index('chmod -R a-w "$PREDICT_STAGE"')
    boundary = source.index('CURRENT_STAGE="dev-score-target-boundary"')
    first_target_read = source.index('assert_sha256 "$DEV_TARGET"')
    score_create = source.index('CURRENT_STAGE="dev-score-create"')
    assert frozen < boundary < first_target_read < score_create


def test_prediction_schema_contains_the_required_near90_fields() -> None:
    predict = _heredocs(_runner(), "PREDICTPY")

    assert len(predict) == 1
    program = predict[0]
    for field in (
        '"video_id"',
        '"sampled_frames"',
        '"valid_frames"',
        '"period_frames"',
        '"confidence"',
        '"selection_source"',
        '"raw_count"',
        '"rounded_count"',
    ):
        assert field in program
    assert "estimate_period_from_projected_position_lag_velocity_fallback" in program
    assert "projected_position_lag_velocity_fallback_diagnostics" in program
    assert "forward_with_pre_pe" in program
    assert "batch_sizes != [32, 32, 20]" in program
    assert "valid_frames not in {0, sampled_frames}" in program
    assert "partial valid-mask coverage would bias" in program
    assert '"dev84_targets_mounted": False' in program


def test_score_validates_predictions_before_opening_targets() -> None:
    score = _heredocs(_runner(), "SCOREPY")

    assert len(score) == 1
    program = score[0]
    validate_call = program.index(
        "prediction, rows, prediction_bytes, receipt_bytes = validate_predictions("
    )
    target_read = program.index("target_sha_before = sha256_file(args.targets)")
    assert validate_call < target_read
    assert "prediction receipt hash mismatch" in program
    assert "prediction schema mismatch" in program
    assert "prediction and target identity/order mismatch" in program


def test_score_reports_requested_metrics_and_paired_bootstrap() -> None:
    score = _heredocs(_runner(), "SCOREPY")[0]

    for metric in (
        '"nmae_rounded"',
        '"mae_raw_prediction"',
        '"rmse_raw_prediction"',
        '"obo_rounded"',
        '"exact_rounded"',
    ):
        assert metric in score
    assert "np.random.default_rng(2026)" in score
    assert "np.empty(10_000)" in score
    assert '"samples": 10_000' in score
    assert '"seed": 2026' in score
    assert '"pairing": "paired_prediction_target_rows"' in score
    assert "np.quantile(values, 0.025)" in score
    assert "np.quantile(values, 0.975)" in score


def test_test105_has_no_host_path_or_container_mount() -> None:
    source = _runner()
    blocks = _docker_blocks(source)

    assert "TEST_INPUT" not in source
    assert "TEST_TARGET" not in source
    assert "TEST_POSE" not in source
    assert all("/pams/protocol/test" not in block for block in blocks)
    assert all("/pams/test" not in block for block in blocks)


def test_failures_are_retained_and_container_exit_code_is_checked() -> None:
    source = _runner()

    assert "trap on_exit EXIT" in source
    assert "write_hash_manifest" in source
    assert 'actual_exit="$(docker inspect "$name" --format' in source
    assert '"$actual_exit" -eq "$CONTAINER_EXIT"' in source
    assert '[[ "$CONTAINER_EXIT" -eq 0 ]] || fail "$stage container failed"' in source
    assert 'chmod -R a-w "$RUN_ROOT"' in source


def test_embedded_python_is_python38_parseable() -> None:
    source = _runner()
    snippets = (
        *_heredocs(source, "HOSTPY"),
        *_heredocs(source, "PREDICTPY"),
        *_heredocs(source, "SCOREPY"),
    )

    assert len(_heredocs(source, "HOSTPY")) == 3
    assert len(_heredocs(source, "PREDICTPY")) == 1
    assert len(_heredocs(source, "SCOREPY")) == 1
    for snippet in snippets:
        ast.parse(snippet, feature_version=(3, 8))
