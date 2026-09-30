from __future__ import annotations

import json

import numpy as np
import pytest

from pams.temporac.certify import (
    bind_natural_certificate_track,
    build_natural_teacher_output,
    certify_target,
)
from pams.temporac.contract import CONTRACT_SHA256, FEATURE_RECEIPT_SCHEMA, SEEDS
from pams.temporac.evaluator import (
    CertifiedDevelopmentTarget,
    EvaluatorError,
    OneUseEvaluator,
    count_metric_fixture_receipt,
    validate_certified_pulse_units,
    validate_certified_target_commitment,
    validate_metric_fixture_receipt,
    validate_vault_join,
    validate_vault_period_units,
)
from pams.temporac.hashio import (
    canonical_json_bytes,
    feature_npz_bytes,
    read_target_npz_bytes,
    sha256_bytes,
)
from pams.temporac.metrics import (
    ARMS,
    CONDITIONS,
    ROUTE_BOOTSTRAP_DRAWS,
    CountMetricRow,
    MetricError,
    decide_k7,
    paired_component_bootstrap,
    parse_metric_rows,
    score_count_metrics,
)
from pams.temporac.preprocess import preprocess_identity
from pams.temporac.receipts import (
    LoadedTargetArtifact,
    build_target_artifact,
    member_payload,
    parse_receipt_bytes,
    receipt_bytes,
    target_count_root_sha256,
)
from pams.temporac.trusted_packer import VaultRow
from pams.temporac.types import FeatureRecord
from pams.temporac.x0 import generate_view


def _rows() -> tuple[CountMetricRow, ...]:
    estimates = {
        ("local", "natural-drift"): 9,
        ("global", "natural-drift"): 8,
        ("uniform", "natural-drift"): 8,
        ("capacity-control", "natural-drift"): 7,
        ("local", "natural-clean"): 10,
        ("global", "natural-clean"): 10,
        ("uniform", "natural-clean"): 10,
        ("capacity-control", "natural-clean"): 10,
    }
    return tuple(
        CountMetricRow(
            arm=arm,
            seed=seed,
            condition=condition,
            video_token=f"video-{component}",
            slot=0,
            component_token=f"component-{component}",
            count_gt=10,
            estimate=estimates[(arm, condition)],
        )
        for component in range(8)
        for arm in ARMS
        for seed in SEEDS
        for condition in CONDITIONS
    )


def _mapping(row: CountMetricRow) -> dict[str, object]:
    return {
        "arm": row.arm,
        "component_token": row.component_token,
        "condition": row.condition,
        "count_gt": row.count_gt,
        "estimate": row.estimate,
        "seed": row.seed,
        "slot": row.slot,
        "video_token": row.video_token,
    }


def _certified_evidence() -> tuple[
    CertifiedDevelopmentTarget,
    tuple[dict[str, object], ...],
    VaultRow,
]:
    view = generate_view(3, 0, resampler="linear", offset=0)
    motion = np.asarray(view.motion[:320], dtype="<f4").copy()
    motion[312:] = 0.0
    frame_mask = np.ones(320, dtype="|u1")
    frame_mask[312:] = 0
    feature = FeatureRecord(
        frame_mask=frame_mask,
        local_person_slot=np.asarray([0], dtype="<i8"),
        motion=motion,
        opaque_sample_key=np.arange(32, dtype="|u1"),
        person_mask=np.asarray([1], dtype="|u1"),
        sampled_frame_indices=np.asarray(view.clock[:320], dtype="<i8"),
        source_length=np.asarray([int(view.clock[-1]) + 1], dtype="<i8"),
    )
    feature_artifact, feature_members = feature_npz_bytes(feature)
    feature_receipt_payload: dict[str, object] = {
        "artifact_bytes": len(feature_artifact),
        "artifact_sha256": sha256_bytes(feature_artifact),
        "contract_sha256": CONTRACT_SHA256,
        "members": member_payload(feature_members),
        "opaque_key_hex": feature.opaque_key_bytes.hex(),
        "schema": FEATURE_RECEIPT_SCHEMA,
        "slot": feature.slot,
        "source_binding_sha256": "a" * 64,
    }
    feature_receipt = receipt_bytes(
        feature_receipt_payload,
        expected_schema=FEATURE_RECEIPT_SCHEMA,
    )
    feature_receipt_sha256 = sha256_bytes(feature_receipt)
    preprocessed = preprocess_identity(
        feature.motion,
        feature.frame_mask,
        feature.sampled_frame_indices,
    )
    output = build_natural_teacher_output(
        feature=feature,
        feature_receipt_bytes=feature_receipt,
        expected_feature_receipt_sha256=feature_receipt_sha256,
        selected_teacher_sha256="1" * 64,
        phase=np.asarray(view.phase[: preprocessed.motion.shape[0]], dtype="<f8"),
        reconstruction=np.asarray(preprocessed.teacher_input[:, :149], dtype="<f8"),
    )
    track = bind_natural_certificate_track(feature=feature, teacher_output=output)
    target = certify_target(track, "natural")
    assert target.certified
    built = build_target_artifact(target)
    target_arrays, target_members = read_target_npz_bytes(built.artifact_bytes)
    loaded = LoadedTargetArtifact(
        arrays=target_arrays,
        artifact_bytes=built.artifact_bytes,
        artifact_sha256=built.artifact_sha256,
        members=target_members,
        receipt=parse_receipt_bytes(built.receipt_bytes),
        receipt_bytes=built.receipt_bytes,
        receipt_sha256=built.receipt_sha256,
    )
    evidence = CertifiedDevelopmentTarget(
        feature=feature,
        feature_receipt_bytes=feature_receipt,
        feature_receipt_sha256=feature_receipt_sha256,
        natural_input_receipt_bytes=output.input_receipt_bytes,
        natural_input_receipt_sha256=output.input_receipt_sha256,
        target=loaded,
    )
    count = evidence.teacher_target_count
    rows = (
        {
            "opaque_key_hex": evidence.identity[0],
            "slot": evidence.identity[1],
            "teacher_target_count": count,
        },
    )
    starts = evidence.source_clocks[:-1][evidence.pulse == 1]
    vault = VaultRow(
        evidence.identity[0],
        evidence.identity[1],
        int(feature.source_length[0]),
        count,
        tuple((int(start), int(start) + 5) for start in starts),
    )
    return evidence, rows, vault


def test_video_first_ordered_seed_point_and_per_draw_comparator_reselection() -> None:
    rows = _rows()
    point = score_count_metrics(rows)
    assert point.avg_mae[("local", "natural-drift")] == pytest.approx(0.1)
    assert point.avg_mae[("global", "natural-drift")] == pytest.approx(0.2)
    assert point.avg_obo[("local", "natural-drift")] == 1.0
    assert point.avg_obo[("global", "natural-drift")] == 0.0
    assert point.strongest_comparator == "global"
    assert point.route_margin == pytest.approx(0.1)
    assert point.clean_delta == 0.0
    bootstrap = paired_component_bootstrap(rows)
    assert bootstrap.route_draws.shape == (ROUTE_BOOTSTRAP_DRAWS,)
    assert bootstrap.route_lower == pytest.approx(0.1)
    assert bootstrap.clean_upper == 0.0
    assert bootstrap.comparator_draw_counts == {
        "global": 10_000,
        "uniform": 0,
        "capacity-control": 0,
    }
    assert decide_k7(point, bootstrap).status == "PASS"


def test_people_within_video_then_equal_video_weighting() -> None:
    rows = list(_rows())
    # Add a second person to video-0 with a much larger local drift error.
    for arm in ARMS:
        for seed in SEEDS:
            for condition in CONDITIONS:
                base = next(
                    row
                    for row in rows
                    if row.video_token == "video-0"
                    and row.arm == arm
                    and row.seed == seed
                    and row.condition == condition
                )
                rows.append(
                    CountMetricRow(
                        arm=arm,
                        seed=seed,
                        condition=condition,
                        video_token="video-0",
                        slot=1,
                        component_token="component-0",
                        count_gt=10,
                        estimate=0
                        if (arm, condition) == ("local", "natural-drift")
                        else base.estimate,
                    )
                )
    point = score_count_metrics(rows)
    # video-0 local drift NAE=(0.1+1.0)/2=.55; seven videos=.1 each; equal-video mean=.15625.
    assert point.avg_mae[("local", "natural-drift")] == pytest.approx(0.15625)


@pytest.mark.parametrize(
    "attack",
    [
        "empty",
        "missing",
        "duplicate",
        "float_gt",
        "nan_estimate",
        "unknown_key",
    ],
)
def test_malformed_duplicate_empty_nonfinite_and_type_attacks_fail_globally(attack: str) -> None:
    payload = [_mapping(row) for row in _rows()]
    if attack == "empty":
        payload = []
    elif attack == "missing":
        payload.pop()
    elif attack == "duplicate":
        payload.append(dict(payload[0]))
    elif attack == "float_gt":
        payload[0]["count_gt"] = 10.0
    elif attack == "nan_estimate":
        payload[0]["estimate"] = float("nan")
    elif attack == "unknown_key":
        payload[0]["count_gt_copy"] = 10
    with pytest.raises(MetricError):
        parsed = parse_metric_rows(payload)
        score_count_metrics(parsed)


def test_vault_join_periods_and_one_use_prevalidation_are_fail_closed() -> None:
    rows = (
        VaultRow("0" * 64, 0, 40, 2, ((0, 10), (20, 30))),
        VaultRow("1" * 64, 0, 40, 1, ((5, 15),)),
    )
    joined = validate_vault_join((("0" * 64, 0), ("1" * 64, 0)), rows, expected_cardinality=2)
    assert set(joined) == {("0" * 64, 0), ("1" * 64, 0)}
    validate_vault_period_units(rows)
    with pytest.raises(EvaluatorError, match="duplicate"):
        validate_vault_join((("0" * 64, 0), ("0" * 64, 0)), rows, expected_cardinality=2)
    with pytest.raises(EvaluatorError, match="sub-P_min"):
        validate_vault_period_units((VaultRow("2" * 64, 0, 20, 1, ((0, 4),)),))
    capability = OneUseEvaluator()
    with pytest.raises(ValueError):
        capability.run(
            grant={},
            g5a_receipt={},
            evaluator_code_sha256="0" * 64,
            population_identities=(),
            vault_rows=(),
            predictions=(),
            certified_development_targets=(),
            target_count_rows=(),
        )
    assert not capability.consumed
    with pytest.raises(ValueError):
        capability.run(
            grant={},
            g5a_receipt={},
            evaluator_code_sha256="0" * 64,
            population_identities=(),
            vault_rows=(),
            predictions=(),
            certified_development_targets=(),
            target_count_rows=(),
        )


def test_k7_consumes_target_count_rows_and_actual_pulse_source_clock_content() -> None:
    evidence, rows, vault_row = _certified_evidence()
    root = target_count_root_sha256(rows)
    pulse_starts = validate_certified_target_commitment(
        (evidence,),
        rows,
        committed_target_count_root_sha256=root,
        prediction_identities=frozenset({evidence.identity}),
    )
    assert pulse_starts[evidence.identity] == tuple(
        int(value) for value in evidence.source_clocks[:-1][evidence.pulse == 1]
    )
    validate_certified_pulse_units({evidence.identity: vault_row}, pulse_starts)

    teacher_target_count = rows[0]["teacher_target_count"]
    assert isinstance(teacher_target_count, int)
    wrong_count_rows = (
        {
            **rows[0],
            "teacher_target_count": teacher_target_count + 1,
        },
    )
    with pytest.raises(EvaluatorError, match="identity-wise equality"):
        validate_certified_target_commitment(
            (evidence,),
            wrong_count_rows,
            committed_target_count_root_sha256=target_count_root_sha256(wrong_count_rows),
            prediction_identities=frozenset({evidence.identity}),
        )

    periods = list(vault_row.periods)
    first_start, first_end = periods[0]
    periods[0] = (first_start + 1, first_end + 1)
    wrong_interval = VaultRow(
        vault_row.opaque_key_hex,
        vault_row.slot,
        vault_row.source_length,
        vault_row.count_gt,
        tuple(periods),
    )
    with pytest.raises(EvaluatorError, match="exactly one pulse"):
        validate_certified_pulse_units({evidence.identity: wrong_interval}, pulse_starts)


def test_p05m_surrounding_receipt_binds_exact_bytes() -> None:
    fixture = canonical_json_bytes([_mapping(row) for row in _rows()])
    expected = canonical_json_bytes({"status": "expected"})
    attacks = canonical_json_bytes({"all_attacks": "GLOBAL_FAIL"})
    receipt = count_metric_fixture_receipt(
        evaluator_code_sha256="a" * 64,
        fixture_input_bytes=fixture,
        expected_output_bytes=expected,
        attack_expected_global_fail_bytes=attacks,
    )
    assert receipt["fixture_input_sha256"] == sha256_bytes(fixture)
    assert receipt["expected_output_sha256"] == sha256_bytes(expected)
    assert json.loads(canonical_json_bytes(dict(receipt)))["status"] == "PASS"
    validate_metric_fixture_receipt(receipt)
    with pytest.raises(EvaluatorError, match="unknown or missing"):
        validate_metric_fixture_receipt({**receipt, "unexpected": "0" * 64})
