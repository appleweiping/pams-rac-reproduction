"""Independent, synthetic-only adversarial probes for the Round-2 review.

This file intentionally imports no test helper and opens no natural, sealed,
heldout, result, or server artifact.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from pams.temporac.certify import (
    CertifiedTarget,
    bind_natural_certificate_track,
    build_natural_teacher_output,
    certificate_track_from_x0,
    certify_target,
)
from pams.temporac.contract import (
    CAPABILITY_GRANT_SCHEMA,
    CONTRACT_SHA256,
    FEATURE_RECEIPT_SCHEMA,
    G5A_RECEIPT_SCHEMA,
    SEEDS,
)
from pams.temporac.evaluator import (
    CertifiedDevelopmentTarget,
    OneUseEvaluator,
    PredictionEnvelope,
    count_metric_fixture_receipt,
    validate_certified_pulse_units,
)
from pams.temporac.hashio import (
    feature_npz_bytes,
    read_target_npz_bytes,
    sha256_bytes,
)
from pams.temporac.prediction import PredictionIdentity, build_identity_prediction
from pams.temporac.preprocess import preprocess_identity
from pams.temporac.receipts import (
    LoadedTargetArtifact,
    build_target_artifact,
    member_payload,
    parse_receipt_bytes,
    receipt_bytes,
    target_count_root_sha256,
)
from pams.temporac.trusted_packer import TrustedSourceObject, VaultRow, vault_bytes
from pams.temporac.types import FeatureRecord
from pams.temporac.x0 import generate_view


def feature_for_view(source_id: int, *, key: bytes, slot: int = 0) -> FeatureRecord:
    view = generate_view(source_id, 0, resampler="linear", offset=0)
    motion = np.asarray(view.motion[:320], dtype="<f4").copy()
    motion[312:] = 0.0
    frame_mask = np.ones(320, dtype="|u1")
    frame_mask[312:] = 0
    return FeatureRecord(
        frame_mask=frame_mask,
        local_person_slot=np.asarray([slot], dtype="<i8"),
        motion=motion,
        opaque_sample_key=np.frombuffer(key, dtype="|u1").copy(),
        person_mask=np.asarray([1], dtype="|u1"),
        sampled_frame_indices=np.asarray(view.clock[:320], dtype="<i8"),
        source_length=np.asarray([int(view.clock[-1]) + 1], dtype="<i8"),
    )


def feature_receipt(feature: FeatureRecord, source_binding: str) -> tuple[bytes, str]:
    artifact, members = feature_npz_bytes(feature)
    payload: dict[str, object] = {
        "artifact_bytes": len(artifact),
        "artifact_sha256": sha256_bytes(artifact),
        "contract_sha256": CONTRACT_SHA256,
        "members": member_payload(members),
        "opaque_key_hex": feature.opaque_key_bytes.hex(),
        "schema": FEATURE_RECEIPT_SCHEMA,
        "slot": feature.slot,
        "source_binding_sha256": source_binding,
    }
    encoded = receipt_bytes(payload, expected_schema=FEATURE_RECEIPT_SCHEMA)
    return encoded, sha256_bytes(encoded)


def natural_output(
    feature: FeatureRecord,
    *,
    source_id: int,
    teacher_sha256: str,
    source_binding: str,
):
    encoded, digest = feature_receipt(feature, source_binding)
    preprocessed = preprocess_identity(
        feature.motion, feature.frame_mask, feature.sampled_frame_indices
    )
    view = generate_view(source_id, 0, resampler="linear", offset=0)
    output = build_natural_teacher_output(
        feature=feature,
        feature_receipt_bytes=encoded,
        expected_feature_receipt_sha256=digest,
        selected_teacher_sha256=teacher_sha256,
        phase=np.asarray(view.phase[: preprocessed.motion.shape[0]], dtype="<f8"),
        reconstruction=np.asarray(preprocessed.teacher_input[:, :149], dtype="<f8"),
    )
    return encoded, digest, output


def loaded_target(target: CertifiedTarget) -> LoadedTargetArtifact:
    built = build_target_artifact(target)
    arrays, members = read_target_npz_bytes(built.artifact_bytes)
    return LoadedTargetArtifact(
        arrays=arrays,
        artifact_bytes=built.artifact_bytes,
        artifact_sha256=built.artifact_sha256,
        members=members,
        receipt=parse_receipt_bytes(built.receipt_bytes),
        receipt_bytes=built.receipt_bytes,
        receipt_sha256=built.receipt_sha256,
    )


def response_for_count(count: int) -> tuple[np.ndarray, np.ndarray]:
    response = np.full(max(1, 2 * count - 1), 0.1, dtype=np.float64)
    if count:
        response[::2] = 0.9
    return response, np.ones(response.size, dtype=np.uint8)


def make_predictions(
    identities: tuple[tuple[str, int], ...], counts: dict[tuple[str, int], int]
) -> tuple[PredictionEnvelope, ...]:
    envelopes: list[PredictionEnvelope] = []
    for component, identity in enumerate(identities):
        key, slot = identity
        ground_truth = counts[identity]
        for arm in ("local", "global", "uniform", "capacity-control"):
            for seed in SEEDS:
                for condition in ("natural-clean", "natural-drift"):
                    estimate = ground_truth
                    if condition == "natural-drift":
                        estimate = ground_truth - (1 if arm == "local" else 2)
                    response, mask = response_for_count(estimate)
                    prediction = build_identity_prediction(
                        PredictionIdentity(bytes.fromhex(key), slot),
                        condition=condition,
                        response=response,
                        edge_mask=mask,
                        decoder_mask=mask,
                        run_bounds=np.asarray([[0, response.size]], dtype=np.int32),
                    )
                    envelopes.append(
                        PredictionEnvelope(
                            arm=arm,  # type: ignore[arg-type]
                            seed=seed,
                            component_token=f"component-{component}",
                            prediction=prediction,
                        )
                    )
    return tuple(envelopes)


def main() -> None:
    probe: dict[str, object] = {}
    key = bytes(range(32))
    feature_a = feature_for_view(3, key=key)
    receipt_a, receipt_a_sha, output_a1 = natural_output(
        feature_a,
        source_id=3,
        teacher_sha256="1" * 64,
        source_binding="a" * 64,
    )
    _, _, output_a2 = natural_output(
        feature_a,
        source_id=3,
        teacher_sha256="2" * 64,
        source_binding="a" * 64,
    )
    target_a1 = certify_target(
        bind_natural_certificate_track(feature=feature_a, teacher_output=output_a1),
        "natural",
    )
    target_a2 = certify_target(
        bind_natural_certificate_track(feature=feature_a, teacher_output=output_a2),
        "natural",
    )
    assert target_a1.certified and target_a2.certified
    probe["arbitrary_teacher_digest_relabel"] = {
        "accepted": True,
        "same_phase": bool(np.array_equal(output_a1.phase, output_a2.phase)),
        "same_reconstruction": bool(
            np.array_equal(output_a1.reconstruction, output_a2.reconstruction)
        ),
        "same_pulse": bool(np.array_equal(target_a1.pulse, target_a2.pulse)),
        "same_chi": bool(np.array_equal(target_a1.chi, target_a2.chi)),
        "teacher_1": target_a1.provenance.teacher_sha256 if target_a1.provenance else None,
        "teacher_2": target_a2.provenance.teacher_sha256 if target_a2.provenance else None,
    }

    immutable_results: dict[str, bool] = {}
    for name, array in {
        "feature_motion": feature_a.motion,
        "teacher_phase": output_a1.phase,
        "certificate_pulse": target_a1.pulse,
        "loaded_pulse": loaded_target(target_a1).arrays["pulse"],
        "x0_phase": generate_view(3, 0, resampler="linear", offset=0).phase,
    }.items():
        try:
            array.flags.writeable = True
            immutable_results[name] = False
        except ValueError:
            immutable_results[name] = True
    probe["bytes_backed_immutability"] = immutable_results

    replacement_rejected = False
    try:
        replace(target_a1, pulse=np.array(target_a1.pulse, copy=True))
    except ValueError:
        replacement_rejected = True
    probe["dataclasses_replace_target_rejected"] = replacement_rejected

    canonical_view = generate_view(3, 0, resampler="linear", offset=0)
    duck_rejected = False
    noncanonical_rejected = False
    try:
        certificate_track_from_x0(object(), teacher_sha256="1" * 64)  # type: ignore[arg-type]
    except ValueError:
        duck_rejected = True
    changed_phase = np.array(canonical_view.phase, copy=True)
    changed_phase[0] *= -1
    try:
        certificate_track_from_x0(
            replace(canonical_view, phase=changed_phase), teacher_sha256="1" * 64
        )
    except ValueError:
        noncanonical_rejected = True
    probe["x0_binding"] = {
        "duck_rejected": duck_rejected,
        "noncanonical_content_rejected": noncanonical_rejected,
    }

    # Same natural key/slot and teacher, but a different canonical feature and
    # teacher-output receipt.  The target/input association has no hash link.
    feature_b = feature_for_view(4, key=key)
    _, _, output_b = natural_output(
        feature_b,
        source_id=4,
        teacher_sha256="1" * 64,
        source_binding="b" * 64,
    )
    target_b = certify_target(
        bind_natural_certificate_track(feature=feature_b, teacher_output=output_b),
        "natural",
    )
    assert target_b.certified
    swapped = CertifiedDevelopmentTarget(
        feature=feature_a,
        feature_receipt_bytes=receipt_a,
        feature_receipt_sha256=receipt_a_sha,
        natural_input_receipt_bytes=output_a1.input_receipt_bytes,
        natural_input_receipt_sha256=output_a1.input_receipt_sha256,
        target=loaded_target(target_b),
    )
    probe["natural_target_input_swap"] = {
        "accepted": True,
        "identity": list(swapped.identity),
        "feature_a_pulse_count": target_a1.event_count,
        "feature_b_pulse_count": target_b.event_count,
        "swapped_pulse_count": swapped.teacher_target_count,
        "pulse_ledgers_differ": bool(not np.array_equal(target_a1.pulse, target_b.pulse)),
    }

    # Full evaluator witness: a G5a receipt claims 9,648 artifacts and G5b
    # joins 402 vault rows, while K7 receives only eight identities (192
    # predictions) and one certified target.  Current code returns PASS.
    evidence = CertifiedDevelopmentTarget(
        feature=feature_a,
        feature_receipt_bytes=receipt_a,
        feature_receipt_sha256=receipt_a_sha,
        natural_input_receipt_bytes=output_a1.input_receipt_bytes,
        natural_input_receipt_sha256=output_a1.input_receipt_sha256,
        target=loaded_target(target_a1),
    )
    target_rows = (
        {
            "opaque_key_hex": evidence.identity[0],
            "slot": evidence.identity[1],
            "teacher_target_count": evidence.teacher_target_count,
        },
    )
    other_identities = tuple((f"{index:064x}", 0) for index in range(1, 402))
    population = (evidence.identity, *other_identities)
    selected = population[:8]
    counts = {identity: (evidence.teacher_target_count if identity == evidence.identity else 10) for identity in selected}
    predictions = make_predictions(selected, counts)
    starts = evidence.source_clocks[:-1][evidence.pulse == 1]
    vault_rows: list[VaultRow] = [
        VaultRow(
            evidence.identity[0],
            evidence.identity[1],
            int(feature_a.source_length[0]),
            evidence.teacher_target_count,
            tuple((int(start), int(start) + 5) for start in starts),
        )
    ]
    for identity in other_identities:
        vault_rows.append(
            VaultRow(identity[0], identity[1], 200, 10, tuple((10 * i, 10 * i + 5) for i in range(10)))
        )
    evaluator_code = "e" * 64
    g5a: dict[str, object] = {
        "checkpoint_root_sha256": "0" * 64,
        "code_sha256": "1" * 64,
        "contract_sha256": CONTRACT_SHA256,
        "environment_sha256": "2" * 64,
        "evaluator_code_sha256": evaluator_code,
        "feature_root_sha256": "3" * 64,
        "population_manifest_sha256": "4" * 64,
        "prediction_clean_root_sha256": "5" * 64,
        "prediction_drift_root_sha256": "6" * 64,
        "receipt_root_sha256": "7" * 64,
        "schema": G5A_RECEIPT_SCHEMA,
        "stub_inclusive_artifact_count": 9_648,
        "target_count_root_sha256": target_count_root_sha256(target_rows),
        "vault_join_commitment_sha256": "8" * 64,
    }
    g5a_sha = sha256_bytes(receipt_bytes(g5a, expected_schema=G5A_RECEIPT_SCHEMA))
    vault_root = sha256_bytes(vault_bytes(vault_rows))
    prediction_root = "9" * 64
    grant: dict[str, object] = {
        "evaluator_code_sha256": evaluator_code,
        "g5a_receipt_sha256": g5a_sha,
        "invocation_count": 1,
        "join_commitment_sha256": "8" * 64,
        "prediction_root_sha256": prediction_root,
        "schema": CAPABILITY_GRANT_SCHEMA,
        "vault_root_sha256": vault_root,
    }
    metric_fixture = count_metric_fixture_receipt(
        evaluator_code_sha256=evaluator_code,
        fixture_input_bytes=b"fixture",
        expected_output_bytes=b"expected",
        attack_expected_global_fail_bytes=b"attacks",
    )
    metric_fixture_sha = sha256_bytes(json.dumps(dict(metric_fixture), sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
    evaluator = OneUseEvaluator()
    result = evaluator.run(
        grant=grant,
        g5a_receipt=g5a,
        evaluator_code_sha256=evaluator_code,
        population_identities=population,
        vault_rows=tuple(vault_rows),
        predictions=predictions,
        certified_development_targets=(evidence,),
        target_count_rows=target_rows,
        prediction_root_sha256=prediction_root,
        p2_metric_fixture_receipt=dict(metric_fixture),
        s0_metric_fixture_receipt_sha256=metric_fixture_sha,
        s0_evaluator_code_sha256=evaluator_code,
    )
    probe["k7_population_subset"] = {
        "accepted": True,
        "k7_status": result.k7_receipt["status"],
        "g5b_join_cardinality": result.g5b_receipt["join_cardinality"],
        "provided_prediction_artifacts": len(predictions),
        "contract_prediction_artifacts": 9_648,
        "provided_prediction_identities": len(selected),
        "joined_population_identities": len(population),
        "provided_certified_targets": 1,
    }

    cancellation_rejected = False
    cancellation_identity = ("f" * 64, 0)
    cancellation_vault = VaultRow(
        cancellation_identity[0], cancellation_identity[1], 40, 2, ((0, 10), (20, 30))
    )
    try:
        validate_certified_pulse_units(
            {cancellation_identity: cancellation_vault},
            {cancellation_identity: (1, 2)},
        )
    except ValueError:
        cancellation_rejected = True
    probe["split_merge_count_cancellation_rejected"] = cancellation_rejected

    source = TrustedSourceObject(
        motion=np.zeros((1, 320, 17, 3), dtype="<f4"),
        person_mask=np.ones(1, dtype=np.bool_),
        frame_mask=np.ones((1, 320), dtype=np.bool_),
        sampled_frame_indices=np.arange(320, dtype="<i8"),
        source_length=320,
        person_object_ids=("person",),
    )
    source.motion.flags.writeable = True
    probe["trusted_source_flag_reenabled"] = bool(source.motion.flags.writeable)
    print(json.dumps(probe, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
