from __future__ import annotations

import hashlib
import inspect
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from pams.config import load_config
from scripts.server import run_pams_v15_teacher_identifiability_probe as runner


def test_cli_surface_has_only_exact_train337_read_only_inputs() -> None:
    parsed = runner._parse_arguments(
        [
            "--encoder-checkpoint",
            "encoder.pt",
            "--encoder-progress",
            "encoder.jsonl",
            "--config",
            "v15.yaml",
            "--pose-cache-dir",
            "train337-pose",
            "--source-receipt",
            "source.receipt.json",
            "--output",
            "probe.json",
        ]
    )
    assert set(vars(parsed)) == {
        "encoder_checkpoint",
        "encoder_progress",
        "config",
        "pose_cache_dir",
        "source_receipt",
        "output",
        "device",
        "batch_size",
    }
    assert set(inspect.signature(runner.run_teacher_identifiability_probe).parameters) == {
        "encoder_checkpoint_path",
        "encoder_progress_path",
        "config_path",
        "pose_cache_dir",
        "source_receipt_path",
        "device",
        "batch_size",
    }
    source = inspect.getsource(runner._parse_arguments)
    for forbidden in (
        "--manifest",
        "--targets",
        "--action",
        "--count",
        "--dev",
        "--test",
        "--train",
        "--optimizer",
        "--sshead",
    ):
        assert f'"{forbidden}"' not in source
        assert f"'{forbidden}'" not in source


def test_probe_reuses_exact_v15_config_identity() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/experiments/pams_projected_teacher_v15.yaml"
    config = load_config(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    runner._v15._validate_exact_v15_config(config, config_sha256=digest)
    assert digest == runner._v15._EXPECTED_CONFIG_SHA256
    assert runner._EXPECTED_ENCODER_CHECKPOINT_SHA256.startswith("aa1750154")
    assert runner._EXPECTED_ENCODER_PROGRESS_SHA256.startswith("70226a5f")


def test_midpoint_anchors_are_nested_exact_subsets() -> None:
    valid = torch.ones(256, dtype=torch.bool)
    anchors = runner._midpoint_quantile_anchor_indices(valid)
    assert anchors.numel() == 64
    assert anchors.tolist()[:4] == [2, 6, 10, 14]
    positions32 = runner._budget_schedule_positions(64, 32)
    positions16 = runner._budget_schedule_positions(64, 16)
    assert positions32.tolist() == list(range(0, 64, 2))
    assert positions16.tolist() == list(range(0, 64, 4))
    assert anchors[positions16].tolist() == anchors[positions32[::2]].tolist()

    sparse = torch.zeros(20, dtype=torch.bool)
    sparse[[1, 5, 9]] = True
    assert runner._midpoint_quantile_anchor_indices(sparse).tolist() == [1, 5, 9]


def test_teacher_permutation_is_exact_length_stratified_and_pair_preserving() -> None:
    periods = torch.tensor([10.0, 20.0, 30.0, 40.0, 4.0])
    confidence = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.0])
    valid_counts = torch.tensor([200, 200, 200, 199, 200])
    video_ids = ("b", "a", "c", "singleton", "zero")
    perm_period, perm_conf, permutable, audit = runner._teacher_permutation(
        periods,
        confidence,
        valid_counts,
        video_ids,
    )
    # Exact-count=200 sorted IDs are a,b,c and receive b,c,a's paired values.
    assert perm_period.tolist()[:3] == [30.0, 10.0, 20.0]
    assert perm_conf.tolist()[:3] == pytest.approx([0.3, 0.1, 0.2])
    assert permutable.tolist() == [True, True, True, False, False]
    assert perm_period[3:].tolist() == [40.0, 4.0]
    assert audit["paired_period_confidence_multiset_preserved"] is True
    assert audit["singleton_positive_confidence_video_total"] == 1


def test_strict_center_never_searches_for_a_nearby_valid_frame() -> None:
    valid = torch.ones(12, dtype=torch.bool)
    valid[8] = False
    selected = runner._strict_center_indices(4.0, valid)
    # anchor 4 future target is exactly 8 and is invalid: no nearest-frame fallback.
    assert int(selected[4, 1]) == -1
    assert int(selected[5, 0]) == 1
    assert int(selected[5, 1]) == 9
    assert int(selected[2, 0]) == -1


def test_physical_groups_are_32_and_every_video_is_anchor_once() -> None:
    identifiers = tuple(f"v{index:03d}" for index in range(337))
    groups, group_by_anchor, audit = runner._physical_contrastive_groups(identifiers)
    assert len(groups) == 11
    assert all(len(group) == 32 for group in groups)
    assert groups[-1] == tuple(range(320, 337)) + tuple(range(15))
    assert group_by_anchor[:32] == (0,) * 32
    assert group_by_anchor[320:] == (10,) * 17
    assert audit["anchor_assignment_total"] == 337
    assert audit["final_group_anchor_video_total"] == 17
    assert audit["final_group_context_video_total"] == 15
    assert len(audit["grouping_sha256"]) == 64


def test_query_local_analytic_gradient_matches_autograd_and_center_deduplicates() -> None:
    video = F.normalize(torch.eye(4).repeat(3, 1), dim=-1)
    other = F.normalize(torch.roll(video, shifts=1, dims=0), dim=-1)
    valid = torch.ones(12, dtype=torch.bool)
    anchors = torch.tensor([4, 5, 6, 7])
    flat = torch.cat((video, other))
    owners = torch.tensor([0] * 12 + [1] * 12)
    generator = torch.Generator().manual_seed(91)
    bank = F.normalize(torch.randn(8, 4, generator=generator), dim=-1)
    clusters = torch.tensor([0, 1, 2, 3, 1, 2, 3, 1])
    temperature = 0.1

    denominator, denominator_grad, normalized_query = runner._denominator_and_row_gradients(
        video,
        valid,
        anchors,
        flat_embeddings=flat,
        flat_owner=owners,
        owner_index=0,
        bank_features=bank,
        bank_clusters=clusters,
        owner_cluster=0,
        temperature=temperature,
    )
    measurement, analytic, selections = runner._video_arm_measurement(
        video,
        valid,
        anchors,
        period=4.0,
        confidence=1.0,
        scales=(0.5,),
        temperature=temperature,
        denominator=denominator,
        denominator_row_gradients=denominator_grad,
        normalized_query=normalized_query,
        budget=64,
        mandatory_center=True,
        canonical_selections=None,
    )
    # Every selected argmax is already the strict center, so adding the center
    # must not double-count any of four anchors x two directions.
    assert measurement["eligible_cell_total"] == 8
    assert measurement["selected_correspondence_total"] == 8

    query = video.index_select(0, anchors).detach().clone().requires_grad_(True)
    q = F.normalize(query, dim=-1)
    within = q @ video.T / temperature
    within_allowed = valid.unsqueeze(0).expand_as(within).clone()
    within_allowed.scatter_(1, anchors.unsqueeze(1), False)
    within_partition = torch.logsumexp(
        within.masked_fill(~within_allowed, float("-inf")),
        dim=-1,
    )
    all_frames = q @ flat.T / temperature
    other_partition = torch.logsumexp(
        all_frames.masked_fill(~(owners != 0).unsqueeze(0), float("-inf")),
        dim=-1,
    )
    bank_logits = q @ bank.T / temperature
    hard = torch.topk(
        bank_logits.masked_fill(~(clusters != 0).unsqueeze(0), float("-inf")),
        k=4,
        dim=-1,
    ).values
    bank_partition = torch.logsumexp(hard, dim=-1)
    direct_denominator = torch.logsumexp(
        torch.stack((within_partition, other_partition, bank_partition), dim=-1),
        dim=-1,
    )
    strict = runner._strict_center_indices(4.0, valid)
    selected = selections["0.5"]
    losses = []
    normalized_video = F.normalize(video, dim=-1)
    for slot, anchor in enumerate(anchors.tolist()):
        for direction in range(2):
            positives = [int(selected[anchor, direction])]
            exact = int(strict[anchor, direction])
            if exact not in positives:
                positives.append(exact)
            positive = normalized_video[positives].mean(dim=0)
            losses.append(direct_denominator[slot] - (q[slot] * positive).sum() / temperature)
    direct_loss = torch.stack(losses).mean()
    direct_gradient = torch.autograd.grad(direct_loss, query)[0]
    assert float(measurement["loss"]) == pytest.approx(float(direct_loss.detach()), abs=1e-6)
    assert torch.allclose(analytic, direct_gradient, atol=2e-6, rtol=2e-5)


def test_geometry_opportunity_uses_current_inferred_window_formula() -> None:
    payload = runner._geometry_opportunity(
        torch.tensor([21.0, 5.0, 4.0]),
        torch.tensor([1.0, 1.0, 0.0]),
        ("a", "b", "zero"),
        scales=(0.5, 1.0, 1.5),
        time=256,
    )
    assert payload["positive_confidence_video_total"] == 2
    rows = payload["rows"]
    assert rows[0]["opportunity_by_scale"] == {
        "0.5": True,
        "1": True,
        "1.5": True,
    }
    assert rows[1]["opportunity_by_scale"]["0.5"] is False
    assert rows[1]["opportunity_by_scale"]["1.5"] is True
    assert all("video_id" not in row for row in rows)


def _classification_fixture(
    *,
    alias: bool = True,
    teacher_state: str = "teacher_blind",
    center: bool = True,
) -> dict[str, object]:
    if alias:
        s8 = {"median": 0.80, "bootstrap_lower_95": 0.70}
        r8 = {"median": 0.01, "bootstrap_upper_95": 0.04}
        w8 = 0.80
    else:
        s8 = {"median": 0.60, "bootstrap_lower_95": 0.55}
        r8 = {"median": 0.10, "bootstrap_upper_95": 0.12}
        w8 = 0.60
    if teacher_state == "teacher_blind":
        aperm = {"median": 0.01, "bootstrap_upper_95": 0.04}
        gperm = {"median": 0.97, "bootstrap_lower_95": 0.93}
    else:
        aperm = {"median": 0.06, "bootstrap_lower_95": 0.03}
        gperm = {"median": 0.85, "bootstrap_upper_95": 0.92}
    return {
        "w8_all_scale_opportunity_fraction": w8,
        "s8_canonical_selected_near_8_multiple": s8,
        "r8_constant8_relative_nll_advantage": r8,
        "aperm_teacher_permutation_relative_nll_advantage": aperm,
        "gperm_current_gradient_cosine": gperm,
        "gperm_mandatory_center_gradient_cosine": {
            "median": 0.80 if center else 0.96,
            "bootstrap_upper_95": 0.90 if center else 0.98,
        },
        "gperm_current_minus_mandatory_center": {
            "median": 0.10 if center else 0.01,
            "bootstrap_lower_95": 0.02 if center else -0.01,
        },
        "alias_confirmed": alias,
        "current_teacher_state": teacher_state,
        "mandatory_center_signal_viable": center,
        "paired_eligible_fraction": 0.95,
        "gradient_zero_fraction": 0.0,
    }


def test_convergence_requires_every_frozen_difference_and_classification() -> None:
    first = _classification_fixture()
    second = deepcopy(first)
    decision = runner._convergence_decision(first, second)
    assert decision["pass"] is True
    changed = deepcopy(second)
    changed["gperm_current_gradient_cosine"]["median"] = 0.90
    assert runner._convergence_decision(first, changed)["pass"] is False
    changed = deepcopy(second)
    changed["current_teacher_state"] = "identifiable"
    assert runner._convergence_decision(first, changed)["pass"] is False


def test_final_authorization_truth_table_never_authorizes_dev_or_sshead() -> None:
    operational = {"pass": True}
    convergence = {"pass": True}
    cases = (
        (True, "identifiable", True, True),
        (True, "teacher_blind", True, True),
        (False, "teacher_blind", True, True),
        (False, "identifiable", True, False),
        (True, "teacher_blind", False, False),
    )
    for alias, state, center, expected in cases:
        decision = runner._final_decision(
            operational=operational,
            convergence=convergence,
            primary={
                "alias_confirmed": alias,
                "current_teacher_state": state,
                "mandatory_center_signal_viable": center,
            },
        )
        assert decision["v16_short_continuation_authorized"] is expected
        assert decision["sshead_training_authorized"] is False
        assert decision["dev84_prediction_authorized"] is False
        assert decision["test105_evaluation_authorized"] is False


def test_artifact_and_receipt_are_exclusive_and_hash_bound(tmp_path: Path) -> None:
    digest = "a" * 64
    payload = {
        "status": "v16_short_continuation_authorized",
        "decision": {"v16_short_continuation_authorized": True},
        "inputs": {
            "encoder_checkpoint_sha256": digest,
            "encoder_progress_sha256": digest,
            "config_sha256": digest,
            "source_export_receipt_sha256": digest,
            "train337_pose_cache_set_sha256": digest,
            "checkpoint_algorithm_source_git_sha": "b" * 40,
            "gate_code_source_git_sha": "c" * 40,
            "code_files_sha256_commitment": digest,
        },
        "hardware_sha256": digest,
        "runtime_sha256": digest,
    }
    output = tmp_path / "probe.json"
    receipt_path, artifact_digest = runner._write_artifact_and_receipt(
        output,
        payload,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert artifact_digest == hashlib.sha256(output.read_bytes()).hexdigest()
    assert receipt["artifact_sha256"] == artifact_digest
    assert receipt["v16_short_continuation_authorized"] is True
    assert receipt["sshead_training_authorized"] is False
    assert receipt["dev84_prediction_authorized"] is False
    assert receipt["test105_evaluation_authorized"] is False
    with pytest.raises(FileExistsError, match="overwrite"):
        runner._write_artifact_and_receipt(output, payload)
