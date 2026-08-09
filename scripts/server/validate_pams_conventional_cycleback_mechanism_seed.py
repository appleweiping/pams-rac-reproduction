#!/usr/bin/env python3
"""Fresh-process semantic validation for a published mechanism seed bundle."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pams.conventional_cycleback.config import parse_conventional_cycleback_config
from pams.conventional_cycleback.fullcontext_trainer import (
    FullContextTrainerContract,
    MechanismSeedPredecessorLineage,
    build_fullcontext_adamw,
    model_state_sha256,
    optimizer_state_sha256,
    validate_fullcontext_optimizer,
    validate_mechanism_seed_checkpoint_payload,
)
from pams.conventional_cycleback.probe import validate_mechanism_probe_contract
from pams.conventional_cycleback.runtime import build_encoder


def _load_json(path: Path) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(value, Mapping):
        raise ValueError("mechanism output must be a JSON object")
    return value


def validate_seed(output: Path, checkpoint: Path, config_path: Path) -> None:
    output_payload = _load_json(output)
    config = parse_conventional_cycleback_config(config_path.read_bytes())
    validate_mechanism_probe_contract(output_payload, config)
    contract = FullContextTrainerContract.from_candidate_config(config)
    encoded = checkpoint.read_bytes()
    checkpoint_record = output_payload.get("mechanism_seed_checkpoint")
    if not isinstance(checkpoint_record, Mapping):
        raise ValueError("mechanism output lacks checkpoint binding")
    if (
        checkpoint_record.get("sha256") != hashlib.sha256(encoded).hexdigest()
        or checkpoint_record.get("bytes") != len(encoded)
    ):
        raise ValueError("checkpoint bytes differ from mechanism JSON")
    value = torch.load(
        io.BytesIO(encoded),
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    if not isinstance(value, Mapping):
        raise ValueError("mechanism checkpoint root must be a mapping")
    raw_lineage = value.get("lineage")
    if not isinstance(raw_lineage, Mapping):
        raise ValueError("mechanism checkpoint lineage is missing")
    lineage = MechanismSeedPredecessorLineage(**dict(raw_lineage))
    validate_mechanism_seed_checkpoint_payload(value, contract, lineage)
    model = build_encoder(config).train()
    optimizer = build_fullcontext_adamw(model, contract)
    model.load_state_dict(value["model_state"], strict=True)
    optimizer.load_state_dict(value["optimizer_state"])
    validate_fullcontext_optimizer(optimizer, contract)
    if (
        model_state_sha256(model) != value["model_state_sha256"]
        or optimizer_state_sha256(optimizer)
        != value["optimizer_state_sha256"]
    ):
        raise ValueError("checkpoint cannot restore its declared model/optimizer state")
    sampler = value["sampler_state"]
    expected = {
        "captured_boundary_model_state_sha256": value["model_state_sha256"],
        "captured_boundary_optimizer_state_sha256": value["optimizer_state_sha256"],
        "captured_boundary_rng_state_sha256": value["rng_state_sha256"],
        "captured_boundary_backend_state_sha256": value["backend_state_sha256"],
        "captured_sampler_state_sha256": value["sampler_state_sha256"],
        "captured_view_state_sha256": value["next_view_seed_state_sha256"],
        "captured_consumed_video_batch_chain_sha256": sampler[
            "consumed_video_batch_chain_sha256"
        ],
        "captured_consumed_pair_row_chain_sha256": sampler[
            "consumed_pair_row_chain_sha256"
        ],
        "trainer_contract_fingerprint": value["trainer_contract_fingerprint"],
        "representation_contract_sha256": value[
            "representation_contract_sha256"
        ],
        "seed_predecessor_lineage_fingerprint": value["lineage_fingerprint"],
    }
    if any(checkpoint_record.get(key) != item for key, item in expected.items()):
        raise ValueError("checkpoint semantic state differs from mechanism JSON")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    validate_seed(arguments.output, arguments.checkpoint, arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
