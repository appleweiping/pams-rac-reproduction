"""Prepare a write-once train337-only native-candidate launch authorization."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from scripts.server import run_pams_native_terminal_readout_gate as gate


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gate-specification", type=Path, required=True)
    parser.add_argument("--pose-snapshot", type=Path, required=True)
    parser.add_argument("--candidate-id", choices=gate._CANDIDATE_ORDER, required=True)
    parser.add_argument(
        "--prior-rejection-artifact", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--prior-rejection-receipt", type=Path, action="append", default=[]
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    payload = gate.prepare_candidate_launch_authorization(
        source_receipt_path=arguments.source_receipt,
        config_path=arguments.config,
        gate_specification_path=arguments.gate_specification,
        pose_snapshot_path=arguments.pose_snapshot,
        candidate_id=arguments.candidate_id,
        prior_rejection_artifact_paths=arguments.prior_rejection_artifact,
        prior_rejection_receipt_paths=arguments.prior_rejection_receipt,
        authorization_runner_path=Path(__file__),
    )
    receipt, digest = gate.write_candidate_launch_authorization(
        arguments.output, payload
    )
    print(
        json.dumps(
            {
                "artifact_sha256": digest,
                "candidate_id": arguments.candidate_id,
                "candidate_training_authorized": True,
                "dev84_pose_or_scoring_authorized": False,
                "output": str(arguments.output),
                "receipt": str(receipt),
                "test105_evaluation_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
