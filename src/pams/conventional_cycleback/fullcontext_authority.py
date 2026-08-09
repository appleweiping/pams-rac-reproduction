"""Non-activated authority boundary for full-context continuation.

Activation values intentionally remain empty.  This scaffold cannot read a
caller path, deserialize a checkpoint, or launch a container until an
independent integration commit freezes the producer and consumer source trees,
immutable image, representation outcome, mechanism outcome/checkpoint, and
stage-specific launch authorization bytes.
"""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

from pams.conventional_cycleback.fullcontext_trainer import TrainingPhase

_APPROVED_FULLCONTEXT_SOURCE_REVISION = ""
_APPROVED_FULLCONTEXT_SOURCE_TREE_SHA256 = ""
_APPROVED_FULLCONTEXT_CONTAINER_IMAGE_ID = ""

_APPROVED_INTEGRATION_OUTCOME_LOCATOR = ""
_APPROVED_INTEGRATION_OUTCOME_SHA256 = ""
_APPROVED_INTEGRATION_OUTCOME_BYTES = 0

_APPROVED_MECHANISM_OUTCOME_LOCATOR = ""
_APPROVED_MECHANISM_OUTCOME_SHA256 = ""
_APPROVED_MECHANISM_OUTCOME_BYTES = 0
_APPROVED_MECHANISM_RUN_RECEIPT_SHA256 = ""
_APPROVED_MECHANISM_RUN_RECEIPT_BYTES = 0
_APPROVED_MECHANISM_SEED_CHECKPOINT_SHA256 = ""
_APPROVED_MECHANISM_SEED_CHECKPOINT_BYTES = 0
_APPROVED_MECHANISM_MODEL_STATE_SHA256 = ""
_APPROVED_MECHANISM_OPTIMIZER_STATE_SHA256 = ""
_APPROVED_MECHANISM_RNG_STATE_SHA256 = ""
_APPROVED_MECHANISM_BACKEND_STATE_SHA256 = ""
_APPROVED_MECHANISM_SAMPLER_STATE_SHA256 = ""
_APPROVED_MECHANISM_VIEW_STATE_SHA256 = ""
_APPROVED_MECHANISM_CONSUMED_VIDEO_BATCH_CHAIN_SHA256 = ""
_APPROVED_MECHANISM_CONSUMED_PAIR_ROW_CHAIN_SHA256 = ""

_APPROVED_EPOCH11_LAUNCH_AUTHORIZATION_LOCATOR = ""
_APPROVED_EPOCH11_LAUNCH_AUTHORIZATION_SHA256 = ""
_APPROVED_EPOCH11_LAUNCH_AUTHORIZATION_BYTES = 0
_APPROVED_EPOCH11_THRESHOLDS_SHA256 = ""

_APPROVED_EPOCH150_LAUNCH_AUTHORIZATION_LOCATOR = ""
_APPROVED_EPOCH150_LAUNCH_AUTHORIZATION_SHA256 = ""
_APPROVED_EPOCH150_LAUNCH_AUTHORIZATION_BYTES = 0
_APPROVED_EPOCH11_GATE_OUTCOME_SHA256 = ""
_APPROVED_EPOCH11_GATE_OUTCOME_BYTES = 0
_APPROVED_EPOCH11_PREFIX_CHECKPOINT_SHA256 = ""
_APPROVED_EPOCH11_PREFIX_CHECKPOINT_BYTES = 0


def authority_scaffold_status() -> dict[str, bool]:
    """Report the current executable authority surface without touching disk."""

    return {
        "mechanism_seed_checkpoint_available": False,
        "epoch11_train337_continuation_authorized": False,
        "epoch150_train337_continuation_authorized": False,
        "development_evaluation_authorized": False,
        "sealed_evaluation_authorized": False,
        "readout_authorized": False,
    }


def _activation_values() -> tuple[object, ...]:
    return (
        _APPROVED_FULLCONTEXT_SOURCE_REVISION,
        _APPROVED_FULLCONTEXT_SOURCE_TREE_SHA256,
        _APPROVED_FULLCONTEXT_CONTAINER_IMAGE_ID,
        _APPROVED_INTEGRATION_OUTCOME_LOCATOR,
        _APPROVED_INTEGRATION_OUTCOME_SHA256,
        _APPROVED_INTEGRATION_OUTCOME_BYTES,
        _APPROVED_MECHANISM_OUTCOME_LOCATOR,
        _APPROVED_MECHANISM_OUTCOME_SHA256,
        _APPROVED_MECHANISM_OUTCOME_BYTES,
        _APPROVED_MECHANISM_RUN_RECEIPT_SHA256,
        _APPROVED_MECHANISM_RUN_RECEIPT_BYTES,
        _APPROVED_MECHANISM_SEED_CHECKPOINT_SHA256,
        _APPROVED_MECHANISM_SEED_CHECKPOINT_BYTES,
        _APPROVED_MECHANISM_MODEL_STATE_SHA256,
        _APPROVED_MECHANISM_OPTIMIZER_STATE_SHA256,
        _APPROVED_MECHANISM_RNG_STATE_SHA256,
        _APPROVED_MECHANISM_BACKEND_STATE_SHA256,
        _APPROVED_MECHANISM_SAMPLER_STATE_SHA256,
        _APPROVED_MECHANISM_VIEW_STATE_SHA256,
        _APPROVED_MECHANISM_CONSUMED_VIDEO_BATCH_CHAIN_SHA256,
        _APPROVED_MECHANISM_CONSUMED_PAIR_ROW_CHAIN_SHA256,
        _APPROVED_EPOCH11_LAUNCH_AUTHORIZATION_LOCATOR,
        _APPROVED_EPOCH11_LAUNCH_AUTHORIZATION_SHA256,
        _APPROVED_EPOCH11_LAUNCH_AUTHORIZATION_BYTES,
        _APPROVED_EPOCH11_THRESHOLDS_SHA256,
        _APPROVED_EPOCH150_LAUNCH_AUTHORIZATION_LOCATOR,
        _APPROVED_EPOCH150_LAUNCH_AUTHORIZATION_SHA256,
        _APPROVED_EPOCH150_LAUNCH_AUTHORIZATION_BYTES,
        _APPROVED_EPOCH11_GATE_OUTCOME_SHA256,
        _APPROVED_EPOCH11_GATE_OUTCOME_BYTES,
        _APPROVED_EPOCH11_PREFIX_CHECKPOINT_SHA256,
        _APPROVED_EPOCH11_PREFIX_CHECKPOINT_BYTES,
    )


def require_fullcontext_activation() -> NoReturn:
    """Fail before registry reads, caller checkouts, imports, or deserialization."""

    if not all(_activation_values()):
        raise RuntimeError(
            "full-context continuation is disabled pending an independent "
            "representation/mechanism/training activation commit"
        )
    raise RuntimeError("full-context activation validator is not implemented")


def validate_fullcontext_predecessor_authority(
    phase: TrainingPhase,
    *,
    integration_outcome_root: str | Path,
    mechanism_outcome_root: str | Path,
    launch_authorization_root: str | Path,
    epoch11_gate_root: str | Path | None = None,
) -> NoReturn:
    """Nonlaunchable placeholder whose first operation is the trust anchor check."""

    require_fullcontext_activation()
    del (
        phase,
        integration_outcome_root,
        mechanism_outcome_root,
        launch_authorization_root,
        epoch11_gate_root,
    )
    raise RuntimeError("unreachable full-context authority scaffold")
