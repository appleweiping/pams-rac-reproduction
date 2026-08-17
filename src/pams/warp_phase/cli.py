"""Isolated command group for the prospective WARP-PHASE pilot."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    name="warp-phase",
    help="Run fail-closed WARP-PHASE pilot gates and isolated jobs.",
    no_args_is_help=True,
    add_completion=False,
)


def _emit(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _finish_receipt(payload: dict[str, object]) -> None:
    """Emit a structured receipt and mirror terminal failure in the shell status."""

    _emit(payload)
    if payload.get("status") in {"BLOCKED", "FAIL"}:
        raise typer.Exit(code=1)


def _emit_frozen(payload: dict[str, object], receipt: Path) -> None:
    from pams.warp_phase.gates import write_canonical_json_exclusive

    payload["receipt_sha256"] = write_canonical_json_exclusive(receipt, payload)
    payload["receipt_path"] = str(receipt)
    _finish_receipt(payload)


@app.command("static-preflight")
def static_preflight(
    receipt: Annotated[
        Path | None,
        typer.Option(
            "--receipt",
            help="Optional new path for the immutable non-authorizing receipt.",
        ),
    ] = None,
) -> None:
    """Verify immutable schema/RNG/K4 bytes without authorizing training."""

    from pams.warp_phase.gates import static_gate_receipt, write_receipt_exclusive

    result = static_gate_receipt()
    payload = asdict(result)
    if receipt is not None:
        payload["receipt_sha256"] = write_receipt_exclusive(receipt, result)
        payload["receipt_path"] = str(receipt)
    _finish_receipt(payload)


@app.command("lock-training-environment")
def lock_training_environment(
    repository_root: Annotated[Path, typer.Option("--repository-root")],
    lock_path: Annotated[Path, typer.Option("--lock")],
) -> None:
    """Create the immutable consumption-only CPython/NumPy training lock."""

    from pams.warp_phase.gates import create_training_environment_lock

    digest = create_training_environment_lock(repository_root, lock_path)
    _emit({"lock_path": str(lock_path), "lock_sha256": digest, "status": "CREATED"})


@app.command("reserve-output-space")
def reserve_output_space(
    output_root: Annotated[Path, typer.Option("--output-root")],
    receipt: Annotated[Path, typer.Option("--receipt")],
) -> None:
    """Reserve one new empty training-output directory with an external receipt."""

    from pams.warp_phase.gates import create_output_space_receipt

    digest = create_output_space_receipt(output_root, receipt)
    _emit(
        {
            "output_root": str(output_root),
            "receipt_path": str(receipt),
            "receipt_sha256": digest,
            "status": "CREATED_EMPTY",
        }
    )


@app.command("gate0")
def gate0(
    repository_root: Annotated[Path, typer.Option("--repository-root")],
    train_source: Annotated[Path, typer.Option("--train-source")],
    val_source: Annotated[Path, typer.Option("--val-source")],
    run_root: Annotated[Path, typer.Option("--run-root")],
    receipt: Annotated[Path, typer.Option("--receipt")],
) -> None:
    """Verify approved source bytes and report the typed-pack blocker."""

    from pams.warp_phase.gates import gate0_pack_receipt, write_receipt_exclusive

    result = gate0_pack_receipt(
        repository_root=repository_root,
        train_source=train_source,
        val_source=val_source,
        run_root=run_root,
    )
    payload = asdict(result)
    payload["receipt_sha256"] = write_receipt_exclusive(receipt, result)
    payload["receipt_path"] = str(receipt)
    _finish_receipt(payload)


@app.command("gate1")
def gate1(
    repository_root: Annotated[Path, typer.Option("--repository-root")],
    receipt: Annotated[Path, typer.Option("--receipt")],
    training_lock: Annotated[Path | None, typer.Option("--training-lock")] = None,
    fixture_lock: Annotated[Path | None, typer.Option("--fixture-lock")] = None,
    fixture_root: Annotated[Path | None, typer.Option("--fixture-root")] = None,
    fixture_pack_receipt: Annotated[
        Path | None,
        typer.Option("--fixture-pack-receipt"),
    ] = None,
    selector_unit_fixture_root: Annotated[
        Path | None,
        typer.Option("--selector-unit-fixture-root"),
    ] = None,
    selector_unit_pack_receipt: Annotated[
        Path | None,
        typer.Option("--selector-unit-pack-receipt"),
    ] = None,
) -> None:
    """Verify immutable environment/fixture inputs and fail closed on gaps."""

    from pams.warp_phase.gates import gate1_fixture_receipt, write_receipt_exclusive

    result = gate1_fixture_receipt(
        repository_root=repository_root,
        training_lock_path=training_lock,
        fixture_lock_path=fixture_lock,
        fixture_root=fixture_root,
        pack_receipt_path=fixture_pack_receipt,
        selector_unit_fixture_root=selector_unit_fixture_root,
        selector_unit_pack_receipt_path=selector_unit_pack_receipt,
    )
    payload = asdict(result)
    payload["receipt_sha256"] = write_receipt_exclusive(receipt, result)
    payload["receipt_path"] = str(receipt)
    _finish_receipt(payload)


@app.command("gate2")
def gate2(
    receipt: Annotated[Path, typer.Option("--receipt")],
    fixture_root: Annotated[Path | None, typer.Option("--fixture-root")] = None,
    fixture_pack_receipt: Annotated[
        Path | None,
        typer.Option("--fixture-pack-receipt"),
    ] = None,
    features_root: Annotated[Path | None, typer.Option("--features-root")] = None,
    order_receipt: Annotated[Path | None, typer.Option("--order-receipt")] = None,
) -> None:
    """Verify immutable selector fixtures and report real-track blockers."""

    from pams.warp_phase.gates import gate2_selector_receipt, write_receipt_exclusive

    result = gate2_selector_receipt(
        fixture_root=fixture_root,
        pack_receipt_path=fixture_pack_receipt,
        features_root=features_root,
        order_receipt_path=order_receipt,
    )
    payload = asdict(result)
    payload["receipt_sha256"] = write_receipt_exclusive(receipt, result)
    payload["receipt_path"] = str(receipt)
    _finish_receipt(payload)


@app.command("gate3-cpu")
def gate3_cpu(
    receipt: Annotated[Path, typer.Option("--receipt")],
) -> None:
    """Run only the zero-job CPU sanity scaffold; never launch a GPU job."""

    from pams.warp_phase.gates import gate3_cpu_sanity_receipt, write_receipt_exclusive

    result = gate3_cpu_sanity_receipt()
    payload = asdict(result)
    payload["receipt_sha256"] = write_receipt_exclusive(receipt, result)
    payload["receipt_path"] = str(receipt)
    _finish_receipt(payload)


@app.command("train")
def train(
    features_root: Annotated[Path, typer.Option("--features-root")],
    output_root: Annotated[Path, typer.Option("--output-root")],
    environment_lock: Annotated[Path, typer.Option("--environment-lock")],
    output_space_receipt: Annotated[Path, typer.Option("--output-space-receipt")],
    receipt: Annotated[Path, typer.Option("--receipt")],
    repository_root: Annotated[Path, typer.Option("--repository-root")] = Path("."),
    split: Annotated[str, typer.Option("--split")] = "train",
) -> None:
    """Feature-only training interface; currently returns a blocked receipt."""

    from pams.warp_phase.gates import (
        verify_output_space_receipt,
        verify_training_environment_lock,
    )
    from pams.warp_phase.training import training_command_preflight

    environment_hash = verify_training_environment_lock(
        repository_root,
        environment_lock,
    )
    output_hash = verify_output_space_receipt(output_root, output_space_receipt)
    result = training_command_preflight(
        features_root=features_root,
        split=split,
        environment_lock_sha256=environment_hash,
        output_space_receipt_sha256=output_hash,
    )
    _emit_frozen(asdict(result), receipt)
