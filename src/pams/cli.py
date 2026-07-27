"""Command-line entry point for the auditable PAMS reproduction."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from pydantic import ValidationError

from pams.baselines import BaselineUnavailableError, create_baseline, list_baselines
from pams.config import PAMSConfig, load_config
from pams.data import (
    TrainingPoseDataset,
    UCFRepManifest,
    assert_split_disjoint,
    load_pose_cache,
    load_ucfrep_manifest,
    pose_cache_path,
    save_ucfrep_manifest,
    split_ucfrep_pose_train_dev,
    split_ucfrep_train_dev,
)
from pams.metrics import compute_count_metrics
from pams.synthetic import SyntheticSpec, generate_synthetic_sample, synthetic_stress_suite

app = typer.Typer(
    name="pams",
    help="Independent, auditable PAMS repetition-counting reproduction.",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(help="Validate frozen experiment configuration.", no_args_is_help=True)
data_app = typer.Typer(help="Validate and split UCFRep manifests.", no_args_is_help=True)
pose_app = typer.Typer(help="Extract and cache MediaPipe 33x3 pose.", no_args_is_help=True)
train_app = typer.Typer(help="Train the encoder or inferred SSHead.", no_args_is_help=True)
evaluate_app = typer.Typer(
    help="Evaluate a checkpoint on sealed cached poses.", no_args_is_help=True
)
baseline_app = typer.Typer(
    help="Inspect baselines or run the explicit spectral proxy.", no_args_is_help=True
)
report_app = typer.Typer(help="Compute frozen RAC metrics from predictions.", no_args_is_help=True)
synthetic_app = typer.Typer(help="Run deterministic synthetic diagnostics.", no_args_is_help=True)

app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(pose_app, name="pose")
app.add_typer(train_app, name="train")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(baseline_app, name="baseline")
app.add_typer(report_app, name="report")
app.add_typer(synthetic_app, name="synthetic")


def _emit(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _abort(message: str, *, code: int = 2) -> None:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code)


def _write_json_exclusive(path: Path, payload: Any, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _load_manifest(path: Path, *, protocol: str | None, exact: bool) -> UCFRepManifest:
    return load_ucfrep_manifest(path, protocol=protocol, validate_exact=exact)


def _validate_experiment_split(manifest: UCFRepManifest) -> None:
    """Validate official or frozen train/dev/test sizes without relabelling rows."""

    assert_split_disjoint(manifest.records)
    counts = manifest.split_counts
    permitted = {
        "ucfrep_526": (
            {"train": 421, "test": 105},
            {"train": 337, "dev": 84, "test": 105},
        ),
        "ucfrep_pose_110": (
            {"train": 89, "test": 21},
            {"train": 71, "dev": 18, "test": 21},
        ),
    }
    if counts not in permitted[manifest.protocol]:
        raise ValueError(
            f"{manifest.protocol} has invalid experiment split counts {counts}; "
            f"expected one of {permitted[manifest.protocol]}"
        )


def _cached_sequences(
    manifest: UCFRepManifest,
    *,
    split: str,
    cache_dir: Path,
    config_sha256: str,
) -> tuple[list[Any], list[int], list[str], list[str]]:
    records = manifest.records_for(split)
    if not records:
        raise ValueError(f"manifest contains no {split!r} records")
    sequences: list[Any] = []
    targets: list[int] = []
    identifiers: list[str] = []
    actions: list[str] = []
    for record in records:
        sequence, _ = load_pose_cache(
            pose_cache_path(cache_dir, record.video_id),
            expected_video_sha256=record.video_sha256,
            expected_config_sha256=config_sha256,
        )
        if sequence.video_id != record.video_id:
            raise ValueError(f"pose cache ID mismatch for {record.video_id!r}")
        sequences.append(sequence)
        targets.append(record.count)
        identifiers.append(record.video_id)
        actions.append(record.action)
    return sequences, targets, identifiers, actions


def _create_cli_run_manifest(
    *,
    output_dir: Path,
    command: list[str],
    config: PAMSConfig,
    dataset_sha256: str,
    notes: list[str] | None = None,
) -> Path:
    from pams.run_manifest import create_run_manifest, write_manifest_exclusive

    manifest = create_run_manifest(
        command=command,
        config_sha256=config.fingerprint,
        dataset_sha256=dataset_sha256,
        seed=config.seed,
        protocol=config.protocol,
        cwd=Path.cwd(),
        notes=notes,
    )
    return write_manifest_exclusive(manifest, output_dir / "manifests" / f"{manifest.run_id}.json")


def _device_or_none(device: str) -> str | None:
    return None if device.strip().lower() == "auto" else device


def _load_checkpoint_model(
    path: Path,
    config: PAMSConfig,
    *,
    expected_stage: Literal["encoder", "sshead"] | None = None,
    device: str = "auto",
) -> Any:
    """Load a training checkpoint without constructing optimizer state."""

    from pams.training import load_model_checkpoint

    return load_model_checkpoint(
        path,
        config,
        device=_device_or_none(device),
        expected_stage=expected_stage,
    )


@config_app.command("validate")
def config_validate(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Strictly validate YAML and print its canonical fingerprint."""

    try:
        config = load_config(path)
    except (OSError, ValueError, ValidationError) as exc:
        _abort(str(exc))
    _emit(
        {
            "valid": True,
            "schema_version": config.schema_version,
            "protocol": config.protocol,
            "seed": config.seed,
            "fingerprint": config.fingerprint,
        }
    )


@data_app.command("validate")
@data_app.command("validate-manifest", hidden=True)
def data_validate(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    protocol: Annotated[str | None, typer.Option("--protocol")] = None,
    exact: Annotated[bool, typer.Option("--exact/--allow-partial")] = True,
) -> None:
    """Validate a UCFRep CSV/JSON manifest and split invariants."""

    try:
        manifest = _load_manifest(path, protocol=protocol, exact=exact)
        if not exact:
            assert_split_disjoint(manifest.records)
    except (OSError, ValueError) as exc:
        _abort(str(exc))
    _emit(
        {
            "valid": True,
            "protocol": manifest.protocol,
            "records": len(manifest.records),
            "split_counts": manifest.split_counts,
            "fingerprint": manifest.fingerprint,
            "exact_official_split": exact,
        }
    )


@data_app.command("prepare-ucfrep")
def data_prepare_ucfrep(
    video_root: Annotated[
        Path,
        typer.Argument(
            file_okay=False,
            help="Directory containing the separately obtained UCF101 AVI files.",
        ),
    ],
    output: Annotated[Path, typer.Option("--output", "-o", dir_okay=False)],
    annotation_archive: Annotated[
        Path,
        typer.Option(
            "--annotations",
            dir_okay=False,
            help="Cached official annotation ZIP; downloaded if absent.",
        ),
    ] = Path("data/annotations/ucf526_annotations.zip"),
    split_dir: Annotated[
        Path,
        typer.Option(
            "--split-dir",
            file_okay=False,
            help="Destination for label-free 421/105 and 337/84 ID lists.",
        ),
    ] = Path("data/splits"),
    hash_existing_videos: Annotated[
        bool,
        typer.Option(
            "--hash-videos/--no-hash-videos",
            help="Hash every video currently present under VIDEO_ROOT.",
        ),
    ] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Build the canonical 421/105 manifest from official UCFRep annotations."""

    try:
        from pams.ucfrep import (
            OFFICIAL_ANNOTATION_SHA256,
            build_official_manifest,
            download_official_annotations,
            materialize_standard_split_ids,
        )

        if output.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing file: {output}")
        archive = download_official_annotations(annotation_archive)
        manifest = build_official_manifest(
            archive,
            video_root=video_root,
            hash_existing_videos=hash_existing_videos,
        )
        save_ucfrep_manifest(manifest, output)
        split_paths = materialize_standard_split_ids(
            manifest,
            split_dir,
            seed=2026,
            overwrite=overwrite,
        )
    except (OSError, ValueError) as exc:
        _abort(str(exc))
    _emit(
        {
            "output": str(output.resolve()),
            "protocol": manifest.protocol,
            "split_counts": manifest.split_counts,
            "manifest_fingerprint": manifest.fingerprint,
            "annotation_sha256": OFFICIAL_ANNOTATION_SHA256,
            "video_hashes_included": hash_existing_videos,
            "split_id_files": {name: str(path.resolve()) for name, path in split_paths.items()},
        }
    )


@data_app.command("validate-run")
def data_validate_run(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Validate an immutable run-manifest JSON document."""

    try:
        from pams.run_manifest import RunManifest

        manifest = RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        _abort(str(exc))
    _emit({"valid": True, "run_id": manifest.run_id, "fingerprint": manifest.fingerprint})


@data_app.command("split")
def data_split(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o", dir_okay=False)],
    seed: Annotated[int, typer.Option("--seed")] = 2026,
    protocol: Annotated[str | None, typer.Option("--protocol")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Create the frozen 337/84 or 71/18 development split."""

    try:
        manifest = _load_manifest(path, protocol=protocol, exact=True)
        if output.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing file: {output}")
        official_train = manifest.records_for("train")
        if manifest.protocol == "ucfrep_526":
            train, dev = split_ucfrep_train_dev(official_train, seed=seed)
        else:
            train, dev = split_ucfrep_pose_train_dev(official_train, seed=seed)
        split_manifest = UCFRepManifest(
            protocol=manifest.protocol,
            records=(*train, *dev, *manifest.records_for("test")),
        )
        _validate_experiment_split(split_manifest)
        save_ucfrep_manifest(split_manifest, output)
    except (OSError, ValueError) as exc:
        _abort(str(exc))
    _emit(
        {
            "output": str(output.resolve()),
            "protocol": split_manifest.protocol,
            "seed": seed,
            "split_counts": split_manifest.split_counts,
            "fingerprint": split_manifest.fingerprint,
        }
    )


@pose_app.command("extract")
def pose_extract(
    manifest_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    cache_dir: Annotated[Path, typer.Argument(file_okay=False)],
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/pams.yaml"
    ),
    split: Annotated[str, typer.Option("--split")] = "all",
    limit: Annotated[int | None, typer.Option("--limit", min=1)] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    keep_full_timeline: Annotated[bool, typer.Option("--keep-full-timeline")] = False,
    failure_ledger: Annotated[
        Path | None,
        typer.Option(
            "--failure-ledger",
            dir_okay=False,
            help="JSON failure ledger; defaults to CACHE_DIR/failures.json.",
        ),
    ] = None,
) -> None:
    """Extract pose caches for a manifest, loading MediaPipe only on demand."""

    try:
        from pams.pose import PoseExtractorConfig, extract_many_with_failures

        config = load_config(config_path)
        manifest = load_ucfrep_manifest(manifest_path, validate_exact=False)
        _validate_experiment_split(manifest)
        records = manifest.records if split == "all" else manifest.records_for(split)
        if limit is not None:
            records = records[:limit]
        if not records:
            raise ValueError(f"no records selected by split {split!r}")
        root = manifest_path.resolve().parent
        videos = tuple(
            (
                record.video_id,
                (
                    Path(record.video_path)
                    if Path(record.video_path).is_absolute()
                    else root / record.video_path
                ),
                record.video_sha256,
            )
            for record in records
        )
        summaries, failures = extract_many_with_failures(
            videos,
            cache_dir=cache_dir,
            config_sha256=config.fingerprint,
            extractor_config=PoseExtractorConfig(
                target_frames=config.data.frames,
                crop_to_longest_valid_span=not keep_full_timeline,
            ),
            overwrite=overwrite,
        )
        ledger_path = failure_ledger or cache_dir / "failures.json"
        _write_json_exclusive(
            ledger_path,
            {
                "schema_version": 1,
                "selected": len(records),
                "completed": len(summaries),
                "failed": len(failures),
                "failures": [failure.to_dict() for failure in failures],
            },
            overwrite=overwrite,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        _abort(str(exc))
    _emit(
        {
            "pose_model": "mediapipe-pose-0.10.14",
            "config_sha256": config.fingerprint,
            "selected": len(records),
            "completed": len(summaries),
            "failed": len(failures),
            "complete": not failures,
            "failure_ledger": str(ledger_path.resolve()),
            "caches": [summary.to_dict() for summary in summaries],
        }
    )
    if failures:
        raise typer.Exit(1)


def _training_inputs(
    manifest_path: Path,
    cache_dir: Path,
    config: PAMSConfig,
    *,
    include_dev: bool,
) -> tuple[UCFRepManifest, TrainingPoseDataset]:
    manifest = load_ucfrep_manifest(manifest_path, validate_exact=False)
    _validate_experiment_split(manifest)
    records = manifest.records_for("train")
    if include_dev:
        records = (*records, *manifest.records_for("dev"))
    dataset = TrainingPoseDataset(
        records,
        cache_dir=cache_dir,
        config_sha256=config.fingerprint,
        include_dev=include_dev,
    )
    return manifest, dataset


@train_app.command("encoder")
def train_encoder_command(
    manifest_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    cache_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/pams.yaml"
    ),
    device: Annotated[str, typer.Option("--device")] = "auto",
    epochs: Annotated[int | None, typer.Option("--epochs", min=1)] = None,
    microbatch_size: Annotated[int | None, typer.Option("--microbatch-size", min=1)] = None,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    include_dev: Annotated[bool, typer.Option("--include-dev")] = False,
) -> None:
    """Train the PAMS encoder from label-free cached poses."""

    try:
        from pams.training import train_encoder

        config = load_config(config_path)
        manifest, sequences = _training_inputs(
            manifest_path, cache_dir, config, include_dev=include_dev
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        _create_cli_run_manifest(
            output_dir=output_dir,
            command=list(sys.argv),
            config=config,
            dataset_sha256=manifest.fingerprint,
            notes=["count and action labels are not exposed to the training dataset"],
        )
        checkpoint_path = output_dir / "encoder.pt"
        result = train_encoder(
            sequences,
            config,
            device=_device_or_none(device),
            microbatch_size=microbatch_size,
            checkpoint_path=checkpoint_path,
            resume=resume,
            stop_after_epoch=epochs,
        )
    except ImportError as exc:
        _abort(f"encoder training module is unavailable: {exc}")
    except (OSError, RuntimeError, ValueError) as exc:
        _abort(str(exc))
    _emit(
        {
            "status": "completed",
            "stage": "encoder",
            "completed_epochs": result.completed_epochs,
            "checkpoint_path": (
                None if result.checkpoint_path is None else str(result.checkpoint_path.resolve())
            ),
            "cluster_assignments": dict(result.cluster_assignments),
            "history": [asdict(row) for row in result.history],
        }
    )


@train_app.command("sshead")
def train_sshead_command(
    encoder_checkpoint: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    manifest_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    cache_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/pams.yaml"
    ),
    device: Annotated[str, typer.Option("--device")] = "auto",
    epochs: Annotated[int | None, typer.Option("--epochs", min=1)] = None,
    microbatch_size: Annotated[int | None, typer.Option("--microbatch-size", min=1)] = None,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    include_dev: Annotated[bool, typer.Option("--include-dev")] = False,
) -> None:
    """Train the inferred self-supervised head with a frozen encoder."""

    try:
        from pams.training import train_sshead

        config = load_config(config_path)
        manifest, sequences = _training_inputs(
            manifest_path, cache_dir, config, include_dev=include_dev
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        _create_cli_run_manifest(
            output_dir=output_dir,
            command=list(sys.argv),
            config=config,
            dataset_sha256=manifest.fingerprint,
            notes=["PAMS-SSHead is an inferred reproduction completion, not author-disclosed"],
        )
        model = _load_checkpoint_model(
            encoder_checkpoint,
            config,
            expected_stage=None if resume else "encoder",
            device=device,
        )
        checkpoint_path = output_dir / "sshead.pt"
        result = train_sshead(
            sequences,
            config,
            model=model,
            device=_device_or_none(device),
            microbatch_size=microbatch_size,
            checkpoint_path=checkpoint_path,
            resume=resume,
            stop_after_epoch=epochs,
        )
    except ImportError as exc:
        _abort(f"SSHead training module is unavailable: {exc}")
    except (OSError, RuntimeError, ValueError) as exc:
        _abort(str(exc))
    _emit(
        {
            "status": "completed",
            "stage": "sshead",
            "inferred_reproduction_completion": True,
            "completed_epochs": result.completed_epochs,
            "checkpoint_path": (
                None if result.checkpoint_path is None else str(result.checkpoint_path.resolve())
            ),
            "history": [asdict(row) for row in result.history],
        }
    )


@evaluate_app.command("checkpoint")
def evaluate_checkpoint_command(
    checkpoint_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    manifest_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    cache_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/pams.yaml"
    ),
    split: Annotated[str, typer.Option("--split")] = "test",
    device: Annotated[str, typer.Option("--device")] = "auto",
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Evaluate one checkpoint; target labels are passed only to evaluation."""

    try:
        from pams.evaluation import evaluate_model

        config = load_config(config_path)
        manifest = load_ucfrep_manifest(manifest_path, validate_exact=False)
        _validate_experiment_split(manifest)
        sequences, targets, identifiers, actions = _cached_sequences(
            manifest,
            split=split,
            cache_dir=cache_dir,
            config_sha256=config.fingerprint,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        _create_cli_run_manifest(
            output_dir=output_dir,
            command=list(sys.argv),
            config=config,
            dataset_sha256=manifest.fingerprint,
            notes=[f"evaluation split: {split}"],
        )
        model = _load_checkpoint_model(
            checkpoint_path,
            config,
            device=device,
        )
        ground_truths = dict(zip(identifiers, targets, strict=True))
        action_mapping = dict(zip(identifiers, actions, strict=True))
        result = evaluate_model(
            model,
            sequences,
            config,
            ground_truths,
            actions=action_mapping,
            device=_device_or_none(device),
            bootstrap_seed=config.seed,
        )
        payload = result.to_dict(include_streams=True)
        _write_json_exclusive(
            output_dir / "evaluation.json",
            payload,
            overwrite=overwrite,
        )
    except ImportError as exc:
        _abort(f"checkpoint evaluation module is unavailable: {exc}")
    except (OSError, RuntimeError, ValueError) as exc:
        _abort(str(exc))
    _emit(payload)


@baseline_app.command("list")
def baseline_list(
    paper_only: Annotated[bool, typer.Option("--paper-only")] = False,
) -> None:
    """List registered status without equating registration with executability."""

    specs = list_baselines(include_references=not paper_only)
    _emit(
        {
            "baselines": [
                {
                    "key": spec.key,
                    "paper_name": spec.paper_name,
                    "status": spec.status.value,
                    "runnable": spec.runnable,
                    "implementation": spec.implementation.value,
                    "paper_baseline": spec.is_paper_baseline,
                    "blocked_reason": spec.blocked_reason,
                }
                for spec in specs
            ]
        }
    )


@baseline_app.command("predict")
def baseline_predict(
    name: Annotated[str, typer.Argument()],
    pose_cache: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path | None, typer.Option("--output", "-o", dir_okay=False)] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Run only the explicit diagnostic spectral proxy on one cached sequence."""

    try:
        if name.strip().lower().replace("_", "-") != "spectral-proxy":
            # Registry construction supplies the precise blocked reason and
            # cannot silently return the diagnostic proxy.
            create_baseline(name)
            raise ValueError(
                "the CLI permits prediction only for spectral-proxy until a "
                "paper baseline has passed its parity and protocol gates"
            )
        adapter = create_baseline("spectral-proxy")
        sequence, metadata = load_pose_cache(pose_cache)
        result = adapter.predict(sequence)
        payload = {
            "method": adapter.spec.key,
            "diagnostic_only": True,
            "eligible_for_paper_table": False,
            "video_id": sequence.video_id,
            "pose_cache_config_sha256": metadata.config_sha256,
            **result.to_dict(),
        }
        if output is not None:
            _write_json_exclusive(output, payload, overwrite=overwrite)
    except (BaselineUnavailableError, KeyError, OSError, ValueError) as exc:
        _abort(str(exc))
    _emit(payload)


def _read_predictions(path: Path) -> dict[str, float]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or any("video_id" not in row or "prediction" not in row for row in rows):
            raise ValueError("prediction CSV requires video_id and prediction columns")
        pairs = [(str(row["video_id"]), float(row["prediction"])) for row in rows]
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "predictions" in payload:
            payload = payload["predictions"]
        if isinstance(payload, dict):
            pairs = [(str(key), float(value)) for key, value in payload.items()]
        elif isinstance(payload, list):
            pairs = [
                (str(row["video_id"]), float(row["prediction"]))
                for row in payload
                if isinstance(row, dict)
            ]
            if len(pairs) != len(payload):
                raise ValueError("prediction JSON rows require video_id and prediction")
        else:
            raise ValueError("prediction JSON must be a mapping or list of rows")
    else:
        raise ValueError("predictions must be .json or .csv")
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("prediction video_id values must be unique")
    return result


@report_app.command("metrics")
def report_metrics(
    predictions_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    manifest_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path | None, typer.Option("--output", "-o", dir_okay=False)] = None,
    split: Annotated[str, typer.Option("--split")] = "test",
    bootstrap_samples: Annotated[int, typer.Option("--bootstrap-samples", min=0)] = 10_000,
    bootstrap_seed: Annotated[int, typer.Option("--bootstrap-seed")] = 2026,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Compute NMAE, raw MAE, RMSE, OBO, exact, and paired bootstrap CIs."""

    try:
        manifest = load_ucfrep_manifest(manifest_path, validate_exact=False)
        _validate_experiment_split(manifest)
        records = manifest.records_for(split)
        predictions = _read_predictions(predictions_path)
        expected = {record.video_id for record in records}
        supplied = set(predictions)
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            raise ValueError(
                f"prediction IDs do not match {split}: missing={missing[:5]}, extra={extra[:5]}"
            )
        report = compute_count_metrics(
            [predictions[record.video_id] for record in records],
            [record.count for record in records],
            video_ids=[record.video_id for record in records],
            actions=[record.action for record in records],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        payload = {
            "protocol": manifest.protocol,
            "split": split,
            "dataset_sha256": manifest.fingerprint,
            **report.to_dict(),
        }
        if output is not None:
            _write_json_exclusive(output, payload, overwrite=overwrite)
    except (OSError, ValueError) as exc:
        _abort(str(exc))
    _emit(payload)


def _parse_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("counts must be comma-separated integers") from exc
    if not counts or len(set(counts)) != len(counts):
        raise ValueError("counts must contain unique comma-separated integers")
    if any(count < 2 or count > 40 for count in counts):
        raise ValueError("synthetic counts must be in [2, 40]")
    return counts


@synthetic_app.command("evaluate")
def synthetic_evaluate(
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/pams.yaml"
    ),
    counts: Annotated[str, typer.Option("--counts")] = "2,4,8,16",
    stress: Annotated[bool, typer.Option("--stress")] = False,
    bootstrap_samples: Annotated[int, typer.Option("--bootstrap-samples", min=0)] = 0,
) -> None:
    """Evaluate the explicitly named spectral proxy on deterministic synthetic pose."""

    try:
        from pams.baselines.proxy import SpectralProxyAdapter

        config = load_config(config_path)
        selected_counts = _parse_counts(counts)
        adapter = SpectralProxyAdapter(
            minimum_period=config.period.minimum,
            maximum_period=config.period.maximum,
        )
        samples = [
            generate_synthetic_sample(
                SyntheticSpec(
                    video_id=f"count-{count:02d}",
                    frames=config.data.frames,
                    count=float(count),
                    seed=config.seed + count,
                )
            )
            for count in selected_counts
        ]
        if stress:
            samples.extend(
                synthetic_stress_suite(
                    count=8.0,
                    frames=config.data.frames,
                    seed=config.seed,
                ).values()
            )
        predictions = [adapter.predict(sample.sequence).count for sample in samples]
        targets = [int(round(sample.target_count)) for sample in samples]
        identifiers = [sample.sequence.video_id for sample in samples]
        report = compute_count_metrics(
            predictions,
            targets,
            video_ids=identifiers,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=config.seed,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        _abort(str(exc))
    _emit(
        {
            "method": "spectral-proxy",
            "diagnostic_only": True,
            "eligible_for_paper_table": False,
            "config_sha256": config.fingerprint,
            **report.to_dict(),
        }
    )


if __name__ == "__main__":
    app()
