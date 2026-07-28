"""Deterministic aggregate JSON and Markdown reporting.

Rows without measured or source-reported metrics remain explicit empty rows.
The renderer never substitutes paper values, proxies, zeroes, or values from a
different protocol.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pams.verification import (
    FROZEN_GATE,
    VerificationDecision,
    VerificationEvidence,
    VerificationStatus,
    evaluate_verification,
)


class ResultTable(str, Enum):
    """Mutually exclusive evidence tables."""

    SOURCE = "source"
    FAIR = "fair"


_RESULT_STATUSES = {"measured", "reported", "diagnostic"}
_NON_RESULT_STATUSES = {
    "no_results",
    "unverified",
    "blocked_unimplemented",
    "blocked_protocol",
    "blocked_resource",
    "smoke_only",
}


@dataclass(frozen=True, slots=True)
class ResultRow:
    """One protocol-labelled aggregate result or explicit unavailable row."""

    method: str
    protocol: str
    status: str
    table: ResultTable
    nmae: float | None = None
    obo: float | None = None
    mae: float | None = None
    rmse: float | None = None
    sample_count: int | None = None
    source_url: str | None = None
    oracle: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("method", "protocol", "status"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))

        if (self.nmae is None) != (self.obo is None):
            raise ValueError("nmae and obo must be supplied together")
        has_metrics = self.nmae is not None
        if has_metrics:
            assert self.nmae is not None
            assert self.obo is not None
            if not math.isfinite(self.nmae) or self.nmae < 0.0:
                raise ValueError("nmae must be finite and non-negative")
            if not math.isfinite(self.obo) or not 0.0 <= self.obo <= 1.0:
                raise ValueError("obo must be finite and in [0, 1]")
            for metric_name, metric_value in (("mae", self.mae), ("rmse", self.rmse)):
                if metric_value is not None and (
                    not math.isfinite(metric_value) or metric_value < 0.0
                ):
                    raise ValueError(f"{metric_name} must be finite and non-negative")
        elif self.mae is not None or self.rmse is not None or self.sample_count is not None:
            raise ValueError("auxiliary metrics require nmae and obo")

        if self.sample_count is not None and (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 1
        ):
            raise ValueError("sample_count must be a positive integer")
        if has_metrics and self.status in _NON_RESULT_STATUSES:
            raise ValueError(f"status {self.status!r} cannot carry result metrics")
        if not has_metrics and self.status in _RESULT_STATUSES:
            raise ValueError(f"status {self.status!r} requires result metrics")
        if self.oracle and self.table is ResultTable.FAIR:
            raise ValueError("oracle diagnostics cannot appear in a fair table")

    @property
    def has_metrics(self) -> bool:
        return self.nmae is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "protocol": self.protocol,
            "status": self.status,
            "table": self.table.value,
            "nmae": self.nmae,
            "obo": self.obo,
            "mae": self.mae,
            "rmse": self.rmse,
            "sample_count": self.sample_count,
            "source_url": self.source_url,
            "oracle": self.oracle,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResultRow:
        expected = {
            "method",
            "protocol",
            "status",
            "table",
            "nmae",
            "obo",
            "mae",
            "rmse",
            "sample_count",
            "source_url",
            "oracle",
            "notes",
        }
        _require_exact_keys(payload, expected, "result row")
        raw_notes = payload["notes"]
        if not isinstance(raw_notes, list):
            raise TypeError("row notes must be a list")
        raw_oracle = payload["oracle"]
        if not isinstance(raw_oracle, bool):
            raise TypeError("oracle must be a boolean")
        return cls(
            method=str(payload["method"]),
            protocol=str(payload["protocol"]),
            status=str(payload["status"]),
            table=ResultTable(str(payload["table"])),
            nmae=_optional_float(payload["nmae"]),
            obo=_optional_float(payload["obo"]),
            mae=_optional_float(payload["mae"]),
            rmse=_optional_float(payload["rmse"]),
            sample_count=_optional_int(payload["sample_count"]),
            source_url=(None if payload["source_url"] is None else str(payload["source_url"])),
            oracle=raw_oracle,
            notes=tuple(str(note) for note in raw_notes),
        )


@dataclass(frozen=True, slots=True)
class AggregateReport:
    """Serializable result bundle whose gate decision is always recomputed."""

    title: str
    evidence: VerificationEvidence
    rows: tuple[ResultRow, ...] = ()
    notes: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        title = str(self.title).strip()
        if not title:
            raise ValueError("title must be non-empty")
        if self.schema_version != 1:
            raise ValueError("unsupported aggregate report schema_version")
        rows = tuple(self.rows)
        identities = [(row.table, row.method, row.protocol) for row in rows]
        if len(identities) != len(set(identities)):
            raise ValueError("result rows must be unique by table, method, and protocol")
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))

    @property
    def decision(self) -> VerificationDecision:
        return evaluate_verification(self.evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "title": self.title,
            "verification_gate": FROZEN_GATE.to_dict(),
            "verification_evidence": self.evidence.to_dict(),
            "verification_decision": self.decision.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AggregateReport:
        expected = {
            "schema_version",
            "title",
            "verification_gate",
            "verification_evidence",
            "verification_decision",
            "rows",
            "notes",
        }
        _require_exact_keys(payload, expected, "aggregate report")
        if payload["verification_gate"] != FROZEN_GATE.to_dict():
            raise ValueError("aggregate JSON does not use the immutable repository gate")
        raw_rows = payload["rows"]
        raw_notes = payload["notes"]
        if not isinstance(raw_rows, list):
            raise TypeError("rows must be a list")
        if not isinstance(raw_notes, list):
            raise TypeError("notes must be a list")
        schema_version = _optional_int(payload["schema_version"])
        if schema_version is None:
            raise TypeError("schema_version cannot be null")
        evidence = VerificationEvidence.from_dict(
            _as_mapping(payload["verification_evidence"], "verification_evidence")
        )
        report = cls(
            schema_version=schema_version,
            title=str(payload["title"]),
            evidence=evidence,
            rows=tuple(ResultRow.from_dict(_as_mapping(row, "result row")) for row in raw_rows),
            notes=tuple(str(note) for note in raw_notes),
        )
        stored_decision = payload["verification_decision"]
        if stored_decision != report.decision.to_dict():
            raise ValueError("stored verification decision does not match recomputed evidence")
        return report


def read_aggregate_json(path: str | Path) -> AggregateReport:
    """Read, strictly validate, and recompute an aggregate result bundle."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return AggregateReport.from_dict(_as_mapping(payload, "aggregate report"))


def write_aggregate_json(
    report: AggregateReport,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write deterministic JSON, refusing overwrite by default."""

    content = json.dumps(
        report.to_dict(),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    return _atomic_write(content + "\n", path, overwrite=overwrite)


def render_markdown(report: AggregateReport) -> str:
    """Render a report with source and fair evidence in separate tables."""

    decision = report.decision
    lines = [
        f"# {report.title}",
        "",
        f"**Verification status:** `{decision.status.value}`",
        "",
        _status_explanation(decision.status),
        "",
        "## Frozen verification gate",
        "",
        "| Check | Result | Detail |",
        "|---|---:|---|",
    ]
    for check in decision.checks:
        result = "unavailable" if check.passed is None else ("pass" if check.passed else "fail")
        lines.append(f"| `{_escape(check.key)}` | {result} | {_escape(check.detail)} |")
    mean_nmae = "—" if decision.mean_metrics is None else f"{decision.mean_metrics.nmae:.3f}"
    mean_obo = "—" if decision.mean_metrics is None else f"{decision.mean_metrics.obo:.3f}"
    std_nmae = (
        "—"
        if decision.population_std_metrics is None
        else f"{decision.population_std_metrics.nmae:.3f}"
    )
    std_obo = (
        "—"
        if decision.population_std_metrics is None
        else f"{decision.population_std_metrics.obo:.3f}"
    )
    lines.extend(
        [
            "",
            f"Mean NMAE: **{mean_nmae}** (population SD **{std_nmae}**); "
            f"mean OBO: **{mean_obo}** (population SD **{std_obo}**); "
            f"individual passing seeds: **{decision.individual_passes}/3**.",
            "",
        ]
    )
    lines.extend(
        _render_result_section("Source-faithful / reported results", report, ResultTable.SOURCE)
    )
    lines.extend(_render_result_section("Fair comparison results", report, ResultTable.FAIR))

    if report.notes:
        lines.extend(["## Report notes", ""])
        lines.extend(f"- {_escape(note)}" for note in report.notes)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_markdown(
    report: AggregateReport,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write the deterministic Markdown rendering."""

    return _atomic_write(render_markdown(report), path, overwrite=overwrite)


def _render_result_section(
    heading: str,
    report: AggregateReport,
    table: ResultTable,
) -> list[str]:
    rows = sorted(
        (row for row in report.rows if row.table is table),
        key=lambda row: (row.method.lower(), row.protocol.lower()),
    )
    output = [f"## {heading}", ""]
    if not rows:
        return [
            *output,
            "No rows are registered for this table. No metric values are inferred.",
            "",
        ]
    output.extend(
        [
            "| Method | Protocol | Status | NMAE | OBO | Raw MAE | RMSE | N |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        method = _escape(row.method)
        if row.source_url:
            method = f"[{method}]({_escape_url(row.source_url)})"
        if row.oracle:
            method += " ⚠ ORACLE"
        output.append(
            "| "
            + " | ".join(
                [
                    method,
                    _escape(row.protocol),
                    f"`{_escape(row.status)}`",
                    _format_metric(row.nmae),
                    _format_metric(row.obo),
                    _format_metric(row.mae),
                    _format_metric(row.rmse),
                    "—" if row.sample_count is None else str(row.sample_count),
                ]
            )
            + " |"
        )
    output.extend(
        [
            "",
            "Empty cells (`—`) mean no admissible result exists; they are not zeroes or estimates.",
            "",
        ]
    )
    return output


def _status_explanation(status: VerificationStatus) -> str:
    explanations = {
        VerificationStatus.NO_RESULTS: (
            "No reproduction measurements are available. No verification claim is made."
        ),
        VerificationStatus.UNVERIFIED: (
            "Some measurements exist, but required seeds or ablations are missing."
        ),
        VerificationStatus.PARTIAL: (
            "All required evidence exists, but at least one preregistered gate failed."
        ),
        VerificationStatus.VERIFIED: ("All preregistered metric, seed, and ablation gates passed."),
    }
    return explanations[status]


def _atomic_write(content: str, path: str | Path, *, overwrite: bool) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite immutable report: {target}")
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=target.parent,
        delete=False,
        newline="\n",
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _format_metric(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _escape(value: str) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _escape_url(value: str) -> str:
    return str(value).replace("(", "%28").replace(")", "%29").replace(" ", "%20")


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("integer fields cannot be bool")
    integer = int(value)
    if integer != value:
        raise ValueError("integer field contains a non-integer value")
    return integer


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    name: str,
) -> None:
    supplied = set(payload)
    if supplied != expected:
        raise ValueError(f"{name} keys must be {sorted(expected)}, got {sorted(supplied)}")


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value
