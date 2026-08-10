from __future__ import annotations

import json

import pytest

from pams.report import (
    AggregateReport,
    ResultRow,
    ResultTable,
    read_aggregate_json,
    render_markdown,
    write_aggregate_json,
)
from pams.verification import MetricPair, SeedResult, VerificationEvidence


def test_no_result_report_is_explicit_and_separates_tables() -> None:
    report = AggregateReport(
        title="UCFRep reproduction",
        evidence=VerificationEvidence(),
        rows=(
            ResultRow(
                method="PAMS-SSHead",
                protocol="ucfrep_526_fair",
                status="no_results",
                table=ResultTable.FAIR,
            ),
            ResultRow(
                method="PoseRAC-v1",
                protocol="ucfrep_pose_110_source",
                status="blocked_unimplemented",
                table=ResultTable.SOURCE,
                oracle=True,
            ),
        ),
    )
    markdown = render_markdown(report)
    assert "**Verification status:** `no_results`" in markdown
    assert "## Source-faithful / reported results" in markdown
    assert "## Fair comparison results" in markdown
    assert "No reproduction measurements are available" in markdown
    assert "0.000" not in markdown
    assert "—" in markdown


def test_result_row_rejects_partial_or_mislabelled_metrics() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        ResultRow(
            method="bad",
            protocol="fixture",
            status="measured",
            table=ResultTable.FAIR,
            nmae=0.2,
        )
    with pytest.raises(ValueError, match="cannot carry"):
        ResultRow(
            method="bad",
            protocol="fixture",
            status="no_results",
            table=ResultTable.FAIR,
            nmae=0.2,
            obo=0.7,
        )
    with pytest.raises(ValueError, match="oracle"):
        ResultRow(
            method="oracle",
            protocol="fixture",
            status="diagnostic",
            table=ResultTable.FAIR,
            nmae=0.1,
            obo=0.9,
            oracle=True,
        )


def test_aggregate_json_round_trip_and_refuses_overwrite(tmp_path) -> None:
    report = AggregateReport(
        title="Fixture results",
        evidence=VerificationEvidence(seed_results=(SeedResult(42, MetricPair(0.2, 0.7)),)),
        rows=(
            ResultRow(
                method="Source method",
                protocol="native_split",
                status="reported",
                table=ResultTable.SOURCE,
                nmae=0.3,
                obo=0.6,
                source_url="https://example.test/paper",
            ),
            ResultRow(
                method="Independent method",
                protocol="ucfrep_526_fair",
                status="measured",
                table=ResultTable.FAIR,
                nmae=0.2,
                obo=0.7,
                mae=1.2,
                rmse=1.5,
                sample_count=105,
            ),
        ),
        notes=("Fixture only.",),
    )
    target = tmp_path / "aggregate.json"
    write_aggregate_json(report, target)
    loaded = read_aggregate_json(target)
    assert loaded.to_dict() == report.to_dict()
    assert loaded.decision.status.value == "unverified"
    with pytest.raises(FileExistsError):
        write_aggregate_json(report, target)


def test_reader_rejects_tampered_gate_or_decision(tmp_path) -> None:
    report = AggregateReport(title="Fixture", evidence=VerificationEvidence())
    payload = report.to_dict()
    payload["verification_gate"]["maximum_mean_nmae"] = 9.0
    target = tmp_path / "tampered.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable repository gate"):
        read_aggregate_json(target)

    payload = report.to_dict()
    payload["verification_decision"]["status"] = "verified"
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        read_aggregate_json(target)
