#!/usr/bin/env python3
"""Regression tests proving that the ICASSP evidence gates fail closed."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from claim_scan import scan_text  # noqa: E402
from evidence_common import sha256  # noqa: E402
from generate_evidence_tex import run as generate  # noqa: E402
from validate_method_manifest import validate as validate_method  # noqa: E402
from validate_pdf_4plus1 import validate as validate_pdf  # noqa: E402
from validate_readiness import validate as validate_readiness  # noqa: E402
from validate_results_manifest import validate_manifest  # noqa: E402

PAPER_DIR = SCRIPT_DIR.parent
ROOT = PAPER_DIR.parent


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _path_hash(root: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256(path)}


def _eligible_bundle(root: Path) -> tuple[Path, Path, Path]:
    paper = root / "paper"
    evidence = paper / "evidence"
    evidence.mkdir(parents=True)
    shutil.copy2(PAPER_DIR / "evidence/results_manifest.schema.json", evidence / "results_manifest.schema.json")
    values = root / "artifacts/values.json"
    _write(
        values,
        json.dumps(
            {
                "protocol": "oracle tracks",
                "result": "0.123",
                "claim": "The frozen comparison changes AvgMAE by 0.123 under the declared protocol.",
                "abstract": "Under the frozen declared protocol, the synchronized model produces the audited result reported in Table 1.",
            }
        ),
    )
    config = root / "artifacts/config.yaml"
    log = root / "artifacts/run.log"
    predictions = root / "artifacts/predictions.json"
    _write(config, "seed: frozen\n")
    _write(log, "completed\n")
    _write(predictions, "[]\n")
    audit_records: dict[str, dict[str, str]] = {}
    for name in ("experiment_audit", "result_to_claim", "paper_claim_audit"):
        audit = root / f"artifacts/{name}.json"
        _write(audit, json.dumps({"status": "PASS"}) + "\n")
        audit_records[name] = {"status": "PASS", **_path_hash(root, audit)}
    source = {
        "id": "values",
        "path": values.relative_to(root).as_posix(),
        "format": "json",
        "sha256": sha256(values),
        "frozen": True,
    }
    manifest = {
        "$schema": "results_manifest.schema.json",
        "schema_version": "1.0",
        "document_type": "icassp_results_manifest",
        "status": "frozen-eligible",
        "artifact_id": "unit-test-artifact",
        "artifact_family": "unit-test-synchronized-multirep",
        "frozen": True,
        "table_eligible": True,
        "method_commit": "1" * 40,
        "dataset": {"name": "MultiRep", "release": "unit", "split": "test", "checksum": "sha256:" + "2" * 64},
        "protocol": {
            "track_modes": ["oracle"],
            "native_contextual_separated": True,
            "common_track_frontend": "unit",
            "evaluation_script": "artifacts/eval.py",
            "evaluation_type": "real_gt",
        },
        "seeds": [1, 2, 3],
        "statistics": {
            "mean": True,
            "sample_standard_deviation": True,
            "paired_video_cluster_bootstrap_95ci": True,
            "bootstrap_unit": "paired_video_cluster",
        },
        "hardware": {"device": "unit"},
        "configuration_files": [_path_hash(root, config)],
        "run_logs": [_path_hash(root, log)],
        "prediction_artifacts": [_path_hash(root, predictions)],
        "source_artifacts": [source],
        "metrics": {name: {} for name in ("Period-mAP", "Period-AP50", "Period-AP75", "AvgMAE", "AvgOBO")},
        "audits": audit_records,
        "paper_bindings": {
            "protocol_fields": [{"key": "trackmode", "source_id": "values", "locator": {"json_pointer": "/protocol"}}],
            "results": [{"key": "mainavgmae", "source_id": "values", "locator": {"json_pointer": "/result"}}],
            "claims": [{"key": "mainclaim", "source_id": "values", "locator": {"json_pointer": "/claim"}}],
            "abstract_result": {"source_id": "values", "locator": {"json_pointer": "/abstract"}},
        },
        "eligibility": {
            "new_synchronized_multirep_family": True,
            "method_commit_bound": True,
            "dataset_release_bound": True,
            "protocol_bound": True,
            "three_or_more_seeds": True,
            "uncertainty_complete": True,
            "raw_artifacts_hash_bound": True,
            "audits_pass": True,
        },
        "blocking_reasons": [],
    }
    manifest_path = evidence / "results_manifest.json"
    _write(manifest_path, json.dumps(manifest, indent=2) + "\n")
    return manifest_path, values, paper / "generated/evidence_values.tex"


class EvidenceGateTests(unittest.TestCase):
    def test_current_draft_is_provisional_but_successful(self) -> None:
        report, code = validate_readiness(PAPER_DIR, "draft", None)
        self.assertEqual(code, 0)
        self.assertEqual(report["status"], "PROVISIONAL")
        self.assertFalse(report["submission_ready"])

    def test_current_submission_fails_closed(self) -> None:
        report, code = validate_readiness(PAPER_DIR, "submission", None)
        self.assertNotEqual(code, 0)
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["submission_ready"])
        self.assertTrue({"method", "results", "authors", "venue", "pdf"}.issubset(set(report["failed_gates"])))

    def test_sync_required_method_is_draft_only(self) -> None:
        draft, draft_code = validate_method(PAPER_DIR, "draft")
        submission, submission_code = validate_method(PAPER_DIR, "submission")
        self.assertEqual(draft_code, 0)
        self.assertEqual(draft["status"], "PROVISIONAL")
        self.assertNotEqual(submission_code, 0)
        self.assertEqual(submission["status"], "FAIL")

    def test_forbidden_claim_and_placeholder_are_detected(self) -> None:
        forbidden, placeholders = scan_text("Our method is SOTA. Results are PENDING.", "unit")
        self.assertEqual([item["token"] for item in forbidden], ["sota"])
        self.assertEqual([item["token"] for item in placeholders], ["PENDING"])

    def test_generation_is_hash_bound_and_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, values, output = _eligible_bundle(root)
            validation, code = validate_manifest(manifest, root, "submission")
            self.assertEqual(code, 0, validation)
            generated, code = generate(manifest, output, root, "submission")
            self.assertEqual(code, 0, generated)
            self.assertTrue(output.is_file())
            verified, code = generate(manifest, output, root, "submission", verify_only=True)
            self.assertEqual(code, 0, verified)
            _write(values, values.read_text(encoding="utf-8") + " ")
            tampered, code = validate_manifest(manifest, root, "submission")
            self.assertNotEqual(code, 0)
            self.assertEqual(tampered["status"], "FAIL")
            refused, code = generate(manifest, output, root, "submission", verify_only=True)
            self.assertNotEqual(code, 0)
            self.assertEqual(refused["status"], "FAIL")

    def test_page5_references_only_passes_and_technical_text_fails(self) -> None:
        from reportlab.pdfgen import canvas

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            good_path = directory / "good.pdf"
            document = canvas.Canvas(str(good_path))
            for index in range(5):
                lines = ["REFERENCES", "[1] Unit reference."] if index == 4 else ["TECHNICAL CONTENT"]
                for offset, line in enumerate(lines):
                    document.drawString(72, 720 - 14 * offset, line)
                document.showPage()
            document.save()
            good, code = validate_pdf(good_path, PAPER_DIR, "submission")
            self.assertEqual(code, 0, good)

            bad_path = directory / "bad.pdf"
            document = canvas.Canvas(str(bad_path))
            for index in range(5):
                lines = ["RESULTS", "A new technical claim.", "REFERENCES", "[1] Unit reference."] if index == 4 else ["TECHNICAL CONTENT"]
                for offset, line in enumerate(lines):
                    document.drawString(72, 720 - 14 * offset, line)
                document.showPage()
            document.save()
            bad, code = validate_pdf(bad_path, PAPER_DIR, "submission")
            self.assertNotEqual(code, 0)
            self.assertEqual(bad["status"], "FAIL")

            continuation_path = directory / "continuation.pdf"
            document = canvas.Canvas(str(continuation_path))
            for index in range(5):
                if index == 3:
                    lines = ["REFERENCES", "[1] A long reference starts on page four and"]
                elif index == 4:
                    lines = ["continues on page five.", "[2] A second unit reference."]
                else:
                    lines = ["TECHNICAL CONTENT"]
                for offset, line in enumerate(lines):
                    document.drawString(72, 720 - 14 * offset, line)
                document.showPage()
            document.save()
            continuation, code = validate_pdf(continuation_path, PAPER_DIR, "submission")
            self.assertEqual(code, 0, continuation)
            self.assertTrue(continuation["page5"]["reference_continuation"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
