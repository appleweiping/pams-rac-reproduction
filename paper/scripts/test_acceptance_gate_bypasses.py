#!/usr/bin/env python3
"""Deterministic regression tests for the round-3 fail-closed bypasses."""

from __future__ import annotations

import hashlib
import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from aggregate_citation_audit import exact_context, reviewed_contexts_match  # noqa: E402
from claim_scan import scan_text  # noqa: E402
from generate_evidence_tex import render_payload  # noqa: E402
from validate_method_contract import (  # noqa: E402
    check_collaborator_binding,
    collaborator_manifest_index,
)
from validate_readiness import build_check, reviewer_health_check, sha256, template_check  # noqa: E402
from validate_result_manifest import (  # noqa: E402
    binding_text,
    resolve_json_pointer,
    verify_protocol_source,
)


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class AcceptanceGateBypassTests(unittest.TestCase):
    def test_primacy_variants_are_blocked(self) -> None:
        variants = (
            "We introduce the first multi-person framework for counting.",
            "This is the first pose-driven repetition-counting architecture.",
            "We are the first to route local tempo.",
            "A first-ever asynchronous counter is proposed.",
        )
        for index, sentence in enumerate(variants):
            violations, _, _ = scan_text(
                f"synthetic-{index}.tex", sentence, digest_bytes(sentence.encode()), {}, set()
            )
            self.assertIn("primacy", {item["rule"] for item in violations}, sentence)

    def test_exact_context_cannot_be_auto_supported(self) -> None:
        text = "Claim one.\\cite{k}\ncontinued.\n\nNext paragraph."
        start = text.index("\\cite")
        context = exact_context(text, start, start + len("\\cite{k}"))
        row = {
            "file": "sections/x.tex",
            "line": 1,
            "context_sha256": "sha256:" + digest_bytes(context.encode()),
        }
        self.assertFalse(reviewed_contexts_match([row], None))
        reviewed = [{**row, "verdict": "SUPPORTS"}]
        self.assertTrue(reviewed_contexts_match([row], reviewed))
        reviewed[0]["context_sha256"] = "sha256:" + "0" * 64
        self.assertFalse(reviewed_contexts_match([row], reviewed))

    def test_protocol_pointer_value_and_source_digest(self) -> None:
        target = {"protocol": {"mode": "oracle_tracks", "nested/key": {"~field": 3}}}
        self.assertEqual(resolve_json_pointer(target, "/protocol/mode"), "oracle_tracks")
        self.assertEqual(resolve_json_pointer(target, "/protocol/nested~1key/~0field"), 3)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "protocol.json"
            source.write_text(json.dumps({"mode": "oracle_tracks"}), encoding="utf-8")
            field = {
                "key": "TrackMode",
                "source": {
                    "path": "protocol.json",
                    "sha256": "sha256:" + sha256(source),
                },
            }
            errors: list[str] = []
            verify_protocol_source(field, root, binding_text("oracle_tracks"), errors)
            self.assertEqual(errors, [])
            field["source"]["sha256"] = "sha256:" + "0" * 64
            verify_protocol_source(field, root, "oracle_tracks", errors)
            self.assertTrue(any("digest mismatch" in item for item in errors))

    def test_external_receipt_is_cli_only_and_acyclic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outer = Path(temporary)
            bundle = outer / "bundle"
            bundle.mkdir()
            manifest = bundle / "result.json"
            package = bundle / "MANIFEST.sha256"
            manifest_data = {
                "created_at": "2026-08-12T00:00:00Z",
                "manifest_integrity": {
                    "algorithm": "sha256",
                    "result_manifest_path": "result.json",
                    "package_manifest_path": "MANIFEST.sha256",
                    "external_delivery_receipt_id": "receipt-test",
                    "package_manifest_excludes": ["MANIFEST.sha256"],
                },
            }
            manifest.write_text(json.dumps(manifest_data, sort_keys=True), encoding="utf-8")
            package.write_text(f"{sha256(manifest)}  result.json\n", encoding="utf-8")
            receipt = outer / "receipt.json"
            receipt.write_text(
                json.dumps(
                    {
                        "schema_version": "2.0",
                        "algorithm": "sha256",
                        "receipt_id": "receipt-test",
                        "delivery_commit": "a" * 40,
                        "result_manifest": {"path": "result.json", "sha256": "sha256:" + sha256(manifest)},
                        "package_manifest": {"path": "MANIFEST.sha256", "sha256": "sha256:" + sha256(package)},
                    }
                ),
                encoding="utf-8",
            )
            base = [
                sys.executable,
                str(SCRIPTS / "evidence_precheck.py"),
                str(manifest),
                "--bundle-root",
                str(bundle),
            ]
            missing = subprocess.run(base, capture_output=True, text=True)
            self.assertNotEqual(missing.returncode, 0)
            passed = subprocess.run(base + ["--external-receipt", str(receipt)], capture_output=True, text=True)
            self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)

    def test_collaborator_confirmed_binding_requires_manifest_and_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outer = Path(temporary)
            repository = outer / "collaborator"
            repository.mkdir()
            subprocess.run(["git", "init", "-q", str(repository)], check=True)
            subprocess.run(["git", "-C", str(repository), "config", "user.email", "gate-test.invalid"], check=True)
            subprocess.run(["git", "-C", str(repository), "config", "user.name", "Gate Test"], check=True)
            source = repository / "model.py"
            source.write_text("def model():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(repository), "add", "model.py"], check=True)
            subprocess.run(["git", "-C", str(repository), "commit", "-q", "-m", "fixture"], check=True)
            commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
            manifest = outer / "collaborator-manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "repository_root": str(repository.resolve()),
                        "commit": commit,
                        "artifacts": [{"path": "model.py", "sha256": "sha256:" + sha256(source)}],
                    }
                ),
                encoding="utf-8",
            )
            freeze = {
                "repository_root": str(repository.resolve()),
                "commit": commit,
                "manifest_path": str(manifest.resolve()),
                "manifest_sha256": "sha256:" + sha256(manifest),
            }
            errors: list[str] = []
            root, _, index = collaborator_manifest_index(outer, freeze, errors)
            self.assertEqual(errors, [])
            record = {
                "path": "model.py",
                "symbol": "model",
                "commit": commit,
                "sha256": "sha256:" + sha256(source),
                "verdict": "PASS",
            }
            check_collaborator_binding(root, index, record, commit, "fixture", errors)
            self.assertEqual(errors, [])
            record["sha256"] = "sha256:" + "0" * 64
            check_collaborator_binding(root, index, record, commit, "fixture-bad", errors)
            self.assertTrue(any("absent from collaborator manifest" in item for item in errors))

    def test_build_report_requires_exact_source_set_and_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paper = root / "paper"
            (paper / "sections").mkdir(parents=True)
            (paper / "figures").mkdir()
            for rel in ("main.tex", "preamble.tex", "math_commands.tex", "references.bib", "sections/a.tex"):
                path = paper / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(rel, encoding="utf-8")
            for rel in ("figures/task_comparison.pdf", "figures/framework.pdf", "main.pdf"):
                (paper / rel).write_bytes(b"%PDF-fixture")
            source_paths = [
                paper / "main.tex", paper / "preamble.tex", paper / "math_commands.tex",
                paper / "references.bib", paper / "sections/a.tex",
                paper / "figures/task_comparison.pdf", paper / "figures/framework.pdf",
            ]
            records = [{"path": p.relative_to(paper).as_posix(), "sha256": sha256(p)} for p in source_paths]
            payload = "\n".join(f"{r['path']}\t{r['sha256']}" for r in records).encode()
            report = {
                "verdict": "PASS",
                "pdf": {"sha256": sha256(paper / "main.pdf")},
                "source_files": records,
                "source_bundle_sha256": digest_bytes(payload),
                "source_freeze_sha256": "f" * 64,
                "checks": {
                    "conclusion_within_eight_pages": True,
                    "letter_page_size": True,
                    "final_log_warning_count": 0,
                    "unembedded_font_count": 0,
                    "duplicate_labels": [],
                    "undefined_citation_keys": [],
                    "unresolved_question_marks": False,
                    "anonymous_author_visible": True,
                },
            }
            report_path = paper / "build.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(build_check(root, report_path, paper / "main.pdf", "f" * 64)["status"], "PASS")
            report["source_files"].append({"path": "extra.tex", "sha256": "0" * 64})
            report_path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(build_check(root, report_path, paper / "main.pdf", "f" * 64)["status"], "FAIL")

    def test_template_and_claude_health_require_digest_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paper = root / "paper"
            vendor = paper / "vendor/cvpr2027"
            vendor.mkdir(parents=True)
            style = vendor / "cvpr.sty"
            bst = vendor / "ieeenat_fullname.bst"
            style.write_text("style", encoding="utf-8")
            bst.write_text("bst", encoding="utf-8")
            (paper / "main.tex").write_text(
                "\\usepackage[review]{cvpr}\n\\bibliographystyle{vendor/cvpr2027/ieeenat_fullname}\n",
                encoding="utf-8",
            )
            (paper / "TEMPLATE_PROVENANCE.md").write_text(
                "# Official CVPR 2027 template\n\n- Current status: official CVPR 2027 final\n"
                "- Tag: `CVPR2027-v1`\n- Commit: `" + "a" * 40 + "`\n\n"
                "| File | SHA-256 |\n|---|---|\n"
                f"| `vendor/cvpr2027/cvpr.sty` | `{sha256(style)}` |\n"
                f"| `vendor/cvpr2027/ieeenat_fullname.bst` | `{sha256(bst)}` |\n",
                encoding="utf-8",
            )
            self.assertEqual(template_check(root)["status"], "PASS")
            aris = paper / ".aris"
            aris.mkdir()
            trace = aris / "claude-trace.jsonl"
            probe = aris / "claude-probe.json"
            trace.write_text("{\"event\":\"response\"}\n", encoding="utf-8")
            probe.write_text("{\"ok\":true}\n", encoding="utf-8")
            health = {
                "verdict": "PASS",
                "provider": "Anthropic Claude",
                "paper_freeze": "b" * 64,
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "trace": {"path": "paper/.aris/claude-trace.jsonl", "sha256": sha256(trace)},
                "probe": {"path": "paper/.aris/claude-probe.json", "sha256": sha256(probe), "status": "PASS"},
            }
            health_path = aris / "reviewer-health.json"
            health_path.write_text(json.dumps(health), encoding="utf-8")
            self.assertEqual(reviewer_health_check(root, "b" * 64, None)["status"], "PASS")
            health["probe"]["sha256"] = "0" * 64
            health_path.write_text(json.dumps(health), encoding="utf-8")
            self.assertEqual(reviewer_health_check(root, "b" * 64, None)["status"], "FAIL")

    def test_evidence_tex_render_is_sorted_and_exact(self) -> None:
        data = {
            "artifact_id": "fixture",
            "paper_bindings": {
                "protocol_fields": [{"key": "B", "value": "2"}, {"key": "A", "value": "1"}],
                "results": [{"key": "R2", "formatted": "2"}, {"key": "R1", "formatted": "1"}],
                "claims": [{"key": "C2", "text": "two"}, {"key": "C1", "text": "one"}],
                "abstract_result": {"sentence": "Audited result."},
            },
        }
        first = render_payload(data)
        second = render_payload(json.loads(json.dumps(data)))
        self.assertEqual(first.encode(), second.encode())
        self.assertLess(first.index("{A}{1}"), first.index("{B}{2}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path)
    options = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AcceptanceGateBypassTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if options.json_output:
        report = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "verdict": "PASS" if result.wasSuccessful() else "FAIL",
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped),
            "test_script_sha256": sha256(Path(__file__)),
        }
        options.json_output.parent.mkdir(parents=True, exist_ok=True)
        options.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
