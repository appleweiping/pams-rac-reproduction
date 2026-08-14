#!/usr/bin/env python3
"""Redacted, PDF-bound anonymity scan for submission sources and binaries."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
from pathlib import Path

from PIL import Image
from pypdf import PdfReader


TEXT_SUFFIXES = {
    ".tex", ".bib", ".md", ".json", ".yaml", ".yml", ".py", ".ps1",
    ".drawio", ".svg", ".txt", ".csv", ".jsonl", ".log"
}
EMAIL = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
SOURCE_AUTHOR_FILES = {
    "paper/main.tex", "paper/submission_wrapper.tex",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tracked_and_untracked(root: Path) -> list[Path]:
    command = ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    raw = subprocess.check_output(command)
    return [root / item.decode("utf-8") for item in raw.split(b"\0") if item]


def submission_facing(label: str) -> bool:
    if label in SOURCE_AUTHOR_FILES:
        return True
    if label.startswith("paper/vendor/") or label.startswith("paper/rendered/") or label.startswith("paper/.aris/"):
        return False
    if label.startswith("paper/"):
        return True
    return label in {
        "NARRATIVE_REPORT.md", "CLAIMS_EVIDENCE_MATRIX.md", "CLAIMS_FROM_RESULTS.md",
        "PAPER_PLAN.md", "PAPER_ACCEPTANCE_CONTRACT.md", "PAPER_IMPROVEMENT_LOG.md",
        "IMPLEMENTATION_SYNC_REPORT_ZH.md", "ARIS_LOCK.md",
    }


def private_hits_bytes(data: bytes, secrets: list[str]) -> int:
    hits = 0
    lower = data.lower()
    for secret in secrets:
        for encoding in ("utf-8", "utf-16-le", "utf-16-be"):
            encoded = secret.encode(encoding, errors="ignore").lower()
            if encoded:
                hits += lower.count(encoded)
    return hits


def metadata_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        metadata = PdfReader(str(path)).metadata or {}
        return "\n".join(f"{key}={value}" for key, value in metadata.items())
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        with Image.open(path) as image:
            parts = [f"{key}={value}" for key, value in image.info.items() if isinstance(value, (str, int, float))]
            try:
                exif = image.getexif()
                parts.extend(f"exif:{key}={value}" for key, value in exif.items())
            except Exception:
                pass
        return "\n".join(parts)
    return ""


def infer_pdf(root: Path, pdf_text: Path) -> Path:
    stem = pdf_text.stem
    if stem.endswith("_submission"):
        return root / "paper" / f"{stem}.pdf"
    return root / "paper/main.pdf"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--secret-file", type=Path, required=True)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--pdf-text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    secret_file = args.secret_file.resolve()
    pdf_text = args.pdf_text.resolve()
    pdf = args.pdf.resolve() if args.pdf else infer_pdf(root, pdf_text).resolve()
    if root == secret_file or root in secret_file.parents:
        raise SystemExit("Secret list must remain outside the repository")
    secrets = [
        line.strip().casefold()
        for line in secret_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not secrets:
        raise SystemExit("Private secret list is empty")

    findings: list[dict] = []
    submission_hashes: list[dict] = []
    scope_digest_lines: list[str] = []
    advisory_email_hits = 0
    output = args.output.resolve()
    paths = [path.resolve() for path in tracked_and_untracked(root) if path.resolve() != output]
    for path in sorted(set(paths)):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        label = path.relative_to(root).as_posix()
        data = path.read_text(encoding="utf-8", errors="replace")
        folded = data.casefold()
        secret_hits = sum(folded.count(secret) for secret in secrets)
        email_hits = len(EMAIL.findall(data))
        record = {"file": label, "sha256": sha256(path)}
        scope_digest_lines.append(f"{label}\t{record['sha256']}")
        if submission_facing(label):
            submission_hashes.append(record)
        else:
            advisory_email_hits += email_hits
        if secret_hits or (submission_facing(label) and email_hits):
            findings.append({
                **record,
                "surface": "text",
                "private_secret_hit_count": secret_hits,
                "email_pattern_hit_count": email_hits,
            })

    freeze = json.loads((root / "paper/evidence/freeze_inventory.json").read_text(encoding="utf-8"))
    release_binaries: list[Path] = []
    for pattern in freeze.get("anonymity_release_globs", []):
        release_binaries.extend(path for path in root.glob(pattern) if path.is_file())
    binary_metadata_records: list[dict] = []
    for path in sorted(set(release_binaries)):
        label = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        meta = metadata_text(path)
        content_text = ""
        if path.suffix.lower() == ".pdf":
            try:
                content_text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
            except Exception:
                content_text = ""
        secret_binary = private_hits_bytes(raw, secrets)
        secret_meta = sum(meta.casefold().count(secret) for secret in secrets)
        secret_content = sum(content_text.casefold().count(secret) for secret in secrets)
        email_meta = len(EMAIL.findall(meta))
        email_content = len(EMAIL.findall(content_text))
        record = {
            "file": label,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "metadata_sha256": hashlib.sha256(meta.encode("utf-8")).hexdigest(),
            "extracted_content_sha256": hashlib.sha256(content_text.encode("utf-8")).hexdigest() if content_text else None,
        }
        binary_metadata_records.append(record)
        if secret_binary or secret_meta or secret_content or email_meta or email_content:
            findings.append({
                **record,
                "surface": "binary_or_metadata",
                "private_secret_binary_hit_count": secret_binary,
                "private_secret_metadata_hit_count": secret_meta,
                "private_secret_extracted_content_hit_count": secret_content,
                "email_pattern_metadata_hit_count": email_meta,
                "email_pattern_extracted_content_hit_count": email_content,
            })

    pdf_binding: dict[str, object]
    if not pdf.is_file() or not pdf_text.is_file():
        pdf_binding = {"status": "FAIL", "pdf_present": pdf.is_file(), "extracted_text_present": pdf_text.is_file()}
        findings.append({"file": "main.pdf", "surface": "pdf_binding", "private_secret_hit_count": 0, "email_pattern_hit_count": 0})
        extracted = ""
    else:
        supplied = pdf_text.read_bytes()
        try:
            regenerated = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"])
            exact = supplied.replace(b"\r\n", b"\n") == regenerated.replace(b"\r\n", b"\n")
        except Exception:
            exact = False
        pdf_binding = {
            "status": "PASS" if exact else "FAIL",
            "pdf_sha256": sha256(pdf),
            "extracted_text_sha256": sha256(pdf_text),
            "extractor": "pdftotext -layout",
            "exact_extraction_match": exact,
        }
        if not exact:
            findings.append({"file": "main.pdf", "surface": "pdf_binding", "private_secret_hit_count": 0, "email_pattern_hit_count": 0})
        extracted = supplied.decode("utf-8", errors="replace")
        folded = extracted.casefold()
        secret_hits = sum(folded.count(secret) for secret in secrets)
        email_hits = len(EMAIL.findall(extracted))
        if secret_hits or email_hits:
            findings.append({
                "file": "main.pdf:extracted-text",
                "sha256": sha256(pdf_text),
                "surface": "pdf_text",
                "private_secret_hit_count": secret_hits,
                "email_pattern_hit_count": email_hits,
            })

    author_sources: list[tuple[str, str]] = []
    for rel in SOURCE_AUTHOR_FILES:
        path = root / rel
        if path.is_file():
            author_sources.append((rel, path.read_text(encoding="utf-8", errors="replace")))
    declarations: list[tuple[str, str]] = []
    for rel, text in author_sources:
        declarations.extend((rel, body.strip()) for body in re.findall(r"\\author\{([^}]*)\}", text, re.DOTALL))
    exact_source_author = len(declarations) == 1 and declarations[0][1] == "Anonymous CVPR submission"
    visible_author_count = len(re.findall(r"Anonymous\s+CVPR\s+submission", extracted, re.I))
    exact_pdf_author = visible_author_count == 1
    if not exact_source_author or not exact_pdf_author:
        findings.append({
            "file": "anonymous-author-declaration",
            "surface": "author_contract",
            "source_declaration_count": len(declarations),
            "pdf_visible_declaration_count": visible_author_count,
            "private_secret_hit_count": 0,
            "email_pattern_hit_count": 0,
        })

    report = {
        "schema_version": 2,
        "verdict": "PASS" if not findings else "FAIL",
        "scope": "submission-facing worktree text, release-selected logs/binary metadata, and text re-extracted from the exact PDF; reachable Git history remains a separate pre-push gate",
        "secret_values_redacted": True,
        "anonymous_author_contract": {
            "source_declaration_count": len(declarations),
            "source_exactly_anonymous": exact_source_author,
            "pdf_visible_declaration_count": visible_author_count,
            "pdf_exactly_one_anonymous_declaration": exact_pdf_author,
        },
        "pdf_binding": pdf_binding,
        "scanned_file_count": len(scope_digest_lines),
        "scanned_scope_sha256": hashlib.sha256("\n".join(sorted(scope_digest_lines)).encode("utf-8")).hexdigest(),
        "submission_file_hashes": submission_hashes,
        "release_binary_metadata": binary_metadata_records,
        "non_submission_generic_email_hits_advisory": advisory_email_hits,
        "findings": findings,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"ANONYMITY_SCAN={report['verdict']}; files={len(scope_digest_lines)}; findings={len(findings)}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
