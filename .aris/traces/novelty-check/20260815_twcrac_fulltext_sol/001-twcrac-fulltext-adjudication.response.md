{
  "schema_version": "1.0",
  "audit_id": "20260815_twcrac_fulltext_sol",
  "audit_type": "primary_source_full_text_novelty_adjudication",
  "as_of_date": "2026-08-15",
  "reviewer": {
    "reviewer_model": "gpt-5.6-sol",
    "reviewer_family": "openai",
    "independence": "same-family",
    "acceptance_status": "provisional",
    "fresh_zero_context": true,
    "role": "primary-source TWCRAC novelty adjudicator"
  },
  "inputs": [
    {
      "path": "idea-stage/NOVELTY_CHECK.json",
      "sha256": "79749893547c834d876801c287bfa2f048b4797c838f7f6d46155318258ed3e5"
    },
    {
      "path": "refine-logs/round-1-refinement.md",
      "sha256": "6840a8e8cec3d8fa6d7c371e14a37a3ad3fdd1be120902bf67f911a247a47b75"
    }
  ],
  "source": {
    "work": "TWCRAC: Leveraging Temporal Cycle Consistency and Dynamic Thresholding for Robust Repetitive Action Counting",
    "authors_from_ssrn_record": [
      "Jun Li",
      "Shizhao Gao",
      "Tianwen Hu",
      "Yinhui Xie",
      "Zhonghua Liu",
      "Qiming Li"
    ],
    "ssrn_abstract_id": "6630216",
    "doi": "10.2139/ssrn.6630216",
    "official_record_url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6630216",
    "official_pdf_url": "https://papers.ssrn.com/sol3/Delivery.cfm/929069d2-a143-4b2f-b90f-e36010c363c9-MECA.pdf?abstractid=6630216&mirid=1",
    "allowed_source_class": "official SSRN record/PDF or author primary copy only",
    "record_metadata": {
      "posted_date": "2026-04-22",
      "page_count_claimed_by_record": 31,
      "page_count_independently_verified": false
    }
  },
  "downloaded_document": {
    "status": "not_obtained",
    "sha256": null,
    "sha256_reason": "No PDF response body was received from an allowed primary endpoint.",
    "byte_length": null,
    "page_count": null,
    "page_count_verified": false,
    "full_text_coverage": {
      "complete": false,
      "coverage": "abstract-level metadata only",
      "pages_obtained": 0,
      "pages_inspected": [],
      "method_complete": false,
      "loss_complete": false,
      "training_complete": false,
      "evaluation_complete": false
    }
  },
  "retrieval_attempts": [
    {
      "attempt_id": "A01",
      "target": "official SSRN record",
      "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6630216",
      "method": "primary-record resolution through web retrieval",
      "result": "The official record metadata and abstract were discoverable, including its claim of 31 pages, but no complete PDF body was exposed."
    },
    {
      "attempt_id": "A02",
      "target": "official SSRN PDF",
      "url": "https://papers.ssrn.com/sol3/Delivery.cfm/929069d2-a143-4b2f-b90f-e36010c363c9-MECA.pdf?abstractid=6630216&mirid=1",
      "method": "direct PDF fetch through the web retrieval service",
      "result": "HTTP 403 Forbidden; zero PDF pages obtained."
    },
    {
      "attempt_id": "A03",
      "target": "official SSRN PDF",
      "url": "https://papers.ssrn.com/sol3/Delivery.cfm/929069d2-a143-4b2f-b90f-e36010c363c9-MECA.pdf?abstractid=6630216&mirid=1",
      "method": "direct curl HEAD with browser user agent and official-record referrer",
      "result": "Connection timed out after 20.013 seconds; no headers or body obtained."
    },
    {
      "attempt_id": "A04",
      "target": "official SSRN record over resolved IPv4",
      "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6630216",
      "method": "direct curl IPv4 GET",
      "result": "DNS resolved papers.ssrn.com to 66.220.149.18, then the connection timed out after 10.014 seconds with HTTP 000 and zero bytes."
    },
    {
      "attempt_id": "A05",
      "target": "official SSRN record and its download control",
      "url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6630216",
      "method": "non-disruptive in-app browser navigation",
      "result": "Navigation timed out before a DOM snapshot or downloadable asset was returned; no file was created."
    },
    {
      "attempt_id": "A06",
      "target": "existing browser session",
      "method": "read-only Chrome CDP safety probe",
      "result": "No debug endpoint was available. Setup would have terminated 59 active Chrome processes, so it was not performed."
    },
    {
      "attempt_id": "A07",
      "target": "author primary copy",
      "method": "exact-title and author/title searches, including arXiv, GitHub, edu.cn, and ac.cn domains",
      "result": "No author-controlled full text was located. Results resolved only to SSRN, unrelated works, or a disallowed ResearchGate request page."
    },
    {
      "attempt_id": "A08",
      "target": "workspace and operating-system temporary directory",
      "method": "filename scan for TWCRAC, 6630216, the SSRN delivery identifier, and PDFs",
      "result": "No candidate TWCRAC PDF was present."
    }
  ],
  "residual_claim_under_review": {
    "candidate": "warp-equivariant-private-phase-flow",
    "narrow_claim": "A discrete warp-integral equivariance objective for signed phase increments in supplied-track multi-person repetitive action counting.",
    "mechanism_test": "For each warped target interval, match its signed phase increment to a stop-gradient oriented overlap integral of the clean branch's piecewise-constant signed phase rate under a known target-to-source time map.",
    "express_non_contributions": [
      "phase learning",
      "temporal equivariance",
      "local windows",
      "intra-video temporal cycle consistency",
      "dynamic thresholding",
      "pose self-supervision",
      "per-identity state",
      "overlap-add decoding"
    ]
  },
  "evidence": {
    "quoted_words_total": 0,
    "full_text_evidence_available": false,
    "items": [
      {
        "evidence_id": "E01",
        "source_url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6630216",
        "page": null,
        "section": "SSRN abstract",
        "equation": null,
        "paraphrase": "The abstract describes intra-video temporal cycle consistency, video-level supervision, pose input, and statistical thresholds computed from local periodicity-score features.",
        "scope_limit": "Abstract-level disclosure only; it cannot establish the exact method, equations, losses, training protocol, or evaluation implementation."
      }
    ],
    "requested_mechanism_matrix": [
      {
        "mechanism": "explicit local time-warp derivative or integral law",
        "status": "NOT_ADJUDICATED",
        "page": null,
        "section": null,
        "equation": null,
        "reason": "Complete method and equations were not obtained."
      },
      {
        "mechanism": "signed phase increments",
        "status": "NOT_ADJUDICATED",
        "page": null,
        "section": null,
        "equation": null,
        "reason": "Complete representation and loss definitions were not obtained."
      },
      {
        "mechanism": "phase-flow or warp equivariance",
        "status": "NOT_ADJUDICATED",
        "page": null,
        "section": null,
        "equation": null,
        "reason": "Complete augmentation and transformation objectives were not obtained."
      },
      {
        "mechanism": "per-identity state",
        "status": "NOT_ADJUDICATED",
        "page": null,
        "section": null,
        "equation": null,
        "reason": "Complete architecture and state management were not obtained."
      },
      {
        "mechanism": "supervision",
        "status": "ABSTRACT_LEVEL_ONLY",
        "page": null,
        "section": "SSRN abstract",
        "equation": null,
        "paraphrase": "The abstract says structured cyclic features are learned with video-level labels, but the exact targets, label access, and training losses remain unresolved."
      },
      {
        "mechanism": "local-window TCC and dynamic thresholding",
        "status": "ABSTRACT_LEVEL_OVERLAP",
        "page": null,
        "section": "SSRN abstract",
        "equation": null,
        "paraphrase": "The abstract explicitly combines intra-video cycle consistency with thresholds derived from local periodicity statistics; exact windows, formulas, and decoder behavior remain unresolved."
      }
    ]
  },
  "comparison": {
    "abstract_level_overlap": "TWCRAC overlaps on pose-based periodic representation learning, intra-video TCC, local periodicity features, dynamic thresholds, video-level supervision, and non-stationary counting. Those elements are expressly outside the residual contribution boundary.",
    "exact_residual_equivalence_test": "NOT_RUN_FULL_TEXT_UNAVAILABLE",
    "method_loss_training_evaluation_comparison": "NOT_ADJUDICATED",
    "absence_inference_used": false,
    "collision_declared": false,
    "distinctness_declared": false
  },
  "verdict": "BLOCKED",
  "k8": {
    "decision": "NOT_PASSED_BLOCKED",
    "reason": "K8 requires complete TWCRAC method inspection. Abstract-only evidence is explicitly insufficient, and no allowed complete primary copy was obtained.",
    "claim_freeze": "BLOCKED",
    "novelty_clearance": false,
    "kill_residual_claim": false,
    "materially_reanchor_claim": false,
    "next_required_action": "Obtain the original SSRN PDF or an author-controlled byte-identical/full copy, record its SHA256 and verified page count, inspect every page of method, loss, training, and evaluation, then rerun this adjudication."
  },
  "caveats": [
    "BLOCKED is not DISTINCT and is not evidence that TWCRAC lacks the residual mechanism.",
    "The SSRN record's 31-page statement was not independently verified from PDF bytes.",
    "Search-result excerpts were used only to locate primary endpoints and author-copy candidates, not to infer method absence.",
    "The abstract-level overlap does not establish equivalence to the signed discrete warp-integral objective.",
    "No EQUIVALENT, OVERLAP_BUT_DISTINCT, or DISTINCT verdict is licensed without complete full text.",
    "This is a same-family OpenAI review and remains provisional."
  ],
  "status": "blocked"
}
