"""Frozen constants for the private ``temporac.execution.v4`` contract."""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from types import MappingProxyType

CONTRACT_NAME = "temporac.execution.v4"
PROPOSAL_SHA256 = "3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391"
EFFECTIVE_CONTRACT_SCHEMA = "temporac.effective-contract-index.v4"
EFFECTIVE_CONTRACT_ROWS: tuple[tuple[int, str, str], ...] = (
    (79_366, "canonical_proposal", PROPOSAL_SHA256),
    (
        312_728,
        "amendment_markdown",
        "2eaa3cfb746804fdc5890fec4c252c877a031a4abedef7b9ef6eeac0a4839395",
    ),
    (
        261_196,
        "amendment_json",
        "638efc8708f74c8514247f4f4c5e2311595e90edc6744c9b371886f0789af164",
    ),
    (
        15_666,
        "accepted_review_markdown",
        "4a3199c3ac18e3fd2a5bb7ecf6d66e10bc4a5cebb82363803c47867234b70e67",
    ),
    (
        10_036,
        "accepted_review_json",
        "fcf8a9d4949f6b456f18bfb4591c512b6628c6ec7eb5a055873a995992f46ad6",
    ),
)
EFFECTIVE_CONTRACT_BYTES = 665
CONTRACT_SHA256 = "c203f4aab00879acf041b90829f0124d0b1de231294f191b03f3166f728e3d5c"

SEEDS: tuple[int, ...] = (20260815, 20260816, 20260817)
SHORTCUT_NAMES: tuple[str, ...] = (
    "no-pose-timestamp",
    "nuisance-only",
    "pose-shuffle",
    "static-code-only",
    "warp-metadata",
    "track-length",
)

TEACHER_JOB_NAMES: tuple[str, ...] = tuple(f"{CONTRACT_NAME}/teacher/seed={seed}" for seed in SEEDS)
CANONICAL_JOB_NAMES: tuple[str, ...] = tuple(
    f"{CONTRACT_NAME}/response/canonical/seed={seed}" for seed in SEEDS
)
CAPACITY_CONTROL_JOB_NAMES: tuple[str, ...] = tuple(
    f"{CONTRACT_NAME}/response/capacity-control/seed={seed}" for seed in SEEDS
)
SHORTCUT_JOB_NAMES: tuple[str, ...] = tuple(
    f"{CONTRACT_NAME}/response/shortcut/{name}/seed={seed}"
    for name in SHORTCUT_NAMES
    for seed in SEEDS
)
RESPONSE_JOB_NAMES = CANONICAL_JOB_NAMES + CAPACITY_CONTROL_JOB_NAMES + SHORTCUT_JOB_NAMES
JOB_NAMES = TEACHER_JOB_NAMES + RESPONSE_JOB_NAMES

if len(JOB_NAMES) != 27 or len(set(JOB_NAMES)) != 27:  # pragma: no cover - import invariant
    raise RuntimeError("the TempoRAC training inventory must contain exactly 27 unique jobs")

# Formula F23, grouped exactly by its frozen S-stage clauses and without
# inserting the preparatory P00--P06 stages.
F23_STAGES: tuple[str, ...] = (
    "S0",
    "S1:G0,K0",
    "S2:G1,K1,G2,K2",
    "S3:G3,K3,K4",
    "S4:G4,K5,K6",
    "freeze natural artifacts",
    "G5a",
    "capability grant",
    "G5b",
    "K7",
)
F23 = " -> ".join(F23_STAGES)
F23_ORDER: tuple[str, ...] = (
    "S0",
    "G0",
    "K0",
    "G1",
    "K1",
    "G2",
    "K2",
    "G3",
    "K3",
    "K4",
    "G4",
    "K5",
    "K6",
    "freeze natural artifacts",
    "G5a",
    "capability grant",
    "G5b",
    "K7",
)

TRAINING_A6000_HOURS = 78.0
HELDOUT_INFERENCE_A6000_HOURS = 3.0
NATURAL_PREDICTION_A6000_HOURS = 12.0
ALLOCATED_A6000_HOURS = 93.0
UNALLOCATED_A6000_HOURS = 7.0
CAMPAIGN_A6000_HOUR_CEILING = 100.0
MAX_CONCURRENCY = 2
CUDA_BYTE_CEILING = 24 * 1024**3

FEATURE_ARCHIVE_MAX_BYTES = 8 * 1024**2
FEATURE_EXPANDED_MAX_BYTES = 8 * 1024**2
ZIP_TIMESTAMP: tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0)


class ReasonCode(IntEnum):
    """Ascending, unique abstention reasons from Section 13.1."""

    SOURCE_SCHEMA = 1
    SOURCE_JOIN = 2
    DUPLICATE_CONFLICT = 3
    CLOCK_ORDER = 4
    NORMALIZATION = 5
    FRAME_SUPPORT = 6
    EDGE_SUPPORT = 7
    GEOMETRY_LENGTH = 8
    TOO_SHORT = 9
    TEACHER_NUMERIC = 10
    LANDMARK = 11
    OBSERVED_PHASE = 12
    DENSE_PHASE = 13
    TOPOLOGY = 14
    RECONSTRUCTION = 15
    COLLISION = 16
    ORIGIN = 17
    PULSE = 18
    SYNTHETIC_PULSE_MISMATCH = 19
    NOLA = 20
    RESPONSE_NONFINITE = 21
    RUNTIME = 22
    ARTIFACT_INTEGRITY = 23
    CONFIG_MISMATCH = 24


AbstentionReason = ReasonCode
REASON_CODES: tuple[ReasonCode, ...] = tuple(ReasonCode)

# Array schemas use NumPy dtype strings.  ``None`` denotes the one variable
# dimension established by the enclosing record.
_MemberSchema = Mapping[str, tuple[str, tuple[int | None, ...]]]
FEATURE_MEMBER_SCHEMA: _MemberSchema = MappingProxyType(
    {
        "frame_mask": ("|u1", (320,)),
        "local_person_slot": ("<i8", (1,)),
        "motion": ("<f4", (320, 17, 3)),
        "opaque_sample_key": ("|u1", (32,)),
        "person_mask": ("|u1", (1,)),
        "sampled_frame_indices": ("<i8", (320,)),
        "source_length": ("<i8", (1,)),
    }
)

TARGET_MEMBER_SCHEMA: _MemberSchema = MappingProxyType(
    {
        "chi": ("<f8", (None,)),
        "contract_sha256": ("|u1", (32,)),
        "edge_mask": ("|u1", (None,)),
        "pulse": ("|u1", (None,)),
        "source_kind": ("|u1", (1,)),
        "target_mask": ("|u1", (None,)),
        "teacher_sha256": ("|u1", (32,)),
    }
)

PREDICTION_MEMBER_SCHEMA: _MemberSchema = MappingProxyType(
    {
        "abstain": ("|u1", (1,)),
        "abstain_reasons": ("<u2", (None,)),
        "component_bounds": ("<i4", (None, 2)),
        "component_location": ("<i4", (None,)),
        "component_score": ("<f4", (None,)),
        "condition": ("|u1", (1,)),
        "contract_sha256": ("|u1", (32,)),
        "count": ("<i8", (1,)),
        "decoder_mask": ("|u1", (None,)),
        "edge_mask": ("|u1", (None,)),
        "local_person_slot": ("<i8", (1,)),
        "opaque_sample_key": ("|u1", (32,)),
        "response": ("<f4", (None,)),
        "run_bounds": ("<i4", (None, 2)),
    }
)

FEATURE_MEMBER_NAMES = tuple(sorted(FEATURE_MEMBER_SCHEMA, key=str.encode))
TARGET_MEMBER_NAMES = tuple(sorted(TARGET_MEMBER_SCHEMA, key=str.encode))
PREDICTION_MEMBER_NAMES = tuple(sorted(PREDICTION_MEMBER_SCHEMA, key=str.encode))

FEATURE_RECEIPT_SCHEMA = "temporac.feature-receipt.v4"
TARGET_RECEIPT_SCHEMA = "temporac.target-receipt.v4"
PREDICTION_RECEIPT_SCHEMA = "temporac.prediction-receipt.v4"
RUN_RECEIPT_SCHEMA = "temporac.run-receipt.v4"
G5A_RECEIPT_SCHEMA = "temporac.g5a-receipt.v4"
CAPABILITY_REQUEST_SCHEMA = "temporac.capability-request.v4"
CAPABILITY_GRANT_SCHEMA = "temporac.capability-grant.v4"
G5B_RECEIPT_SCHEMA = "temporac.g5b-receipt.v4"
K7_RECEIPT_SCHEMA = "temporac.k7-receipt.v4"
PRE_G5A_STAGE_RECEIPT_SCHEMA = "temporac.pre-g5a-stage-receipt.v4"
G1_TEACHER_SELECTION_RECEIPT_SCHEMA = "temporac.g1-teacher-selection-receipt.v4"
K1_CERTIFICATE_OUTCOME_RECEIPT_SCHEMA = "temporac.k1-certificate-outcome-receipt.v4"
X0_INFERENCE_RECEIPT_SCHEMA = "temporac.x0-inference-receipt.v4"
K3_RECEIPT_SCHEMA = "temporac.k3-receipt.v4"
K4_RECEIPT_SCHEMA = "temporac.k4-receipt.v4"
NATURAL_PREDICTION_COMPLETION_RECEIPT_SCHEMA = "temporac.natural-prediction-completion-receipt.v4"
PRE_G5A_RECEIPT_INVENTORY_SCHEMA = "temporac.pre-g5a-receipt-inventory.v4"
RECEIPT_DAG_SCHEMA = "temporac.receipt-dag.v4"
K1_CERTIFICATE_OUTCOME_INDEX_SCHEMA = "temporac.k1-certificate-outcome-index.v4"

EFFECTIVE_CONTRACT_KEYS = frozenset({"rows", "schema"})
EFFECTIVE_CONTRACT_ROW_KEYS = frozenset({"bytes", "role", "sha256"})
PRE_G5A_INVENTORY_KEYS = frozenset({"contract_sha256", "rows", "schema"})
PRE_G5A_INVENTORY_ROW_KEYS = frozenset(
    {
        "node_class",
        "owner",
        "receipt_schema",
        "receipt_sha256",
        "upstream_owner_tokens",
        "upstream_receipt_sha256",
    }
)
RECEIPT_DAG_KEYS = frozenset({"contract_sha256", "edges", "nodes", "schema"})
RECEIPT_DAG_NODE_KEYS = frozenset({"class", "owner_key", "receipt_sha256"})
RECEIPT_DAG_EDGE_KEYS = frozenset({"from_receipt_sha256", "ordinal", "role", "to_receipt_sha256"})
K1_CERTIFICATE_OUTCOME_INDEX_KEYS = frozenset(
    {"contract_sha256", "population_manifest_sha256", "rows", "schema"}
)
K1_CERTIFICATE_OUTCOME_ROW_KEYS = frozenset(
    {
        "certificate_reasons",
        "certificate_status",
        "component_key_hex",
        "eligible",
        "feature_artifact_sha256_or_null",
        "feature_receipt_sha256_or_null",
        "natural_input_receipt_sha256_or_null",
        "opaque_key_hex",
        "population_row_sha256",
        "slot",
        "split",
        "target_artifact_sha256_or_null",
        "target_receipt_sha256_or_null",
    }
)

RECEIPT_DAG_NODE_CLASSES: tuple[str, ...] = (
    "feature",
    "g1-teacher-selection",
    "k1-certificate-outcome",
    "k3",
    "k4",
    "natural-certificate-input",
    "natural-prediction-completion",
    "pre-g5a-stage",
    "prediction",
    "run",
    "target",
    "teacher-checkpoint",
    "teacher-selection",
    "teacher-tune-evaluation",
    "teacher-tune-input",
    "x0-inference",
)
RECEIPT_DAG_ROLES: tuple[str, ...] = (
    "checkpoint.run",
    "evaluation.checkpoint",
    "evaluation.tune-input",
    "g1.selection",
    "g1.teacher-run",
    "g1.tune-input",
    "k1.feature",
    "k1.natural-input",
    "k1.target",
    "k1.upstream",
    "natp.k6",
    "natp.prediction",
    "natural-input.feature",
    "natural-input.selection",
    "prediction.feature",
    "prediction.run",
    "run.upstream",
    "selection.per-seed-winner",
    "selection.tune-input",
    "stage.upstream",
    "target.natural-input",
    "target.selection",
)

# Exact Amendment005 owner order.  It is built only from frozen tuples above;
# no filesystem discovery or lexical re-sorting is involved.
PRE_G5A_OWNER_ROSTER: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("pre-g5a-stage", "P00-IMPLEMENT", PRE_G5A_STAGE_RECEIPT_SCHEMA, ()),
    (
        "pre-g5a-stage",
        "P01-STATIC",
        PRE_G5A_STAGE_RECEIPT_SCHEMA,
        ("P00-IMPLEMENT",),
    ),
    (
        "pre-g5a-stage",
        "P02-X0-FIXTURE",
        PRE_G5A_STAGE_RECEIPT_SCHEMA,
        ("P01-STATIC",),
    ),
    (
        "pre-g5a-stage",
        "P03-TOPO-FIXTURE",
        PRE_G5A_STAGE_RECEIPT_SCHEMA,
        ("P02-X0-FIXTURE",),
    ),
    (
        "pre-g5a-stage",
        "P04-OP-FIXTURE",
        PRE_G5A_STAGE_RECEIPT_SCHEMA,
        ("P03-TOPO-FIXTURE",),
    ),
    (
        "pre-g5a-stage",
        "P05-GRAPH-FIXTURE",
        PRE_G5A_STAGE_RECEIPT_SCHEMA,
        ("P04-OP-FIXTURE",),
    ),
    (
        "pre-g5a-stage",
        "P05M-COUNT-METRIC",
        PRE_G5A_STAGE_RECEIPT_SCHEMA,
        ("P01-STATIC",),
    ),
    (
        "pre-g5a-stage",
        "P06-FRESH-PREFLIGHT",
        PRE_G5A_STAGE_RECEIPT_SCHEMA,
        ("P01-STATIC", "P05-GRAPH-FIXTURE", "P05M-COUNT-METRIC"),
    ),
    (
        "pre-g5a-stage",
        "S0-COMMIT",
        PRE_G5A_STAGE_RECEIPT_SCHEMA,
        (
            "P00-IMPLEMENT",
            "P01-STATIC",
            "P02-X0-FIXTURE",
            "P03-TOPO-FIXTURE",
            "P04-OP-FIXTURE",
            "P05-GRAPH-FIXTURE",
            "P05M-COUNT-METRIC",
            "P06-FRESH-PREFLIGHT",
        ),
    ),
    (
        "pre-g5a-stage",
        "G0-ACQUIRE",
        PRE_G5A_STAGE_RECEIPT_SCHEMA,
        ("S0-COMMIT",),
    ),
    (
        "pre-g5a-stage",
        "K0-FIREWALL",
        PRE_G5A_STAGE_RECEIPT_SCHEMA,
        ("G0-ACQUIRE",),
    ),
    *(
        ("run", owner, RUN_RECEIPT_SCHEMA, ("S0-COMMIT", "K0-FIREWALL"))
        for owner in TEACHER_JOB_NAMES
    ),
    (
        "g1-teacher-selection",
        "G1-TEACHER-AGG",
        G1_TEACHER_SELECTION_RECEIPT_SCHEMA,
        TEACHER_JOB_NAMES,
    ),
    (
        "k1-certificate-outcome",
        "K1-COVERAGE",
        K1_CERTIFICATE_OUTCOME_RECEIPT_SCHEMA,
        ("G0-ACQUIRE", "G1-TEACHER-AGG"),
    ),
    (
        "pre-g5a-stage",
        "G2-CONFORMANCE",
        PRE_G5A_STAGE_RECEIPT_SCHEMA,
        ("P05-GRAPH-FIXTURE", "K1-COVERAGE"),
    ),
    (
        "pre-g5a-stage",
        "K2-TARGETS",
        PRE_G5A_STAGE_RECEIPT_SCHEMA,
        ("G1-TEACHER-AGG", "K1-COVERAGE", "G2-CONFORMANCE"),
    ),
    *(("run", owner, RUN_RECEIPT_SCHEMA, ("K2-TARGETS",)) for owner in RESPONSE_JOB_NAMES),
    (
        "pre-g5a-stage",
        "G3-RESPONSE-AGG",
        PRE_G5A_STAGE_RECEIPT_SCHEMA,
        ("K2-TARGETS", *RESPONSE_JOB_NAMES),
    ),
    *(
        (
            "x0-inference",
            f"X0I-{seed}",
            X0_INFERENCE_RECEIPT_SCHEMA,
            ("G3-RESPONSE-AGG",),
        )
        for seed in SEEDS
    ),
    (
        "k3",
        "K3-ROUTE",
        K3_RECEIPT_SCHEMA,
        tuple(f"X0I-{seed}" for seed in SEEDS),
    ),
    (
        "k4",
        "K4-DIAG",
        K4_RECEIPT_SCHEMA,
        ("K3-ROUTE", *(f"X0I-{seed}" for seed in SEEDS)),
    ),
    (
        "pre-g5a-stage",
        "G4-OPERATOR",
        PRE_G5A_STAGE_RECEIPT_SCHEMA,
        ("P04-OP-FIXTURE", "K4-DIAG", *(f"X0I-{seed}" for seed in SEEDS)),
    ),
    (
        "pre-g5a-stage",
        "K5-BOUNDARY",
        PRE_G5A_STAGE_RECEIPT_SCHEMA,
        ("G4-OPERATOR",),
    ),
    (
        "pre-g5a-stage",
        "K6-RESAMPLER",
        PRE_G5A_STAGE_RECEIPT_SCHEMA,
        ("G1-TEACHER-AGG", "G4-OPERATOR", "K5-BOUNDARY"),
    ),
    *(
        (
            "natural-prediction-completion",
            f"NATP-{seed}",
            NATURAL_PREDICTION_COMPLETION_RECEIPT_SCHEMA,
            ("K6-RESAMPLER",),
        )
        for seed in SEEDS
    ),
)

if len(PRE_G5A_OWNER_ROSTER) != 54 or len({row[1] for row in PRE_G5A_OWNER_ROSTER}) != 54:
    raise RuntimeError("the pre-G5a inventory must contain exactly 54 unique owners")
if sum(len(row[3]) for row in PRE_G5A_OWNER_ROSTER) != 106:
    raise RuntimeError("the pre-G5a owner roster must generate exactly 106 direct edges")

OPERATOR_CANDIDATE_DIRECTORY_NAME = "operator_candidate_v2_20260816"
OPERATOR_CANDIDATE_RECEIPT_SCHEMA = "temporac.operator-manifest-candidate-receipt.v3"
OPERATOR_REVIEW_SCHEMA = "temporac.operator-implementation-review.v2"
OPERATOR_CANDIDATE_SOURCE_SNAPSHOT_SHA256 = (
    "ded975e31ced2cf1c7d69fb217610125f65d419f641a8e0832d115ec7a4870c0"
)
OPERATOR_REVIEW_MARKDOWN_SHA256 = "73c660079377683b2066e4e04e8b6400f3568e1b7473069303908f773d0caad1"
OPERATOR_REVIEW_MARKDOWN_BYTES = 15_244
OPERATOR_REVIEW_JSON_SHA256 = "25792a855e15769ec778276b5066052d15796eea18cad5d44a16f741cb8bd35f"
OPERATOR_REVIEW_JSON_BYTES = 14_108
OPERATOR_CANDIDATE_MEMBERS: tuple[tuple[str, int, str], ...] = (
    (
        "temporac.operator-manifest.v4.json",
        15_039_512,
        "1f39378d6f7a970b1f183558ed145bd97107f2aeda41902f4736e6778930e1bb",
    ),
    (
        "operator_manifest_schema.json",
        3_607,
        "3b29937c641274e73d10afd19bea943f28288b3d2851e8a5bf2313d7ea3e0fc3",
    ),
    (
        "environment_lock.json",
        3_976,
        "47c2227b74b627a03bf9b5e4930dd02484caaae5a695ae5c0129f04781115cd8",
    ),
    (
        "replay_witness.json",
        2_896,
        "348254cfe64cea35d6e29861755489721ebbbd4031f3ec1c044d7fa24fe38010",
    ),
    (
        "candidate_receipt.json",
        6_180,
        "d53fcc1138754eee90a255952606cd054aa04c72cb3f97d66bf4464d5de672cb",
    ),
)
OPERATOR_CANDIDATE_RECEIPT_KEYS = frozenset(
    {
        "authority",
        "bindings",
        "blockers",
        "candidate_status",
        "members",
        "p2_status",
        "s0_status",
        "schema",
        "validated_row_count",
    }
)
OPERATOR_CANDIDATE_BINDING_KEYS = frozenset(
    {
        "amendment_002_json_sha256",
        "amendment_002_md_sha256",
        "amendment_002_review_json_sha256",
        "amendment_002_review_md_sha256",
        "amendment_004_json_sha256",
        "amendment_004_md_sha256",
        "amendment_004_review_json_sha256",
        "amendment_004_review_md_sha256",
        "amendment_006_json_sha256",
        "amendment_006_md_sha256",
        "amendment_006_review_json_sha256",
        "amendment_006_review_md_sha256",
        "generation_project_files",
        "proposal_sha256",
        "review_project_files",
        "runtime_lock_sha256",
        "source_snapshot_sha256",
    }
)
OPERATOR_CANDIDATE_AUTHORITY_KEYS = frozenset(
    {"P2", "S0", "data", "gate", "git", "launch", "paper_claim", "server", "training"}
)
OPERATOR_CANDIDATE_BLOCKERS: tuple[str, ...] = (
    "fresh_implementation_review_required",
    "P2_not_claimed",
    "S0_not_claimed",
    "launch_authorization_zero",
)
OPERATOR_MANIFEST_ROW_KEYS: tuple[str, ...] = (
    "amplitude",
    "association_sha256",
    "component_location_sha256",
    "component_score_sha256",
    "decoded_components_sha256",
    "decoder_mask_sha256",
    "edge_count",
    "edge_mask_sha256",
    "expected_components_sha256",
    "gap",
    "generated_response_sha256",
    "key_hex",
    "layout",
    "nola_quantized_response_sha256",
    "offset",
    "reset_edge",
    "run_bounds_sha256",
    "target_mask_sha256",
    "truth_negative_components",
    "truth_plateau_mask_sha256",
    "truth_plateaus_sha256",
    "width",
    "windows_sha256",
)
OPERATOR_REVIEW_KEYS = frozenset(
    {
        "acceptance_status",
        "audit_skill",
        "audited_inputs",
        "auditor",
        "author_contact",
        "authority",
        "blockers",
        "bridge_protocol_read",
        "candidate_disposition",
        "date",
        "deterministic_evidence_acceptance",
        "excluded_inputs",
        "executor_family",
        "executor_model",
        "findings",
        "generated_at",
        "integrity_checks",
        "integrity_status",
        "manifest_audit",
        "overall_verdict",
        "quality_checks",
        "reason_code",
        "review_independence",
        "review_md",
        "reviewer_family",
        "reviewer_model",
        "reviewer_reasoning",
        "runtime_replay",
        "schema",
        "summary",
        "trace_path",
        "verdict",
    }
)
OPERATOR_REVIEW_AUTHORITY_KEYS = frozenset(
    {
        "P2",
        "P3",
        "S0",
        "candidate_only",
        "data",
        "eligible_input_to_future_separately_authorized_P2_fixture_gate",
        "gate",
        "git",
        "launch",
        "paper_claim",
        "server",
        "training",
    }
)

MEMBER_RECEIPT_KEYS = frozenset({"bytes", "dtype", "name", "sha256", "shape"})
RESOURCE_RECEIPT_KEYS = frozenset(
    {"completed_steps", "gpu_seconds", "max_cuda_bytes", "wall_seconds"}
)
RECEIPT_KEYS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        FEATURE_RECEIPT_SCHEMA: frozenset(
            {
                "artifact_bytes",
                "artifact_sha256",
                "contract_sha256",
                "members",
                "opaque_key_hex",
                "schema",
                "slot",
                "source_binding_sha256",
            }
        ),
        TARGET_RECEIPT_SCHEMA: frozenset(
            {
                "artifact_bytes",
                "artifact_sha256",
                "certificate_status",
                "contract_sha256",
                "members",
                "schema",
                "source_key_hex",
                "source_kind",
                "source_unit_index",
                "teacher_sha256",
            }
        ),
        PREDICTION_RECEIPT_SCHEMA: frozenset(
            {
                "arm",
                "artifact_bytes",
                "artifact_sha256",
                "checkpoint_sha256",
                "contract_sha256",
                "condition",
                "feature_receipt_sha256",
                "job_name_hex",
                "members",
                "schema",
                "seed",
            }
        ),
        RUN_RECEIPT_SCHEMA: frozenset(
            {
                "code_sha256",
                "config_sha256",
                "contract_sha256",
                "environment_sha256",
                "final_step",
                "job_name_hex",
                "optimizer_sha256",
                "ordered_checkpoint_sha256",
                "resource",
                "rng_sha256",
                "schema",
                "seed",
                "source_cycle_sha256",
                "status",
                "upstream_receipt_sha256",
            }
        ),
        G5A_RECEIPT_SCHEMA: frozenset(
            {
                "checkpoint_root_sha256",
                "code_sha256",
                "contract_sha256",
                "environment_sha256",
                "evaluator_code_sha256",
                "feature_root_sha256",
                "population_manifest_sha256",
                "prediction_clean_root_sha256",
                "prediction_drift_root_sha256",
                "receipt_root_sha256",
                "schema",
                "stub_inclusive_artifact_count",
                "target_count_root_sha256",
                "vault_join_commitment_sha256",
            }
        ),
        CAPABILITY_REQUEST_SCHEMA: frozenset(
            {"g5a_receipt_sha256", "previous_record_sha256", "schema", "sequence"}
        ),
        CAPABILITY_GRANT_SCHEMA: frozenset(
            {
                "evaluator_code_sha256",
                "g5a_receipt_sha256",
                "invocation_count",
                "join_commitment_sha256",
                "prediction_root_sha256",
                "schema",
                "vault_root_sha256",
            }
        ),
        G5B_RECEIPT_SCHEMA: frozenset(
            {
                "g5a_receipt_sha256",
                "join_cardinality",
                "join_commitment_sha256",
                "schema",
                "status",
                "vault_root_sha256",
            }
        ),
        K7_RECEIPT_SCHEMA: frozenset(
            {
                "g5a_receipt_sha256",
                "g5b_receipt_sha256",
                "metric_payload_sha256",
                "schema",
                "status",
            }
        ),
        NATURAL_PREDICTION_COMPLETION_RECEIPT_SCHEMA: frozenset(
            {
                "clean_seed_index_sha256",
                "clean_seed_root_sha256",
                "code_index_sha256",
                "contract_sha256",
                "drift_seed_index_sha256",
                "drift_seed_root_sha256",
                "environment_sha256",
                "k6_receipt_sha256",
                "owner",
                "pilot_scope_manifest_sha256",
                "schema",
                "seed",
                "status",
                "upstream_receipt_sha256",
            }
        ),
    }
)
