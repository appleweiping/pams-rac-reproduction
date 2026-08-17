"""One-way evaluator/vault join and one-use capability skeleton.

Raw vault and joined rows never appear in the returned result.  Count-blind
commitments and certified primitive evidence are fully validated before the
one-use capability is consumed; any later vault or metric failure is terminal.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import cast

import numpy as np
from numpy.typing import NDArray

from pams.temporac.certify import (
    parse_natural_input_receipt_bytes,
    validate_natural_input_receipt,
)
from pams.temporac.contract import (
    CAPABILITY_GRANT_SCHEMA,
    G5A_RECEIPT_SCHEMA,
    G5B_RECEIPT_SCHEMA,
    K7_RECEIPT_SCHEMA,
    SEEDS,
)
from pams.temporac.hashio import canonical_json_bytes, sha256_bytes
from pams.temporac.metrics import (
    ARMS,
    CONDITIONS,
    Arm,
    BootstrapResult,
    Condition,
    CountMetricRow,
    K7Decision,
    MetricPoint,
    decide_k7,
    paired_component_bootstrap,
    score_count_metrics,
)
from pams.temporac.receipts import (
    LoadedTargetArtifact,
    receipt_bytes,
    target_count_root_sha256,
    validate_feature_record_receipt,
    validate_receipt,
    validate_target_count_rows,
)
from pams.temporac.trusted_packer import VaultRow, vault_bytes
from pams.temporac.types import ContractError, FeatureRecord, PredictionRecord

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_METRIC_FIXTURE_SCHEMA = "temporac.count-metric-fixture-receipt.v1"
_METRIC_FIXTURE_KEYS = frozenset(
    {
        "attack_expected_global_fail_sha256",
        "evaluator_code_sha256",
        "expected_output_sha256",
        "fixture_input_sha256",
        "schema",
        "status",
    }
)


class EvaluatorError(ContractError):
    """A consumed capability, vault join, primitive unit, or metric failure."""


def validate_metric_fixture_receipt(payload: Mapping[str, object]) -> None:
    """Validate the exact surrounding P05M receipt named by the plan."""

    if not isinstance(payload, Mapping) or set(payload) != _METRIC_FIXTURE_KEYS:
        raise EvaluatorError("P05M receipt has unknown or missing keys")
    if payload["schema"] != _METRIC_FIXTURE_SCHEMA or payload["status"] != "PASS":
        raise EvaluatorError("P05M receipt schema/status is invalid")
    for key in _METRIC_FIXTURE_KEYS - {"schema", "status"}:
        value = payload[key]
        if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
            raise EvaluatorError(f"P05M receipt {key} is not lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class PredictionEnvelope:
    """Evaluator-only binding of one immutable prediction to an arm and seed."""

    arm: Arm
    seed: int
    component_token: str
    prediction: PredictionRecord

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise EvaluatorError("prediction envelope arm is invalid")
        if type(self.seed) is not int or self.seed not in SEEDS:
            raise EvaluatorError("prediction envelope seed must be one of the exact three seeds")
        if (
            not isinstance(self.component_token, str)
            or not self.component_token
            or not self.component_token.isascii()
        ):
            raise EvaluatorError("prediction component token must be nonempty ASCII")

    @property
    def identity(self) -> tuple[str, int]:
        return self.prediction.opaque_sample_key.tobytes().hex(), int(
            self.prediction.local_person_slot[0]
        )

    @property
    def condition(self) -> Condition:
        return "natural-clean" if int(self.prediction.condition[0]) == 0 else "natural-drift"

    @property
    def estimate(self) -> int:
        return 0 if int(self.prediction.abstain[0]) == 1 else int(self.prediction.count[0])


@dataclass(frozen=True, slots=True)
class CertifiedDevelopmentTarget:
    """Actual natural target/feature/input-receipt evidence consumed by K7."""

    feature: FeatureRecord
    feature_receipt_bytes: bytes
    feature_receipt_sha256: str
    natural_input_receipt_bytes: bytes
    natural_input_receipt_sha256: str
    target: LoadedTargetArtifact
    source_clocks: NDArray[np.int64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.feature, FeatureRecord) or not isinstance(
            self.target, LoadedTargetArtifact
        ):
            raise EvaluatorError("certified evidence requires typed feature and target records")
        feature_receipt = validate_feature_record_receipt(
            self.feature,
            self.feature_receipt_bytes,
            expected_receipt_sha256=self.feature_receipt_sha256,
        )
        natural_receipt = parse_natural_input_receipt_bytes(
            self.feature,
            self.natural_input_receipt_bytes,
            expected_receipt_sha256=self.natural_input_receipt_sha256,
        )
        preprocessed, _, _, _ = validate_natural_input_receipt(self.feature, natural_receipt)
        target_receipt = self.target.receipt
        if (
            natural_receipt["feature_artifact_sha256"] != feature_receipt["artifact_sha256"]
            or natural_receipt["feature_receipt_sha256"] != self.feature_receipt_sha256
            or natural_receipt["source_binding_sha256"]
            != feature_receipt["source_binding_sha256"]
            or target_receipt["source_kind"] != 1
            or target_receipt["source_key_hex"] != self.feature.opaque_key_bytes.hex()
            or target_receipt["source_unit_index"] != self.feature.slot
            or target_receipt["teacher_sha256"] != natural_receipt["teacher_sha256"]
            or int(self.target.arrays["source_kind"][0]) != 1
            or self.target.arrays["teacher_sha256"].tobytes(order="C").hex()
            != natural_receipt["teacher_sha256"]
        ):
            raise EvaluatorError("certified target identity/teacher/source binding is inconsistent")
        pulse = self.target.arrays["pulse"]
        if pulse.shape != (preprocessed.clocks.size - 1,):
            raise EvaluatorError("certified pulse axis differs from its canonical source clock")
        backing = np.ascontiguousarray(preprocessed.clocks, dtype="<i8").tobytes(order="C")
        clocks = np.frombuffer(backing, dtype="<i8").reshape(preprocessed.clocks.shape)
        object.__setattr__(self, "source_clocks", clocks)

    @property
    def identity(self) -> tuple[str, int]:
        return self.feature.opaque_key_bytes.hex(), self.feature.slot

    @property
    def pulse(self) -> NDArray[np.uint8]:
        return self.target.arrays["pulse"]  # type: ignore[return-value]

    @property
    def teacher_target_count(self) -> int:
        return int(np.sum(self.pulse, dtype=np.int64))


def validate_vault_join(
    population_identities: Sequence[tuple[str, int]],
    vault_rows: Sequence[VaultRow],
    *,
    expected_cardinality: int = 402,
) -> Mapping[tuple[str, int], VaultRow]:
    """Verify exact total one-to-one opaque-key/slot join without returning labels."""

    if type(expected_cardinality) is not int or expected_cardinality <= 0:
        raise EvaluatorError("expected join cardinality must be positive")
    population = tuple(population_identities)
    if len(population) != expected_cardinality or len(set(population)) != len(population):
        raise EvaluatorError("population join side has wrong cardinality or duplicate identities")
    for key, slot in population:
        if _SHA256_RE.fullmatch(key) is None or type(slot) is not int or slot < 0:
            raise EvaluatorError("population identity is malformed")
    vault: dict[tuple[str, int], VaultRow] = {}
    for row in vault_rows:
        if not isinstance(row, VaultRow) or row.identity in vault:
            raise EvaluatorError("vault join side is malformed or duplicate")
        vault[row.identity] = row
    if len(vault) != expected_cardinality or set(vault) != set(population):
        raise EvaluatorError("vault join is missing, extra, or non-total")
    return MappingProxyType(vault)


def validate_vault_period_units(rows: Sequence[VaultRow], *, minimum_period: int = 5) -> None:
    """Apply the K7 raw interval/count kill checks globally."""

    if type(minimum_period) is not int or minimum_period != 5 or not rows:
        raise EvaluatorError("vault period validation requires nonempty rows and P_min=5")
    for row in rows:
        if len(row.periods) != row.count_gt:
            raise EvaluatorError("vault count and raw period-list length are inconsistent")
        previous_end = -1
        for start, end in row.periods:
            if end - start < minimum_period:
                raise EvaluatorError("vault contains a sub-P_min evaluator interval")
            if start < previous_end:
                raise EvaluatorError("vault intervals are non-increasing or overlap")
            previous_end = end


def _validate_prediction_population_before_consume(
    predictions: Sequence[PredictionEnvelope],
) -> frozenset[tuple[str, int]]:
    if not predictions:
        raise EvaluatorError("prediction population is empty")
    seen: set[tuple[Arm, int, Condition, tuple[str, int]]] = set()
    identities: set[tuple[str, int]] = set()
    for envelope in predictions:
        if not isinstance(envelope, PredictionEnvelope):
            raise EvaluatorError("prediction population contains an untyped envelope")
        key = (envelope.arm, envelope.seed, envelope.condition, envelope.identity)
        if key in seen:
            raise EvaluatorError("prediction population contains a duplicate observation")
        seen.add(key)
        identities.add(envelope.identity)
    for identity in identities:
        for arm in ARMS:
            for seed in SEEDS:
                for condition in CONDITIONS:
                    if (arm, seed, condition, identity) not in seen:
                        raise EvaluatorError(
                            "prediction population cross-product is incomplete before consume"
                        )
    return frozenset(identities)


def validate_certified_target_commitment(
    evidence: Sequence[CertifiedDevelopmentTarget],
    target_count_rows: Sequence[Mapping[str, object]],
    *,
    committed_target_count_root_sha256: str,
    prediction_identities: frozenset[tuple[str, int]],
) -> Mapping[tuple[str, int], tuple[int, ...]]:
    """Consume target-count content and actual immutable pulse/source-clock evidence."""

    if not evidence:
        raise EvaluatorError("certified development target evidence is empty")
    normalized_rows = validate_target_count_rows(target_count_rows)
    if target_count_root_sha256(normalized_rows) != committed_target_count_root_sha256:
        raise EvaluatorError("target-count row content differs from the G5a committed root")
    row_counts = {
        (str(row["opaque_key_hex"]), cast(int, row["slot"])): cast(
            int, row["teacher_target_count"]
        )
        for row in normalized_rows
    }
    observed: dict[tuple[str, int], tuple[int, ...]] = {}
    for item in evidence:
        if not isinstance(item, CertifiedDevelopmentTarget) or item.identity in observed:
            raise EvaluatorError("certified development evidence is untyped or duplicate")
        if item.identity not in prediction_identities:
            raise EvaluatorError("certified target identity is absent from prediction population")
        pulse = item.pulse
        if (
            pulse.shape != (item.source_clocks.size - 1,)
            or not np.isin(pulse, (0, 1)).all()
            or np.any(np.diff(item.source_clocks) <= 0)
        ):
            raise EvaluatorError("certified pulse/source-clock content is malformed")
        observed[item.identity] = tuple(
            int(value) for value in item.source_clocks[:-1][pulse == 1]
        )
    if {identity: len(starts) for identity, starts in observed.items()} != row_counts:
        raise EvaluatorError(
            "target-count rows are not an identity-wise equality to certified pulse counts"
        )
    return MappingProxyType(observed)


def validate_certified_pulse_units(
    vault: Mapping[tuple[str, int], VaultRow],
    committed_pulse_starts: Mapping[tuple[str, int], tuple[int, ...]],
) -> None:
    """Apply the exact one-pulse-per-interval/no-outside K7 primitive check."""

    for identity, starts in committed_pulse_starts.items():
        if identity not in vault:
            raise EvaluatorError("certified identity is absent from vault or target-count rows")
        row = vault[identity]
        pulse_starts = np.asarray(starts, dtype="<i8")
        if (
            pulse_starts.size != len(row.periods)
            or pulse_starts.size != row.count_gt
        ):
            raise EvaluatorError("teacher/vault count equality fails for a certified identity")
        ownership = np.zeros(pulse_starts.size, dtype=np.int64)
        for start, end in row.periods:
            inside = (pulse_starts >= start) & (pulse_starts < end)
            if int(np.count_nonzero(inside)) != 1:
                raise EvaluatorError("evaluator interval does not contain exactly one pulse start")
            ownership += inside.astype(np.int64)
        if not np.all(ownership == 1):
            raise EvaluatorError("a certified pulse start lies outside the evaluator interval union")


def metric_rows_from_predictions(
    predictions: Sequence[PredictionEnvelope], vault: Mapping[tuple[str, int], VaultRow]
) -> tuple[CountMetricRow, ...]:
    """Join only inside the evaluator and emit desensitized metric rows."""

    if not predictions:
        raise EvaluatorError("prediction population is empty")
    rows: list[CountMetricRow] = []
    for envelope in predictions:
        if not isinstance(envelope, PredictionEnvelope) or envelope.identity not in vault:
            raise EvaluatorError("prediction identity is absent from the evaluator vault")
        vault_row = vault[envelope.identity]
        rows.append(
            CountMetricRow(
                arm=envelope.arm,
                seed=envelope.seed,
                condition=envelope.condition,
                video_token=envelope.identity[0],
                slot=envelope.identity[1],
                component_token=envelope.component_token,
                count_gt=vault_row.count_gt,
                estimate=envelope.estimate,
            )
        )
    return tuple(rows)


def _point_payload(point: MetricPoint) -> dict[str, object]:
    return {
        "avg_mae": {
            f"{arm}/{condition}": point.avg_mae[(arm, condition)]
            for arm in ARMS
            for condition in CONDITIONS
        },
        "avg_obo": {
            f"{arm}/{condition}": point.avg_obo[(arm, condition)]
            for arm in ARMS
            for condition in CONDITIONS
        },
        "clean_delta": point.clean_delta,
        "route_margin": point.route_margin,
        "strongest_comparator": point.strongest_comparator,
    }


def _metric_payload(
    point: MetricPoint, bootstrap: BootstrapResult, decision: K7Decision
) -> dict[str, object]:
    return {
        "bootstrap": {
            "clean_upper": bootstrap.clean_upper,
            "comparator_draw_counts": dict(bootstrap.comparator_draw_counts),
            "route_lower": bootstrap.route_lower,
            "route_upper": bootstrap.route_upper,
        },
        "decision": {"failures": list(decision.failures), "status": decision.status},
        "point": _point_payload(point),
        "schema": "temporac.metric-payload.v4",
    }


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Hash-bound receipts and aggregate metrics; never raw joined rows."""

    g5b_receipt: Mapping[str, object]
    k7_receipt: Mapping[str, object]
    metric_payload: Mapping[str, object]

    def __post_init__(self) -> None:
        validate_receipt(self.g5b_receipt, expected_schema=G5B_RECEIPT_SCHEMA)
        validate_receipt(self.k7_receipt, expected_schema=K7_RECEIPT_SCHEMA)
        object.__setattr__(self, "g5b_receipt", MappingProxyType(dict(self.g5b_receipt)))
        object.__setattr__(self, "k7_receipt", MappingProxyType(dict(self.k7_receipt)))
        object.__setattr__(self, "metric_payload", MappingProxyType(dict(self.metric_payload)))


class OneUseEvaluator:
    """In-memory witness of a consumed, non-resumable evaluator capability."""

    __slots__ = ("_consumed",)

    def __init__(self) -> None:
        self._consumed = False

    @property
    def consumed(self) -> bool:
        return self._consumed

    def run(
        self,
        *,
        grant: Mapping[str, object],
        g5a_receipt: Mapping[str, object],
        evaluator_code_sha256: str,
        population_identities: Sequence[tuple[str, int]],
        vault_rows: Sequence[VaultRow],
        predictions: Sequence[PredictionEnvelope],
        certified_development_targets: Sequence[CertifiedDevelopmentTarget],
        target_count_rows: Sequence[Mapping[str, object]],
        prediction_root_sha256: str | None = None,
        p2_metric_fixture_receipt: Mapping[str, object] | None = None,
        s0_metric_fixture_receipt_sha256: str | None = None,
        s0_evaluator_code_sha256: str | None = None,
    ) -> EvaluationResult:
        """Validate commitments, consume once, then execute G5b and K7 in order."""

        if self._consumed:
            raise EvaluatorError("evaluator capability has already been consumed")
        validate_receipt(grant, expected_schema=CAPABILITY_GRANT_SCHEMA)
        validate_receipt(g5a_receipt, expected_schema=G5A_RECEIPT_SCHEMA)
        g5a_bytes = receipt_bytes(g5a_receipt, expected_schema=G5A_RECEIPT_SCHEMA)
        g5a_sha256 = sha256_bytes(g5a_bytes)
        if grant["g5a_receipt_sha256"] != g5a_sha256:
            raise EvaluatorError("capability does not bind the exact G5a receipt bytes")
        if (
            g5a_receipt["evaluator_code_sha256"] != evaluator_code_sha256
            or grant["evaluator_code_sha256"] != evaluator_code_sha256
            or s0_evaluator_code_sha256 != evaluator_code_sha256
        ):
            raise EvaluatorError("P2/S0/G5a/grant evaluator code hashes differ")
        if prediction_root_sha256 != grant["prediction_root_sha256"]:
            raise EvaluatorError("capability prediction root differs from the committed root")
        if p2_metric_fixture_receipt is None:
            raise EvaluatorError("P05M receipt is required before G5b/K7")
        validate_metric_fixture_receipt(p2_metric_fixture_receipt)
        p2_receipt_sha256 = sha256_bytes(canonical_json_bytes(p2_metric_fixture_receipt))
        if (
            p2_metric_fixture_receipt["evaluator_code_sha256"] != evaluator_code_sha256
            or s0_metric_fixture_receipt_sha256 != p2_receipt_sha256
        ):
            raise EvaluatorError("P05M evaluator/receipt hashes differ from S0")
        prediction_identities = _validate_prediction_population_before_consume(predictions)
        committed_pulse_starts = validate_certified_target_commitment(
            certified_development_targets,
            target_count_rows,
            committed_target_count_root_sha256=str(g5a_receipt["target_count_root_sha256"]),
            prediction_identities=prediction_identities,
        )
        self._consumed = True
        vault = validate_vault_join(population_identities, vault_rows)
        observed_vault_root = sha256_bytes(vault_bytes(vault_rows))
        if grant["vault_root_sha256"] != observed_vault_root:
            raise EvaluatorError("capability vault root differs from exact vault bytes")
        validate_vault_period_units(vault_rows)
        validate_certified_pulse_units(
            vault,
            committed_pulse_starts,
        )
        g5b: dict[str, object] = {
            "g5a_receipt_sha256": g5a_sha256,
            "join_cardinality": 402,
            "join_commitment_sha256": grant["join_commitment_sha256"],
            "schema": G5B_RECEIPT_SCHEMA,
            "status": "PASS",
            "vault_root_sha256": observed_vault_root,
        }
        validate_receipt(g5b, expected_schema=G5B_RECEIPT_SCHEMA)
        rows = metric_rows_from_predictions(predictions, vault)
        point = score_count_metrics(rows)
        bootstrap = paired_component_bootstrap(rows)
        decision = decide_k7(point, bootstrap)
        metric_payload = _metric_payload(point, bootstrap, decision)
        metric_sha256 = sha256_bytes(canonical_json_bytes(metric_payload))
        g5b_sha256 = sha256_bytes(receipt_bytes(g5b, expected_schema=G5B_RECEIPT_SCHEMA))
        k7: dict[str, object] = {
            "g5a_receipt_sha256": g5a_sha256,
            "g5b_receipt_sha256": g5b_sha256,
            "metric_payload_sha256": metric_sha256,
            "schema": K7_RECEIPT_SCHEMA,
            "status": decision.status,
        }
        validate_receipt(k7, expected_schema=K7_RECEIPT_SCHEMA)
        return EvaluationResult(g5b, k7, metric_payload)


def count_metric_fixture_receipt(
    *,
    evaluator_code_sha256: str,
    fixture_input_bytes: bytes,
    expected_output_bytes: bytes,
    attack_expected_global_fail_bytes: bytes,
) -> Mapping[str, object]:
    """Build the surrounding P05M receipt named by the current plan."""

    if _SHA256_RE.fullmatch(evaluator_code_sha256) is None:
        raise EvaluatorError("fixture evaluator code hash must be lowercase SHA-256")
    payload: dict[str, object] = {
        "attack_expected_global_fail_sha256": sha256_bytes(attack_expected_global_fail_bytes),
        "evaluator_code_sha256": evaluator_code_sha256,
        "expected_output_sha256": sha256_bytes(expected_output_bytes),
        "fixture_input_sha256": sha256_bytes(fixture_input_bytes),
        "schema": _METRIC_FIXTURE_SCHEMA,
        "status": "PASS",
    }
    validate_metric_fixture_receipt(payload)
    return MappingProxyType(payload)
