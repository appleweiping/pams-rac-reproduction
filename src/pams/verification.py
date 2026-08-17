"""Frozen verification gate for the PAMS UCFRep reproduction.

The thresholds and required evidence in this module are protocol constants.
They are intentionally not loaded from a result file: a result artifact cannot
move its own goalposts.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


class VerificationStatus(str, Enum):
    """Machine-readable state of the reproduction claim."""

    NO_RESULTS = "no_results"
    UNVERIFIED = "unverified"
    PARTIAL = "partial"
    VERIFIED = "verified"


@dataclass(frozen=True, slots=True)
class MetricPair:
    """The two metrics used by the frozen acceptance gate."""

    nmae: float
    obo: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.nmae) or self.nmae < 0.0:
            raise ValueError("nmae must be finite and non-negative")
        if not math.isfinite(self.obo) or not 0.0 <= self.obo <= 1.0:
            raise ValueError("obo must be finite and in [0, 1]")

    def to_dict(self) -> dict[str, float]:
        return {"nmae": self.nmae, "obo": self.obo}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MetricPair:
        _require_exact_keys(payload, {"nmae", "obo"}, "metric pair")
        return cls(nmae=float(payload["nmae"]), obo=float(payload["obo"]))


@dataclass(frozen=True, slots=True)
class SeedResult:
    """Aggregate held-out metrics from one preregistered seed."""

    seed: int
    metrics: MetricPair

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or int(self.seed) != self.seed:
            raise TypeError("seed must be an integer")

    def to_dict(self) -> dict[str, float | int]:
        return {"seed": self.seed, **self.metrics.to_dict()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SeedResult:
        _require_exact_keys(payload, {"seed", "nmae", "obo"}, "seed result")
        seed = _strict_int(payload["seed"], "seed")
        return cls(
            seed=seed,
            metrics=MetricPair(
                nmae=float(payload["nmae"]),
                obo=float(payload["obo"]),
            ),
        )


@dataclass(frozen=True, slots=True)
class VerificationEvidence:
    """All measured evidence consumed by the acceptance gate."""

    seed_results: tuple[SeedResult, ...] = ()
    ablations: Mapping[str, MetricPair] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        results = tuple(self.seed_results)
        seeds = [result.seed for result in results]
        if len(seeds) != len(set(seeds)):
            raise ValueError("seed_results must not contain duplicate seeds")
        normalized: dict[str, MetricPair] = {}
        for raw_key, metrics in self.ablations.items():
            key = str(raw_key).strip()
            if not key:
                raise ValueError("ablation keys must be non-empty")
            if key in normalized:
                raise ValueError(f"duplicate ablation key: {key}")
            if not isinstance(metrics, MetricPair):
                raise TypeError("ablation values must be MetricPair instances")
            normalized[key] = metrics
        object.__setattr__(self, "seed_results", results)
        object.__setattr__(self, "ablations", MappingProxyType(normalized))

    @property
    def has_any_results(self) -> bool:
        return bool(self.seed_results or self.ablations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_results": [
                result.to_dict() for result in sorted(self.seed_results, key=lambda item: item.seed)
            ],
            "ablations": {key: self.ablations[key].to_dict() for key in sorted(self.ablations)},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VerificationEvidence:
        _require_exact_keys(payload, {"seed_results", "ablations"}, "verification evidence")
        raw_seed_results = payload["seed_results"]
        raw_ablations = payload["ablations"]
        if not isinstance(raw_seed_results, list):
            raise TypeError("seed_results must be a list")
        if not isinstance(raw_ablations, Mapping):
            raise TypeError("ablations must be a mapping")
        seed_results = tuple(
            SeedResult.from_dict(_as_mapping(item, "seed result")) for item in raw_seed_results
        )
        ablations = {
            str(key): MetricPair.from_dict(_as_mapping(value, "ablation metric pair"))
            for key, value in raw_ablations.items()
        }
        return cls(seed_results=seed_results, ablations=ablations)


@dataclass(frozen=True, slots=True)
class VerificationGate:
    """Immutable preregistered acceptance criteria."""

    required_seeds: tuple[int, int, int]
    maximum_mean_nmae: float
    minimum_mean_obo: float
    minimum_individual_passes: int
    required_ablations: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(set(self.required_seeds)) != len(self.required_seeds):
            raise ValueError("required_seeds must be unique")
        if not 0 <= self.minimum_individual_passes <= len(self.required_seeds):
            raise ValueError("minimum_individual_passes is outside the seed range")
        if self.maximum_mean_nmae < 0.0:
            raise ValueError("maximum_mean_nmae must be non-negative")
        if not 0.0 <= self.minimum_mean_obo <= 1.0:
            raise ValueError("minimum_mean_obo must be in [0, 1]")
        if len(set(self.required_ablations)) != len(self.required_ablations):
            raise ValueError("required_ablations must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_seeds": list(self.required_seeds),
            "maximum_mean_nmae": self.maximum_mean_nmae,
            "minimum_mean_obo": self.minimum_mean_obo,
            "minimum_individual_passes": self.minimum_individual_passes,
            "required_ablations": list(self.required_ablations),
        }


TABLE2_ABLATIONS = (
    "baseline",
    "pams_no_multi_scale",
    "pams_multi_scale",
    "multi_expert_no_multi_expert",
    "multi_expert",
    "full",
)

FROZEN_GATE = VerificationGate(
    required_seeds=(42, 2026, 3407),
    maximum_mean_nmae=0.228,
    minimum_mean_obo=0.666,
    minimum_individual_passes=2,
    required_ablations=TABLE2_ABLATIONS,
)


@dataclass(frozen=True, slots=True)
class VerificationCheck:
    """One gate check; ``None`` means evidence was unavailable."""

    key: str
    passed: bool | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True, slots=True)
class VerificationDecision:
    """Canonical result returned by :func:`evaluate_verification`."""

    status: VerificationStatus
    mean_metrics: MetricPair | None
    population_std_metrics: MetricPair | None
    individual_passes: int
    checks: tuple[VerificationCheck, ...]
    missing_seeds: tuple[int, ...]
    missing_ablations: tuple[str, ...]

    @property
    def verified(self) -> bool:
        return self.status is VerificationStatus.VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "mean_metrics": (None if self.mean_metrics is None else self.mean_metrics.to_dict()),
            "population_std_metrics": (
                None
                if self.population_std_metrics is None
                else self.population_std_metrics.to_dict()
            ),
            "individual_passes": self.individual_passes,
            "missing_seeds": list(self.missing_seeds),
            "missing_ablations": list(self.missing_ablations),
            "checks": [check.to_dict() for check in self.checks],
        }


def _strictly_better(candidate: MetricPair, reference: MetricPair) -> bool:
    return candidate.nmae < reference.nmae and candidate.obo > reference.obo


def _comparison_check(
    *,
    evidence: VerificationEvidence,
    key: str,
    candidate_key: str,
    reference_key: str,
) -> VerificationCheck:
    candidate = evidence.ablations.get(candidate_key)
    reference = evidence.ablations.get(reference_key)
    if candidate is None or reference is None:
        return VerificationCheck(
            key=key,
            passed=None,
            detail=f"requires ablations {candidate_key!r} and {reference_key!r}",
        )
    passed = _strictly_better(candidate, reference)
    return VerificationCheck(
        key=key,
        passed=passed,
        detail=(
            f"{candidate_key} must have strictly lower NMAE and higher OBO than {reference_key}"
        ),
    )


def evaluate_verification(evidence: VerificationEvidence) -> VerificationDecision:
    """Apply the repository's fixed verification gate.

    Unknown seeds and ablation keys are rejected. This prevents a result bundle
    from silently replacing a preregistered run or comparison.
    """

    required_seeds = set(FROZEN_GATE.required_seeds)
    supplied_seeds = {result.seed for result in evidence.seed_results}
    unknown_seeds = supplied_seeds - required_seeds
    if unknown_seeds:
        raise ValueError(f"results contain non-preregistered seeds: {sorted(unknown_seeds)}")

    required_ablations = set(FROZEN_GATE.required_ablations)
    unknown_ablations = set(evidence.ablations) - required_ablations
    if unknown_ablations:
        raise ValueError(
            f"results contain non-preregistered ablations: {sorted(unknown_ablations)}"
        )

    missing_seeds = tuple(seed for seed in FROZEN_GATE.required_seeds if seed not in supplied_seeds)
    missing_ablations = tuple(
        key for key in FROZEN_GATE.required_ablations if key not in evidence.ablations
    )
    seeds_complete = not missing_seeds
    ablations_complete = not missing_ablations

    result_by_seed = {result.seed: result for result in evidence.seed_results}
    individual_passes = sum(
        result.metrics.nmae <= FROZEN_GATE.maximum_mean_nmae
        and result.metrics.obo >= FROZEN_GATE.minimum_mean_obo
        for result in evidence.seed_results
    )

    mean_metrics: MetricPair | None = None
    population_std_metrics: MetricPair | None = None
    if seeds_complete:
        ordered = [result_by_seed[seed].metrics for seed in FROZEN_GATE.required_seeds]
        mean_metrics = MetricPair(
            nmae=sum(metrics.nmae for metrics in ordered) / len(ordered),
            obo=sum(metrics.obo for metrics in ordered) / len(ordered),
        )
        population_std_metrics = MetricPair(
            nmae=math.sqrt(
                sum((metrics.nmae - mean_metrics.nmae) ** 2 for metrics in ordered)
                / len(ordered)
            ),
            obo=math.sqrt(
                sum((metrics.obo - mean_metrics.obo) ** 2 for metrics in ordered)
                / len(ordered)
            ),
        )

    checks: list[VerificationCheck] = [
        VerificationCheck(
            key="required_seeds_complete",
            passed=seeds_complete,
            detail=f"required seeds are {FROZEN_GATE.required_seeds}",
        ),
        VerificationCheck(
            key="mean_threshold",
            passed=(
                None
                if mean_metrics is None
                else mean_metrics.nmae <= FROZEN_GATE.maximum_mean_nmae
                and mean_metrics.obo >= FROZEN_GATE.minimum_mean_obo
            ),
            detail=(
                f"mean NMAE <= {FROZEN_GATE.maximum_mean_nmae:.3f} and "
                f"mean OBO >= {FROZEN_GATE.minimum_mean_obo:.3f}"
            ),
        ),
        VerificationCheck(
            key="individual_seed_threshold",
            passed=(
                None
                if not seeds_complete
                else individual_passes >= FROZEN_GATE.minimum_individual_passes
            ),
            detail=(
                f"at least {FROZEN_GATE.minimum_individual_passes}/"
                f"{len(FROZEN_GATE.required_seeds)} seeds meet both thresholds"
            ),
        ),
        VerificationCheck(
            key="required_ablations_complete",
            passed=ablations_complete,
            detail=f"required rows are {FROZEN_GATE.required_ablations}",
        ),
        _comparison_check(
            evidence=evidence,
            key="multiscale_improves",
            candidate_key="pams_multi_scale",
            reference_key="pams_no_multi_scale",
        ),
        _comparison_check(
            evidence=evidence,
            key="multiexpert_improves",
            candidate_key="multi_expert",
            reference_key="multi_expert_no_multi_expert",
        ),
    ]

    if ablations_complete:
        full = evidence.ablations["full"]
        full_best = all(
            _strictly_better(full, evidence.ablations[key])
            for key in FROZEN_GATE.required_ablations
            if key != "full"
        )
        checks.append(
            VerificationCheck(
                key="full_is_best",
                passed=full_best,
                detail="full must have strictly lower NMAE and higher OBO than every ablation",
            )
        )
    else:
        checks.append(
            VerificationCheck(
                key="full_is_best",
                passed=None,
                detail="requires every preregistered Table 2 ablation row",
            )
        )

    if not evidence.has_any_results:
        status = VerificationStatus.NO_RESULTS
    elif not (seeds_complete and ablations_complete):
        status = VerificationStatus.UNVERIFIED
    elif all(check.passed is True for check in checks):
        status = VerificationStatus.VERIFIED
    else:
        status = VerificationStatus.PARTIAL

    return VerificationDecision(
        status=status,
        mean_metrics=mean_metrics,
        population_std_metrics=population_std_metrics,
        individual_passes=individual_passes,
        checks=tuple(checks),
        missing_seeds=missing_seeds,
        missing_ablations=missing_ablations,
    )


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


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value
