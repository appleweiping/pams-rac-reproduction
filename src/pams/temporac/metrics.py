"""Fail-closed desensitized count metrics and paired component bootstrap."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from pams.temporac.contract import SEEDS
from pams.temporac.types import ContractError

Arm = Literal["local", "global", "uniform", "capacity-control"]
Condition = Literal["natural-clean", "natural-drift"]

ARMS: tuple[Arm, ...] = ("local", "global", "uniform", "capacity-control")
COMPARATOR_TIE_ORDER: tuple[Arm, ...] = ("global", "uniform", "capacity-control")
CONDITIONS: tuple[Condition, ...] = ("natural-clean", "natural-drift")
ROUTE_BOOTSTRAP_DRAWS = 10_000
ROUTE_BOOTSTRAP_SEED = int.from_bytes(
    hashlib.sha256(b"temporac.k7.route-and-clean.v4").digest()[:8], "little"
)
ROW_KEYS = frozenset(
    {
        "arm",
        "component_token",
        "condition",
        "count_gt",
        "estimate",
        "seed",
        "slot",
        "video_token",
    }
)


class MetricError(ContractError):
    """Any metric input attack fails the complete scorer globally."""


def _token(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or not value.isascii() or "\x00" in value:
        raise MetricError(f"{name} must be a nonempty NUL-free opaque ASCII token")
    return value


def _exact_int(value: object, *, name: str, minimum: int) -> int:
    if type(value) is not int or cast(int, value) < minimum:
        raise MetricError(f"{name} must be an integer >= {minimum}")
    return cast(int, value)


@dataclass(frozen=True, slots=True)
class CountMetricRow:
    """One desensitized arm/seed/condition/person count observation."""

    arm: Arm
    seed: int
    condition: Condition
    video_token: str
    slot: int
    component_token: str
    count_gt: int
    estimate: int

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise MetricError("metric arm is outside the exact four-arm inventory")
        if type(self.seed) is not int or self.seed not in SEEDS:
            raise MetricError("metric seed is outside the exact ordered three-seed inventory")
        if self.condition not in CONDITIONS:
            raise MetricError("metric condition must be natural-clean or natural-drift")
        _token(self.video_token, name="video_token")
        _token(self.component_token, name="component_token")
        _exact_int(self.slot, name="slot", minimum=0)
        _exact_int(self.count_gt, name="count_gt", minimum=1)
        _exact_int(self.estimate, name="estimate", minimum=0)

    @property
    def identity(self) -> tuple[str, int]:
        return self.video_token, self.slot

    @property
    def nae(self) -> float:
        return float(abs(self.estimate - self.count_gt) / self.count_gt)

    @property
    def obo(self) -> float:
        return float(abs(self.estimate - self.count_gt) <= 1)


def parse_metric_rows(payload: Sequence[Mapping[str, object]]) -> tuple[CountMetricRow, ...]:
    """Parse all rows or fail without returning a partial population."""

    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)) or not payload:
        raise MetricError("metric input must be a nonempty row sequence")
    parsed: list[CountMetricRow] = []
    for index, row in enumerate(payload):
        if not isinstance(row, Mapping) or set(row) != ROW_KEYS:
            raise MetricError(f"metric row {index} has unknown or missing keys")
        parsed.append(
            CountMetricRow(
                arm=cast(Arm, row["arm"]),
                seed=_exact_int(row["seed"], name="seed", minimum=0),
                condition=cast(Condition, row["condition"]),
                video_token=_token(row["video_token"], name="video_token"),
                slot=_exact_int(row["slot"], name="slot", minimum=0),
                component_token=_token(row["component_token"], name="component_token"),
                count_gt=_exact_int(row["count_gt"], name="count_gt", minimum=1),
                estimate=_exact_int(row["estimate"], name="estimate", minimum=0),
            )
        )
    return tuple(parsed)


@dataclass(frozen=True, slots=True)
class MetricPoint:
    """Video-first, then ordered-seed point estimands for all arms/conditions."""

    avg_mae: Mapping[tuple[Arm, Condition], float]
    avg_obo: Mapping[tuple[Arm, Condition], float]
    strongest_comparator: Arm
    route_margin: float
    clean_delta: float

    def __post_init__(self) -> None:
        expected = {(arm, condition) for arm in ARMS for condition in CONDITIONS}
        if set(self.avg_mae) != expected or set(self.avg_obo) != expected:
            raise MetricError("metric point mappings are incomplete")
        for mapping in (self.avg_mae, self.avg_obo):
            if any(
                type(value) is not float or not math.isfinite(value) for value in mapping.values()
            ):
                raise MetricError("metric point contains a nonfinite or non-float value")
        if self.strongest_comparator not in COMPARATOR_TIE_ORDER:
            raise MetricError("point strongest comparator is invalid")
        if any(
            type(value) is not float or not math.isfinite(value)
            for value in (self.route_margin, self.clean_delta)
        ):
            raise MetricError("route margin and clean delta must be finite float64")
        object.__setattr__(self, "avg_mae", MappingProxyType(dict(self.avg_mae)))
        object.__setattr__(self, "avg_obo", MappingProxyType(dict(self.avg_obo)))


@dataclass(frozen=True, slots=True)
class _ValidatedPopulation:
    rows: tuple[CountMetricRow, ...]
    identities: tuple[tuple[str, int], ...]
    videos: tuple[str, ...]
    components: tuple[str, ...]
    video_component: Mapping[str, str]
    index: Mapping[tuple[Arm, int, Condition, str, int], CountMetricRow]


def _validate_population(rows: Sequence[CountMetricRow]) -> _ValidatedPopulation:
    if not rows:
        raise MetricError("metric population is empty")
    index: dict[tuple[Arm, int, Condition, str, int], CountMetricRow] = {}
    identity_meta: dict[tuple[str, int], tuple[str, int]] = {}
    video_component: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, CountMetricRow):
            raise MetricError("metric population contains a non-CountMetricRow value")
        key = (row.arm, row.seed, row.condition, row.video_token, row.slot)
        if key in index:
            raise MetricError("metric population contains a duplicate person observation")
        index[key] = row
        identity = row.identity
        metadata = (row.component_token, row.count_gt)
        if identity in identity_meta and identity_meta[identity] != metadata:
            raise MetricError(
                "person component or positive ground truth changes across observations"
            )
        identity_meta[identity] = metadata
        previous_component = video_component.setdefault(row.video_token, row.component_token)
        if previous_component != row.component_token:
            raise MetricError("one opaque video belongs to more than one source component")
    identities = tuple(
        sorted(identity_meta, key=lambda value: (value[0].encode("ascii"), value[1]))
    )
    expected_count = len(identities) * len(ARMS) * len(SEEDS) * len(CONDITIONS)
    if len(index) != expected_count:
        raise MetricError("metric population has missing arm/seed/condition/person values")
    for identity in identities:
        for arm in ARMS:
            for seed in SEEDS:
                for condition in CONDITIONS:
                    if (arm, seed, condition, identity[0], identity[1]) not in index:
                        raise MetricError("metric population cross-product is incomplete")
    videos = tuple(sorted(video_component, key=lambda value: value.encode("ascii")))
    if not videos:
        raise MetricError("metric population has an empty video level")
    components = tuple(
        sorted(set(video_component.values()), key=lambda value: value.encode("ascii"))
    )
    if not components:
        raise MetricError("metric population has an empty component level")
    return _ValidatedPopulation(
        rows=tuple(rows),
        identities=identities,
        videos=videos,
        components=components,
        video_component=MappingProxyType(video_component),
        index=MappingProxyType(index),
    )


def _video_metric(
    population: _ValidatedPopulation,
    *,
    arm: Arm,
    seed: int,
    condition: Condition,
    video: str,
    metric: Literal["nae", "obo"],
) -> float:
    slots = tuple(slot for candidate, slot in population.identities if candidate == video)
    if not slots:
        raise MetricError("video has an empty person level")
    values = [
        getattr(population.index[(arm, seed, condition, video, slot)], metric) for slot in slots
    ]
    if any(not math.isfinite(value) for value in values):
        raise MetricError("person metric is nonfinite")
    return float(math.fsum(values) / len(values))


def _aggregate(
    population: _ValidatedPopulation,
    *,
    arm: Arm,
    condition: Condition,
    metric: Literal["nae", "obo"],
    video_multiplicity: Mapping[str, int] | None = None,
) -> float:
    per_seed: list[float] = []
    for seed in SEEDS:
        weighted: list[float] = []
        denominator = 0
        for video in population.videos:
            multiplicity = 1 if video_multiplicity is None else video_multiplicity.get(video, 0)
            if type(multiplicity) is not int or multiplicity < 0:
                raise MetricError("video bootstrap multiplicity is invalid")
            if multiplicity:
                value = _video_metric(
                    population,
                    arm=arm,
                    seed=seed,
                    condition=condition,
                    video=video,
                    metric=metric,
                )
                weighted.extend([value] * multiplicity)
                denominator += multiplicity
        if denominator == 0:
            raise MetricError("metric aggregation has an empty replicated video level")
        per_seed.append(float(math.fsum(weighted) / denominator))
    return float(math.fsum(per_seed) / len(SEEDS))


def score_count_metrics(rows: Sequence[CountMetricRow]) -> MetricPoint:
    """Compute F20/F21 point estimands with no filtering or fallback."""

    population = _validate_population(rows)
    mae: dict[tuple[Arm, Condition], float] = {}
    obo: dict[tuple[Arm, Condition], float] = {}
    for arm in ARMS:
        for condition in CONDITIONS:
            mae[(arm, condition)] = _aggregate(
                population, arm=arm, condition=condition, metric="nae"
            )
            obo[(arm, condition)] = _aggregate(
                population, arm=arm, condition=condition, metric="obo"
            )
    strongest = min(
        COMPARATOR_TIE_ORDER,
        key=lambda arm: (mae[(arm, "natural-drift")], COMPARATOR_TIE_ORDER.index(arm)),
    )
    route = mae[(strongest, "natural-drift")] - mae[("local", "natural-drift")]
    clean = mae[("local", "natural-clean")] - mae[(strongest, "natural-clean")]
    return MetricPoint(mae, obo, strongest, float(route), float(clean))


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    route_lower: float
    route_upper: float
    clean_upper: float
    comparator_draw_counts: Mapping[Arm, int]
    route_draws: NDArray[np.float64]
    clean_draws: NDArray[np.float64]

    def __post_init__(self) -> None:
        for name in ("route_lower", "route_upper", "clean_upper"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value):
                raise MetricError(f"{name} must be finite float64")
        expected = set(COMPARATOR_TIE_ORDER)
        if (
            set(self.comparator_draw_counts) != expected
            or sum(self.comparator_draw_counts.values()) != ROUTE_BOOTSTRAP_DRAWS
        ):
            raise MetricError("bootstrap comparator counts are incomplete")
        for name in ("route_draws", "clean_draws"):
            array = np.asarray(getattr(self, name))
            if array.dtype.str != "<f8" or array.shape != (ROUTE_BOOTSTRAP_DRAWS,):
                raise MetricError(f"{name} must be <f8[10000]")
            if not np.isfinite(array).all():
                raise MetricError(f"{name} contains nonfinite values")
            copied = np.array(array, dtype="<f8", order="C", copy=True)
            copied.flags.writeable = False
            object.__setattr__(self, name, copied)
        object.__setattr__(
            self, "comparator_draw_counts", MappingProxyType(dict(self.comparator_draw_counts))
        )


def paired_component_bootstrap(rows: Sequence[CountMetricRow]) -> BootstrapResult:
    """Run the exact shared 10,000-draw route-and-clean PCG64 stream."""

    population = _validate_population(rows)
    if len(population.components) < 8:
        raise MetricError("paired bootstrap requires at least eight development components")
    component_index = {component: index for index, component in enumerate(population.components)}
    video_component_index = np.asarray(
        [component_index[population.video_component[video]] for video in population.videos],
        dtype=np.int64,
    )
    if set(video_component_index.tolist()) != set(range(len(population.components))):
        raise MetricError("bootstrap component has no opaque video")
    video_values: dict[tuple[Arm, Condition, int], NDArray[np.float64]] = {}
    for arm in ARMS:
        for condition in CONDITIONS:
            for seed in SEEDS:
                video_values[(arm, condition, seed)] = np.asarray(
                    [
                        _video_metric(
                            population,
                            arm=arm,
                            seed=seed,
                            condition=condition,
                            video=video,
                            metric="nae",
                        )
                        for video in population.videos
                    ],
                    dtype="<f8",
                )

    def draw_aggregate(arm: Arm, condition: Condition, multiplicity: NDArray[np.int64]) -> float:
        denominator = int(np.sum(multiplicity, dtype=np.int64))
        if denominator <= 0:
            raise MetricError("metric aggregation has an empty replicated video level")
        per_seed = [
            float(
                np.sum(
                    video_values[(arm, condition, seed)] * multiplicity,
                    dtype=np.float64,
                )
                / denominator
            )
            for seed in SEEDS
        ]
        return float(math.fsum(per_seed) / len(SEEDS))

    generator = np.random.Generator(np.random.PCG64(ROUTE_BOOTSTRAP_SEED))
    route = np.empty(ROUTE_BOOTSTRAP_DRAWS, dtype="<f8")
    clean = np.empty(ROUTE_BOOTSTRAP_DRAWS, dtype="<f8")
    selected: Counter[Arm] = Counter()
    component_count = len(population.components)
    for draw in range(ROUTE_BOOTSTRAP_DRAWS):
        indices = generator.integers(0, component_count, size=component_count, dtype=np.int64)
        counts = np.bincount(indices, minlength=component_count)
        multiplicity = np.asarray(counts[video_component_index], dtype=np.int64)
        drift = {arm: draw_aggregate(arm, "natural-drift", multiplicity) for arm in ARMS}
        strongest = min(
            COMPARATOR_TIE_ORDER,
            key=lambda arm: (drift[arm], COMPARATOR_TIE_ORDER.index(arm)),
        )
        selected[strongest] += 1
        route[draw] = drift[strongest] - drift["local"]
        local_clean = draw_aggregate("local", "natural-clean", multiplicity)
        selected_clean = draw_aggregate(strongest, "natural-clean", multiplicity)
        clean[draw] = local_clean - selected_clean
    route_sorted = np.sort(route)
    clean_sorted = np.sort(clean)
    counts_payload = {arm: int(selected[arm]) for arm in COMPARATOR_TIE_ORDER}
    return BootstrapResult(
        route_lower=float(route_sorted[249]),
        route_upper=float(route_sorted[9749]),
        clean_upper=float(clean_sorted[9749]),
        comparator_draw_counts=counts_payload,
        route_draws=route,
        clean_draws=clean,
    )


@dataclass(frozen=True, slots=True)
class K7Decision:
    status: Literal["PASS", "FAIL"]
    failures: tuple[str, ...]


def decide_k7(point: MetricPoint, bootstrap: BootstrapResult) -> K7Decision:
    """Apply the fixed route and paired-clean rules once, without fallback."""

    failures: list[str] = []
    comparator = point.strongest_comparator
    drift_n = point.avg_mae[(comparator, "natural-drift")]
    if drift_n == 0.0 or point.route_margin / drift_n < 0.05:
        failures.append("DRIFT_RELATIVE")
    if bootstrap.route_lower <= 0.0:
        failures.append("DRIFT_LOWER")
    if point.clean_delta > 0.01:
        failures.append("CLEAN_POINT")
    if bootstrap.clean_upper > 0.02:
        failures.append("CLEAN_UPPER")
    clean_n = point.avg_mae[(comparator, "natural-clean")]
    clean_local = point.avg_mae[("local", "natural-clean")]
    if clean_n > 0.0:
        if clean_local / clean_n > 1.05:
            failures.append("CLEAN_RATIO")
    elif clean_local != 0.0:
        failures.append("CLEAN_ZERO")
    return K7Decision("FAIL" if failures else "PASS", tuple(failures))
