"""Auditable registry of paper baselines and diagnostic references."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from pams.baselines.base import (
    BaselineAdapter,
    BaselineSpec,
    BaselineStatus,
    BaselineUnavailableError,
    ImplementationKind,
    ProtocolRecord,
)
from pams.baselines.countllm_lite import countllm_lite_spec
from pams.baselines.proxy import SPECTRAL_PROXY_SPEC, SpectralProxyAdapter

UCFREP_526_FAIR = ProtocolRecord(
    key="ucfrep_526_fair",
    dataset="UCFRep-526",
    train_split="421 videos",
    evaluation_split="105 held-out videos",
    modality="method-specific; shared video IDs and metrics",
    source_faithful=False,
    fair_comparison=True,
    notes="No ground-truth count or action identity may be read at inference.",
)

UCFREP_POSE_110_FAIR = ProtocolRecord(
    key="ucfrep_pose_110_fair",
    dataset="UCFRep-pose-110 (five selected action classes)",
    train_split="89 videos",
    evaluation_split="21 held-out videos",
    modality="33x3 pose",
    source_faithful=False,
    fair_comparison=True,
    notes="Fair clean-room protocol; no test count is used to select an output.",
)

UCFREP_POSE_110_SOURCE = ProtocolRecord(
    key="ucfrep_pose_110_source",
    dataset="UCFRep-pose-110 (five selected action classes)",
    train_split="89 videos",
    evaluation_split="21 held-out videos",
    modality="33x3 pose",
    source_faithful=True,
    fair_comparison=False,
    notes="Source-faithful sanity protocol, not directly comparable to UCFRep-526.",
)

POSERAC_V1_ORACLE = ProtocolRecord(
    key="ucfrep_pose_110_poserac_v1_oracle",
    dataset="UCFRep-pose-110 (five selected action classes)",
    train_split="89 videos",
    evaluation_split="21 held-out videos",
    modality="33x3 pose",
    source_faithful=True,
    fair_comparison=False,
    ground_truth_count_at_inference=True,
    notes=(
        "The released evaluator selects the action-conditioned candidate with "
        "minimum error to the test ground-truth count; diagnostic only."
    ),
)


def _blocked(
    *,
    key: str,
    paper_name: str,
    protocols: tuple[ProtocolRecord, ...],
    source_urls: tuple[str, ...],
    reason: str,
    status: BaselineStatus = BaselineStatus.BLOCKED_UNIMPLEMENTED,
) -> BaselineSpec:
    return BaselineSpec(
        key=key,
        paper_name=paper_name,
        status=status,
        implementation=ImplementationKind.NOT_IMPLEMENTED,
        protocols=protocols,
        source_urls=source_urls,
        config_path=f"configs/baselines/{key.replace('-', '_')}.yaml",
        blocked_reason=reason,
    )


_STATIC_SPECS: tuple[BaselineSpec, ...] = (
    _blocked(
        key="repnet",
        paper_name="RepNet",
        protocols=(UCFREP_526_FAIR,),
        source_urls=(
            "https://sites.google.com/view/repnet",
            "https://github.com/materight/RepNet-pytorch",
        ),
        reason=(
            "Clean-room RGB architecture, converted weights, and full UCFRep-526 "
            "evaluation adapter have not yet passed parity tests."
        ),
    ),
    _blocked(
        key="transrac",
        paper_name="TransRAC",
        protocols=(UCFREP_526_FAIR,),
        source_urls=("https://github.com/SvipRepetitionCounting/TransRAC",),
        reason=(
            "Official-checkpoint modern-compatibility development sanity is "
            "complete, but original-protocol parity, a separately authored "
            "UCFRep-526 implementation, and sealed test evaluation remain blocked."
        ),
    ),
    _blocked(
        key="escounts",
        paper_name="Every Shot Counts (ESCounts)",
        protocols=(UCFREP_526_FAIR,),
        source_urls=("https://github.com/sinhasaptarshi/EveryShotCounts",),
        reason="The clean-room VideoMAE-token model and checkpoint parity test are incomplete.",
    ),
    _blocked(
        key="ivac-p2l",
        paper_name="IVAC-P2L",
        protocols=(UCFREP_526_FAIR,),
        source_urls=("https://github.com/hwang-cs-ime/IVAC-P2L",),
        reason=(
            "Only a source audit is registered; the fair UCFRep-526 zero-shot "
            "adapter and artifact checks are not implemented."
        ),
    ),
    _blocked(
        key="poserac-v1",
        paper_name="PoseRAC-v1 (released five-class pose protocol)",
        protocols=(POSERAC_V1_ORACLE, UCFREP_POSE_110_FAIR),
        source_urls=(
            "https://arxiv.org/abs/2303.08450",
            "https://github.com/MiracleDance/PoseRAC",
        ),
        reason=(
            "The released evaluator contains ground-truth-count selection. A fair "
            "action-agnostic replacement must be implemented and reported separately."
        ),
    ),
    _blocked(
        key="poserac-iconip24",
        paper_name="PoseRAC-ICONIP24 (standard-protocol target)",
        protocols=(UCFREP_526_FAIR,),
        source_urls=("https://github.com/MiracleDance/PoseRAC",),
        reason=(
            "The identifier and standard UCFRep-526 adaptation require a clean-room "
            "implementation and protocol verification; it is not aliased to PoseRAC-v1."
        ),
    ),
    _blocked(
        key="gmfl",
        paper_name="GMFL",
        protocols=(UCFREP_POSE_110_SOURCE, UCFREP_POSE_110_FAIR),
        source_urls=("https://arxiv.org/abs/2409.00330",),
        reason=(
            "No public implementation was located; coordinate, distance, angle, "
            "MIA, and bilinear branches remain to be independently implemented."
        ),
    ),
    _blocked(
        key="jtsps-count-only",
        paper_name="JTSPS-count-only",
        protocols=(UCFREP_526_FAIR,),
        source_urls=("https://doi.org/10.1109/TCSVT.2024.3402728",),
        reason=(
            "The public paper/assets do not uniquely determine a source-faithful "
            "training and evaluation protocol; status remains protocol-unverifiable."
        ),
        status=BaselineStatus.BLOCKED_PROTOCOL,
    ),
    _blocked(
        key="spkdb",
        paper_name="SPKDB",
        protocols=(UCFREP_POSE_110_SOURCE, UCFREP_POSE_110_FAIR),
        source_urls=("https://www.sciencedirect.com/science/article/pii/S1077314225001572",),
        reason=(
            "No public implementation was located; the global/salient-joint branches "
            "and fair test-time policy remain clean-room work."
        ),
    ),
    _blocked(
        key="bigc",
        paper_name="BIGC",
        protocols=(UCFREP_POSE_110_SOURCE, UCFREP_POSE_110_FAIR),
        source_urls=("https://doi.org/10.1016/j.engappai.2025.110996",),
        reason=(
            "No public implementation was located; inter-part and intra-part graph "
            "branches have not been independently implemented."
        ),
    ),
    SPECTRAL_PROXY_SPEC,
)


@dataclass(frozen=True, slots=True)
class _Registration:
    spec_provider: Callable[[], BaselineSpec]
    factory: Callable[[], BaselineAdapter] | None


def _constant_spec(spec: BaselineSpec) -> Callable[[], BaselineSpec]:
    def provide() -> BaselineSpec:
        return spec

    return provide


class BaselineRegistry:
    """Lookup service that never substitutes one method for another."""

    def __init__(self) -> None:
        self._entries: dict[str, _Registration] = {}
        for spec in _STATIC_SPECS:
            factory: Callable[[], BaselineAdapter] | None = None
            if spec.key == SPECTRAL_PROXY_SPEC.key:
                factory = SpectralProxyAdapter
            self._register(spec.key, _constant_spec(spec), factory)
        self._register("countllm-lite", countllm_lite_spec, None)

        self._aliases = {
            "every-shot-counts": "escounts",
            "es-counts": "escounts",
            "ivac_p2l": "ivac-p2l",
            "pose-rac-v1": "poserac-v1",
            "pose-rac-iconip24": "poserac-iconip24",
            "jtsps": "jtsps-count-only",
            "countllm_lite": "countllm-lite",
            "spectral_proxy": "spectral-proxy",
        }

    def _register(
        self,
        key: str,
        spec_provider: Callable[[], BaselineSpec],
        factory: Callable[[], BaselineAdapter] | None,
    ) -> None:
        if key in self._entries:
            raise ValueError(f"duplicate baseline key: {key}")
        self._entries[key] = _Registration(spec_provider=spec_provider, factory=factory)

    def canonical_key(self, name: str) -> str:
        key = name.strip().lower().replace(" ", "-")
        key = self._aliases.get(key, key)
        if key not in self._entries:
            choices = ", ".join(sorted(self._entries))
            raise KeyError(f"unknown baseline '{name}'; registered keys: {choices}")
        return key

    def get(self, name: str) -> BaselineSpec:
        """Return the current status, including dynamic resource gates."""

        key = self.canonical_key(name)
        return self._entries[key].spec_provider()

    def list(
        self,
        *,
        include_references: bool = True,
    ) -> tuple[BaselineSpec, ...]:
        specs = (entry.spec_provider() for entry in self._entries.values())
        selected: Iterable[BaselineSpec] = specs
        if not include_references:
            selected = (spec for spec in selected if spec.is_paper_baseline)
        return tuple(sorted(selected, key=lambda spec: spec.key))

    def create(self, name: str) -> BaselineAdapter:
        """Construct only genuinely runnable adapters."""

        key = self.canonical_key(name)
        registration = self._entries[key]
        spec = registration.spec_provider()
        spec.require_runnable()
        if registration.factory is None:
            raise BaselineUnavailableError(
                key=spec.key,
                status=spec.status,
                reason="no adapter factory is registered",
            )
        adapter = registration.factory()
        if adapter.spec.key != key:
            raise RuntimeError(
                f"factory registration mismatch: requested {key}, got {adapter.spec.key}"
            )
        return adapter


REGISTRY = BaselineRegistry()


def get_baseline_spec(name: str) -> BaselineSpec:
    return REGISTRY.get(name)


def list_baselines(*, include_references: bool = True) -> tuple[BaselineSpec, ...]:
    return REGISTRY.list(include_references=include_references)


def create_baseline(name: str) -> BaselineAdapter:
    return REGISTRY.create(name)
