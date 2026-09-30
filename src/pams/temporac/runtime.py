"""Exact runtime, RNG, initialization, schedule, and job records for v4."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from types import MappingProxyType
from typing import TypeVar, cast

import numpy as np
import torch
from numpy.typing import NDArray

from pams.temporac.contract import (
    CANONICAL_JOB_NAMES,
    CAPACITY_CONTROL_JOB_NAMES,
    CUDA_BYTE_CEILING,
    JOB_NAMES,
    MAX_CONCURRENCY,
    SEEDS,
    SHORTCUT_JOB_NAMES,
    TEACHER_JOB_NAMES,
)
from pams.temporac.hashio import H, uint16_be, uint32_be, uint64_be
from pams.temporac.types import CheckpointRecord, ContractError, ResourceRecord


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """The singular accepted neural runtime and deterministic flags."""

    operating_system: str = "Linux"
    architecture: str = "x86_64"
    cpython: str = "3.12.4"
    numpy: str = "2.1.0"
    scipy: str = "1.14.1"
    pytorch: str = "2.4.1+cu124"
    cuda_runtime: str = "12.4"
    cudnn: str = "9.1.0.70"
    nvidia_driver: str = "550.54.14"
    gpu: str = "RTX A6000"
    model_optimizer_dtype: str = "float32"
    reduction_dtype: str = "float64"
    autocast: bool = False
    grad_scaler: bool = False
    matmul_tf32: bool = False
    cudnn_tf32: bool = False
    deterministic_algorithms: bool = True
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False
    cublas_workspace_config: str = ":4096:8"
    matmul_precision: str = "highest"
    pythonhashseed: str = "0"
    processes_per_gpu: int = 1
    loader_workers: int = 0
    max_concurrency: int = MAX_CONCURRENCY

    def __post_init__(self) -> None:
        observed = tuple(asdict(self).values())
        expected = (
            "Linux",
            "x86_64",
            "3.12.4",
            "2.1.0",
            "1.14.1",
            "2.4.1+cu124",
            "12.4",
            "9.1.0.70",
            "550.54.14",
            "RTX A6000",
            "float32",
            "float64",
            False,
            False,
            False,
            False,
            True,
            True,
            False,
            ":4096:8",
            "highest",
            "0",
            1,
            0,
            2,
        )
        if observed != expected:
            raise ContractError("runtime config differs from the singular v4 runtime")

    def as_dict(self) -> dict[str, str | bool | int]:
        return cast(dict[str, str | bool | int], asdict(self))


RUNTIME_CONFIG = RuntimeConfig()
RUNTIME_CONFIG_KEYS = frozenset(RUNTIME_CONFIG.as_dict())


def validate_runtime_config(payload: Mapping[str, object]) -> RuntimeConfig:
    """Reject unknown, missing, mistyped, or changed runtime fields."""

    if not isinstance(payload, Mapping) or set(payload) != RUNTIME_CONFIG_KEYS:
        raise ContractError("runtime config has unknown or missing keys")
    if dict(payload) != RUNTIME_CONFIG.as_dict():
        raise ContractError("runtime config values differ from v4")
    return RUNTIME_CONFIG


class RngDomain(str, Enum):
    """Frozen length-prefixed PCG64 selection domains."""

    TEACHER_SOURCE_CYCLE = "temporac.teacher-source-cycle.v4"
    RESPONSE_X0_CYCLE = "temporac.response-x0-cycle.v4"
    RESPONSE_COMPONENT_CYCLE = "temporac.response-component-cycle.v4"
    RESPONSE_IDENTITY_CYCLE = "temporac.response-identity-cycle.v4"
    TORCH_CPU = "temporac.torch-cpu.v4"
    TORCH_CUDA = "temporac.torch-cuda.v4"


def rng_seed(
    domain: RngDomain,
    *fields: bytes | str,
) -> int:
    """Use the first eight little-endian SHA bytes as the isolated RNG seed."""

    if not isinstance(domain, RngDomain):
        raise ContractError("RNG domain must be a frozen RngDomain value")
    return int.from_bytes(H(domain.value, *fields)[:8], "little", signed=False)


def torch_generator(
    domain: RngDomain,
    *,
    job_name: str,
    seed: int,
    fields: tuple[bytes | str, ...] = (),
    device: str = "cpu",
) -> torch.Generator:
    """Create an isolated Torch CPU/CUDA generator from its own H domain."""

    if job_name not in JOB_NAMES or seed not in SEEDS:
        raise ContractError("Torch RNG requires a frozen job name and seed")
    expected_domain = RngDomain.TORCH_CPU if device == "cpu" else RngDomain.TORCH_CUDA
    if device not in {"cpu", "cuda"} or domain is not expected_domain:
        raise ContractError("Torch RNG domain and device must be the matching CPU/CUDA pair")
    derived = rng_seed(domain, job_name, uint64_be(seed), *fields)
    generator = torch.Generator(device=device)
    generator.manual_seed(derived)
    return generator


def cycle_permutation(
    domain: RngDomain,
    *,
    job_name: str,
    seed: int,
    cycle: int,
    size: int,
    extra_fields: tuple[bytes | str, ...] = (),
) -> NDArray[np.int64]:
    """Make exactly one PCG64 permutation call on int64 ``0..size-1``."""

    if domain not in {
        RngDomain.TEACHER_SOURCE_CYCLE,
        RngDomain.RESPONSE_X0_CYCLE,
        RngDomain.RESPONSE_COMPONENT_CYCLE,
        RngDomain.RESPONSE_IDENTITY_CYCLE,
    }:
        raise ContractError("cycle permutation requires a selection RNG domain")
    if job_name not in JOB_NAMES:
        raise ContractError("cycle permutation job name is not in the frozen inventory")
    if seed not in SEEDS or type(cycle) is not int or cycle < 0:
        raise ContractError("cycle permutation has an invalid seed or cycle")
    if type(size) is not int or size <= 0:
        raise ContractError("cycle permutation size must be positive")
    teacher_domain = domain is RngDomain.TEACHER_SOURCE_CYCLE
    if teacher_domain != (job_name in TEACHER_JOB_NAMES):
        raise ContractError("selection RNG domain does not match the job family")
    if domain in {RngDomain.TEACHER_SOURCE_CYCLE, RngDomain.RESPONSE_X0_CYCLE} and size != 24:
        raise ContractError("X0 source cycles must permute exactly 24 source orbits")
    if domain is RngDomain.RESPONSE_IDENTITY_CYCLE:
        if (
            len(extra_fields) != 1
            or not isinstance(extra_fields[0], bytes)
            or len(extra_fields[0]) != 32
        ):
            raise ContractError("identity cycles require exactly one raw component key")
    elif extra_fields:
        raise ContractError("only identity cycles accept an additional component-key field")
    derived = rng_seed(
        domain,
        job_name,
        uint64_be(seed),
        uint32_be(cycle),
        *extra_fields,
    )
    generator = np.random.Generator(np.random.PCG64(derived))
    permutation = generator.permutation(np.arange(size, dtype=np.int64))
    if permutation.dtype.str != "<i8" or permutation.shape != (size,):
        raise ContractError("PCG64 did not return the required int64 permutation")
    permutation.flags.writeable = False
    return permutation


def initialization_seed(job_name: str, seed: int) -> int:
    """Derive the special non-H parameter-initialization seed."""

    if job_name not in JOB_NAMES or seed not in SEEDS:
        raise ContractError("initialization requires a frozen job name and seed")
    job_bytes = job_name.encode("ascii")
    if len(job_bytes) >= 2**16:
        raise ContractError("job name does not fit the uint16 length field")
    import hashlib

    digest = hashlib.sha256(
        b"temporac.init.v4\x00" + uint16_be(len(job_bytes)) + job_bytes + uint64_be(seed)
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _owning_module(module: torch.nn.Module, parameter_name: str) -> tuple[torch.nn.Module, str]:
    parts = parameter_name.split(".")
    owner = module
    for part in parts[:-1]:
        try:
            owner = owner.get_submodule(part)
        except AttributeError as exc:
            raise ContractError(f"cannot resolve parameter owner: {parameter_name}") from exc
    return owner, parts[-1]


def initialize_module_cpu(module: torch.nn.Module, *, job_name: str, seed: int) -> int:
    """Initialize all v4 trainable parameters on CPU in ASCII name order."""

    if not isinstance(module, torch.nn.Module):
        raise ContractError("CPU initialization requires a torch Module")
    derived_seed = initialization_seed(job_name, seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derived_seed)
    named = list(module.named_parameters())
    if [name for name, _ in named] != sorted(name for name, _ in named):
        named = sorted(named, key=lambda item: item[0].encode("ascii"))
    if len({name for name, _ in named}) != len(named) or any(
        not name.isascii() for name, _ in named
    ):
        raise ContractError("parameter names must be unique ASCII strings")
    with torch.no_grad():
        for name, parameter in named:
            if parameter.device.type != "cpu" or parameter.dtype != torch.float32:
                raise ContractError(
                    "model parameters must be float32 CPU tensors at initialization"
                )
            owner, leaf = _owning_module(module, name)
            if isinstance(owner, torch.nn.LayerNorm) and leaf in {"weight", "bias"}:
                if leaf == "weight":
                    parameter.fill_(1.0)
                else:
                    parameter.zero_()
                continue
            if isinstance(owner, torch.nn.Linear) and leaf in {"weight", "bias"}:
                fan_in = owner.in_features
            elif isinstance(owner, torch.nn.Conv1d) and leaf in {"weight", "bias"}:
                fan_in = owner.in_channels * owner.kernel_size[0] // owner.groups
            else:
                raise ContractError(f"unfrozen trainable parameter initializer: {name}")
            bound = 1.0 / math.sqrt(float(fan_in))
            parameter.uniform_(-bound, bound, generator=generator)
    return derived_seed


def configure_torch_determinism() -> None:
    """Apply the v4 flags before any model/optimizer execution."""

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = RUNTIME_CONFIG.cublas_workspace_config
    os.environ["PYTHONHASHSEED"] = RUNTIME_CONFIG.pythonhashseed
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")


@dataclass(frozen=True, slots=True)
class JobSpec:
    """One member of the exact 27-job fixed-step inventory."""

    name: str
    seed: int
    kind: str
    steps: int
    base_lr: float
    checkpoint_interval: int
    checkpoint_count: int
    gpu_hour_ceiling: float
    cuda_byte_ceiling: int

    def __post_init__(self) -> None:
        if self.name not in JOB_NAMES or self.seed not in SEEDS:
            raise ContractError("job spec name/seed is outside the frozen inventory")
        expected_teacher = self.name in TEACHER_JOB_NAMES
        expected = (
            ("teacher", 20_000, 3e-4, 40, 6.0)
            if expected_teacher
            else ("response", 10_000, 1e-3, 20, 2.5)
        )
        if (
            (self.kind, self.steps, self.base_lr, self.checkpoint_count, self.gpu_hour_ceiling)
            != expected
            or self.checkpoint_interval != 500
            or self.cuda_byte_ceiling != CUDA_BYTE_CEILING
        ):
            raise ContractError("job spec differs from the exact v4 schedule/resource contract")


def _make_job_spec(name: str) -> JobSpec:
    seed = int(name.rsplit("=", maxsplit=1)[1])
    teacher = name in TEACHER_JOB_NAMES
    return JobSpec(
        name=name,
        seed=seed,
        kind="teacher" if teacher else "response",
        steps=20_000 if teacher else 10_000,
        base_lr=3e-4 if teacher else 1e-3,
        checkpoint_interval=500,
        checkpoint_count=40 if teacher else 20,
        gpu_hour_ceiling=6.0 if teacher else 2.5,
        cuda_byte_ceiling=CUDA_BYTE_CEILING,
    )


JOB_SPECS: Mapping[str, JobSpec] = MappingProxyType(
    {name: _make_job_spec(name) for name in JOB_NAMES}
)

# These assertions are contract witnesses, not alternate inventory builders.
if set(CANONICAL_JOB_NAMES + CAPACITY_CONTROL_JOB_NAMES + SHORTCUT_JOB_NAMES) != set(
    JOB_NAMES
) - set(TEACHER_JOB_NAMES):  # pragma: no cover - import invariant
    raise RuntimeError("response job partition drift")


def learning_rate(completed_step: int, *, total_steps: int, base_lr: float) -> float:
    """Formula F22 evaluated in binary64 for completed step ``1..T``."""

    if type(completed_step) is not int or type(total_steps) is not int:
        raise ContractError("LR step values must be Python integers")
    if total_steps not in {10_000, 20_000} or not 1 <= completed_step <= total_steps:
        raise ContractError("LR step/horizon is outside the fixed schedule")
    expected_base = 3e-4 if total_steps == 20_000 else 1e-3
    if type(base_lr) is not float or base_lr != expected_base:
        raise ContractError("base LR does not match the fixed schedule horizon")
    warmup = max(1, math.ceil(0.05 * total_steps))
    if completed_step <= warmup:
        return float(base_lr * completed_step / warmup)
    phase = math.pi * (completed_step - warmup) / (total_steps - warmup)
    return float(base_lr * (0.1 + 0.9 * (1.0 + math.cos(phase)) / 2.0))


def job_learning_rate(job: JobSpec, completed_step: int) -> float:
    return learning_rate(completed_step, total_steps=job.steps, base_lr=job.base_lr)


def checkpoint_steps(job: JobSpec | int) -> tuple[int, ...]:
    total_steps = job.steps if isinstance(job, JobSpec) else job
    if total_steps not in {10_000, 20_000}:
        raise ContractError("checkpoint horizon must be a fixed teacher/response horizon")
    return tuple(range(500, total_steps + 1, 500))


def select_checkpoint(job: JobSpec, candidates: tuple[CheckpointRecord, ...]) -> CheckpointRecord:
    """Require every candidate and select finite minimum, tie smaller step."""

    expected = checkpoint_steps(job)
    observed = tuple(candidate.completed_step for candidate in candidates)
    if observed != expected:
        raise ContractError("checkpoint candidates are missing, extra, or out of order")
    if any(candidate.resource.max_cuda_bytes > job.cuda_byte_ceiling for candidate in candidates):
        raise ContractError("checkpoint exceeded the CUDA memory ceiling")
    return min(
        candidates, key=lambda candidate: (candidate.tune_objective, candidate.completed_step)
    )


def validate_final_resource(job: JobSpec, resource: ResourceRecord) -> None:
    """Check scheduled completion and the non-reassignable per-job ceilings."""

    if resource.completed_steps != job.steps:
        raise ContractError("job did not finish its exact fixed-step horizon")
    if resource.gpu_seconds > job.gpu_hour_ceiling * 3600.0:
        raise ContractError("job exceeded its A6000-hour ceiling")
    if resource.max_cuda_bytes > job.cuda_byte_ceiling:
        raise ContractError("job exceeded its 24-GiB CUDA ceiling")


T = TypeVar("T")


def require_exact_keys(payload: Mapping[str, T], expected: frozenset[str], *, name: str) -> None:
    """Shared fail-closed helper for config-like mappings."""

    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ContractError(f"{name} has unknown or missing keys")
