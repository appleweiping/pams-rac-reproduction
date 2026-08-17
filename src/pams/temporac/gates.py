"""Kill-only P-stage/F23 receipt state for the unauthorized local wave.

The machine can attest P0, P1, and both P2 receipts.  P3 is deliberately an
external pending prerequisite, so this module cannot mark S0 PASS, request a
capability, or authorize a launch.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from pams.temporac.contract import F23_ORDER
from pams.temporac.types import ContractError


class GateStatus(str, Enum):
    PENDING = "PENDING"
    PASS = "PASS"
    FAIL = "FAIL"
    BLOCKED = "BLOCKED"


P_STAGE_ORDER = ("P0", "P1", "P2-FIXTURE", "P2-METRIC", "P3", "S0")
CAMPAIGN_GATE_ORDER = F23_ORDER[1:]
GATE_ORDER = P_STAGE_ORDER + CAMPAIGN_GATE_ORDER
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class GateBlockedError(ContractError):
    """A missing receipt, failed predecessor, or authority boundary blocked progress."""


def _receipt_digest(value: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ContractError("gate receipt must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class GateRecord:
    name: str
    status: GateStatus
    receipt_sha256: tuple[str, ...] = ()
    detail: str = ""

    def __post_init__(self) -> None:
        if self.name not in GATE_ORDER:
            raise ContractError("gate record name is outside P-stage/F23 order")
        if not isinstance(self.status, GateStatus):
            raise ContractError("gate record status must be a GateStatus")
        for digest in self.receipt_sha256:
            _receipt_digest(digest)
        if self.status is GateStatus.PASS and not self.receipt_sha256:
            raise ContractError("a passing gate requires at least one receipt digest")
        if self.status in {GateStatus.PENDING, GateStatus.BLOCKED} and self.receipt_sha256:
            raise ContractError("pending/blocked gates cannot claim a receipt")


class GateMachine:
    """Mutable kill-only coordinator whose authority ends after P2."""

    def __init__(self) -> None:
        self._records: dict[str, GateRecord] = {
            name: GateRecord(name, GateStatus.PENDING) for name in GATE_ORDER
        }
        self._killed = False

    @property
    def launch_authorized(self) -> bool:
        return False

    @property
    def capability_request_allowed(self) -> bool:
        return False

    @property
    def killed(self) -> bool:
        return self._killed

    @property
    def records(self) -> Mapping[str, GateRecord]:
        return MappingProxyType(dict(self._records))

    @property
    def next_gate(self) -> str | None:
        for name in GATE_ORDER:
            if self._records[name].status is GateStatus.PENDING:
                return name
        return None

    def _require_live_current(self, name: str) -> None:
        if self._killed:
            raise GateBlockedError("a failed gate killed the complete campaign")
        if name not in GATE_ORDER or self.next_gate != name:
            raise GateBlockedError("gate transition is not the next exact precedence item")

    def _pass(self, name: str, receipts: tuple[str, ...], detail: str = "") -> None:
        self._require_live_current(name)
        normalized = tuple(_receipt_digest(receipt) for receipt in receipts)
        self._records[name] = GateRecord(name, GateStatus.PASS, normalized, detail)

    def pass_p0(self, implementation_receipt_sha256: str) -> None:
        self._pass("P0", (implementation_receipt_sha256,))

    def pass_p1(self, static_receipt_sha256: str) -> None:
        self._pass("P1", (static_receipt_sha256,))

    def pass_p2(
        self,
        fixture_receipt_sha256: str,
        metric_receipt_sha256: str,
    ) -> None:
        """Atomically attest the separately required P2 fixture/metric receipts."""

        self._require_live_current("P2-FIXTURE")
        fixture = _receipt_digest(fixture_receipt_sha256)
        metric = _receipt_digest(metric_receipt_sha256)
        self._records["P2-FIXTURE"] = GateRecord("P2-FIXTURE", GateStatus.PASS, (fixture,))
        self._records["P2-METRIC"] = GateRecord("P2-METRIC", GateStatus.PASS, (metric,))

    def fail_current(self, name: str, receipt_sha256: str, *, detail: str) -> None:
        self._require_live_current(name)
        if not detail:
            raise ContractError("a failed gate requires a nonempty detail")
        self._records[name] = GateRecord(
            name,
            GateStatus.FAIL,
            (_receipt_digest(receipt_sha256),),
            detail,
        )
        failed_index = GATE_ORDER.index(name)
        for later in GATE_ORDER[failed_index + 1 :]:
            self._records[later] = GateRecord(
                later,
                GateStatus.BLOCKED,
                detail=f"blocked by {name}",
            )
        self._killed = True

    def request_s0_pass(self, s0_candidate_receipt_sha256: str) -> None:
        """Always refuse: P3 is outside this wave and remains PENDING."""

        _receipt_digest(s0_candidate_receipt_sha256)
        if self._records["P3"].status is not GateStatus.PASS:
            raise GateBlockedError("P3 is PENDING; S0 PASS is forbidden")
        raise GateBlockedError("this local state machine has no S0 commitment authority")

    def record_campaign_gate(self, name: str, receipt_sha256: str) -> None:
        """Always refuse G/K transitions because this wave cannot pass S0."""

        if name not in CAMPAIGN_GATE_ORDER:
            raise ContractError("campaign gate name is outside F23")
        _receipt_digest(receipt_sha256)
        raise GateBlockedError("S0 is not PASS; no F23 G/K receipt may advance")


__all__ = [
    "CAMPAIGN_GATE_ORDER",
    "GATE_ORDER",
    "P_STAGE_ORDER",
    "GateBlockedError",
    "GateMachine",
    "GateRecord",
    "GateStatus",
]
