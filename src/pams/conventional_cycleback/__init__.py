"""Independent conventional cycle-back TCC proxy.

This namespace is deliberately separate from :mod:`pams.losses` and the
historical fixed-period PAMS proxy.  The PAMS paper names a conventional TCC
baseline but does not disclose enough implementation detail to reconstruct it
uniquely.  Nothing exported here is an author implementation or eligible for
a paper-table claim.
"""

from pams.conventional_cycleback.config import (
    ConventionalCycleBackConfig,
    load_conventional_cycleback_config,
)
from pams.conventional_cycleback.loss import (
    ConventionalCycleBackLoss,
    CycleBackLossOutput,
)
from pams.conventional_cycleback.windows import (
    NativeWindowPairBatch,
    enumerate_native_window_pairs,
)

__all__ = [
    "ConventionalCycleBackConfig",
    "ConventionalCycleBackLoss",
    "CycleBackLossOutput",
    "NativeWindowPairBatch",
    "enumerate_native_window_pairs",
    "load_conventional_cycleback_config",
]
