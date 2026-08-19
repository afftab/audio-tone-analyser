"""Hard spend cap on the one paid component (the tone/intensity LLM call).

Every other stage runs locally at $0, so capping this call caps the whole
deployment's API bill. The cap exists because the dashboard is public-facing
for the evaluation period: anyone who can log in can queue batches, and
without a ceiling the only bound on spend is how much audio they upload.

Two properties matter and neither is free:

Survives restarts. The ledger is a small JSON file, not process state, so
restarting the app (or a LaunchAgent respawn) does not silently reset the
budget back to zero and hand out another cap's worth of spend.

Survives concurrency. LLM calls run LLM_CONCURRENCY-wide, so a plain
"check remaining, then call" would let every in-flight worker pass the same
check and collectively overshoot. Instead a worker reserves headroom before
calling and settles the reservation against the measured cost when the call
returns, which bounds the worst-case overshoot to what is already in flight.

Read-only from the dashboard by design: the cap comes from the environment
(VTA_SPEND_CAP_USD), so an evaluator clicking through the UI cannot raise it.
"""

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

# Total USD of API spend this deployment may ever incur. Deliberately small:
# the whole 3-clip demo batch costs ~$0.0014, and the largest plausible
# evaluation batch is a few cents, so a few dollars is generous headroom
# while still bounding a runaway or a hostile uploader.
SPEND_CAP_USD = float(os.environ.get("VTA_SPEND_CAP_USD", "5.00"))

# Headroom held while one call is in flight, before its true cost is known.
# Observed cost on the provided clips is $0.00034-$0.00062 per call, so this
# is roughly 3x the top of the measured range. It is replaced by the measured
# cost the moment the call returns, so over-reserving costs nothing except a
# slightly earlier stop at the very edge of the budget.
RESERVE_PER_CALL_USD = float(os.environ.get("VTA_RESERVE_PER_CALL_USD", "0.002"))

LEDGER_PATH = Path(
    os.environ.get(
        "VTA_SPEND_LEDGER",
        Path(__file__).resolve().parents[2] / "data" / "cache" / "spend_ledger.json",
    )
)


class BudgetExhausted(Exception):
    """No headroom left under the spend cap. Raised instead of making the call."""


@dataclass(frozen=True)
class BudgetState:
    """A point-in-time view of the ledger, safe to hand to a template."""

    spent_usd: float
    cap_usd: float
    reserved_usd: float
    calls: int
    reserve_per_call_usd: float

    @property
    def remaining_usd(self) -> float:
        return max(self.cap_usd - self.spent_usd - self.reserved_usd, 0.0)

    @property
    def pct_used(self) -> float:
        if self.cap_usd <= 0:
            return 100.0
        return min(100.0 * self.spent_usd / self.cap_usd, 100.0)

    @property
    def exhausted(self) -> bool:
        return self.remaining_usd < self.reserve_per_call_usd

    def as_dict(self) -> dict:
        return {
            "spent_usd": self.spent_usd,
            "cap_usd": self.cap_usd,
            "reserved_usd": self.reserved_usd,
            "calls": self.calls,
            "remaining_usd": self.remaining_usd,
            "pct_used": self.pct_used,
            "exhausted": self.exhausted,
        }


class SpendLedger:
    """Cumulative API spend against a fixed cap, persisted to one JSON file.

    Reservations are in-memory only: they describe calls in flight in this
    process, which by definition do not outlive it. Only settled spend is
    written to disk.
    """

    def __init__(
        self,
        path: Path | None = None,
        cap_usd: float | None = None,
        reserve_per_call_usd: float | None = None,
    ) -> None:
        self.path = Path(path) if path is not None else LEDGER_PATH
        self.cap_usd = SPEND_CAP_USD if cap_usd is None else cap_usd
        self.reserve_per_call_usd = (
            RESERVE_PER_CALL_USD
            if reserve_per_call_usd is None
            else reserve_per_call_usd
        )
        self._lock = threading.Lock()
        self._reserved_usd = 0.0
        self._spent_usd, self._calls = self._read()

    # --- persistence ---

    def _read(self) -> tuple[float, int]:
        """Load settled spend. A missing or unreadable ledger reads as zero.

        Failing closed (treating an unreadable ledger as cap-exhausted) would
        turn one corrupt file into a dead dashboard; failing open risks at
        most one cap's worth of spend, and the file is ours alone to write.
        """
        try:
            data = json.loads(self.path.read_text())
            return float(data["spent_usd"]), int(data.get("calls", 0))
        except (OSError, ValueError, KeyError, TypeError):
            return 0.0, 0

    def _write(self) -> None:
        """Replace the ledger atomically, so a crash mid-write cannot truncate it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"spent_usd": self._spent_usd, "calls": self._calls}, indent=2)
        )
        os.replace(tmp, self.path)

    # --- reservation lifecycle ---

    def reserve(self) -> bool:
        """Claim headroom for one call. False means the caller must not call."""
        with self._lock:
            projected = self._spent_usd + self._reserved_usd + self.reserve_per_call_usd
            if projected > self.cap_usd:
                return False
            self._reserved_usd += self.reserve_per_call_usd
            return True

    def settle(self, actual_usd: float) -> None:
        """Swap a reservation for the call's measured cost and persist it."""
        with self._lock:
            self._reserved_usd = max(self._reserved_usd - self.reserve_per_call_usd, 0.0)
            self._spent_usd += max(actual_usd, 0.0)
            self._calls += 1
            self._write()

    def release(self) -> None:
        """Drop a reservation whose call failed, so a failure costs no budget."""
        with self._lock:
            self._reserved_usd = max(self._reserved_usd - self.reserve_per_call_usd, 0.0)

    # --- reads ---

    def state(self) -> BudgetState:
        with self._lock:
            return BudgetState(
                spent_usd=self._spent_usd,
                cap_usd=self.cap_usd,
                reserved_usd=self._reserved_usd,
                calls=self._calls,
                reserve_per_call_usd=self.reserve_per_call_usd,
            )


# Process-wide ledger. Tests build their own against a tmp path.
LEDGER = SpendLedger()
