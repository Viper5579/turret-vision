"""Match camera frames to the ESP32 trigger that exposed them.

The camera numbers frames from when streaming started; the ESP32 numbers
triggers from when it booted. Neither knows the other. We find the constant
difference k = trigger_seq - cam_seq by timing (a frame arrives a roughly
constant USB delay after its trigger), lock onto it once several frames agree,
and from then on match by sequence number, which survives jitter and drops.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

from ..link.messages import FrameSync


@dataclass
class MatcherStats:
    matched: int = 0
    unmatched: int = 0
    relocks: int = 0
    delay_s: float = 0.010  # running estimate of trigger -> arrival


class FrameMatcher:
    LOCK_AGREE = 3
    MIN_DELAY_S = 0.001
    MAX_DELAY_S = 0.080

    def __init__(self, esp_to_jetson: Callable[[int], float]) -> None:
        self._to_j = esp_to_jetson
        self._syncs: OrderedDict[int, FrameSync] = OrderedDict()
        self._k: int | None = None
        self._candidate: tuple[int, int] | None = None  # (k, agreeing count)
        self._misses = 0
        self.stats = MatcherStats()

    @property
    def locked(self) -> bool:
        return self._k is not None

    def add_sync(self, fs: FrameSync) -> None:
        self._syncs[fs.seq] = fs
        while len(self._syncs) > 512:
            self._syncs.popitem(last=False)

    def match(self, cam_seq: int, arrival: float, waited_s: float) -> FrameSync | str | None:
        """Return the FrameSync, None if the frame cannot be matched, or "wait" to retry later."""
        if self._k is not None:
            fs = self._syncs.get(cam_seq + self._k)
            if fs is not None:
                d = arrival - self._to_j(fs.t_us)
                if self.MIN_DELAY_S <= d <= self.MAX_DELAY_S:
                    self._ok(d)
                    return fs
            elif waited_s < 0.03:
                return "wait"
            self._misses += 1
            if self._misses >= 5:  # lost lock (camera restarted, ESP32 rebooted): find k again
                self._k = None
                self._candidate = None
                self.stats.relocks += 1
            self.stats.unmatched += 1
            return None

        # Unlocked: pick the trigger whose delay best fits the running estimate.
        best = None
        best_err = float("inf")
        for fs in reversed(self._syncs.values()):
            d = arrival - self._to_j(fs.t_us)
            if d < self.MIN_DELAY_S:
                continue
            if d > self.MAX_DELAY_S:
                break
            err = abs(d - self.stats.delay_s)
            if err < best_err:
                best, best_err = fs, err
        if best is None:
            if waited_s < 0.03:
                return "wait"
            self.stats.unmatched += 1
            return None
        k = best.seq - cam_seq
        if self._candidate and self._candidate[0] == k:
            self._candidate = (k, self._candidate[1] + 1)
        else:
            self._candidate = (k, 1)
        if self._candidate[1] >= self.LOCK_AGREE:
            self._k = k
            self._misses = 0
        self._ok(arrival - self._to_j(best.t_us))
        return best

    def _ok(self, d: float) -> None:
        self._misses = 0
        self.stats.matched += 1
        self.stats.delay_s += 0.05 * (d - self.stats.delay_s)
