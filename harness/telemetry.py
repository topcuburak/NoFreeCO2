"""Central time-aligned sampler.

A background thread polls every PowerSource at `sample_hz`, stamping each sample
with time.monotonic() (CLOCK_MONOTONIC) so NVML / IPMI / perf land on ONE clock
(measurement pitfall 3). CounterSources are read on demand at window boundaries.
"""
from __future__ import annotations

import threading
import time

from .sources.base import Source, PowerSource, CounterSource


class PowerTrace:
    """Per-source time series of (monotonic_ts, watts)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.ts: list[float] = []
        self.w: list[float] = []

    def add(self, t: float, watts: float | None) -> None:
        if watts is not None:
            self.ts.append(t)
            self.w.append(watts)

    def mean_between(self, t0: float, t1: float) -> float | None:
        vals = [w for t, w in zip(self.ts, self.w) if t0 <= t <= t1]
        return sum(vals) / len(vals) if vals else None

    def peak_between(self, t0: float, t1: float) -> float | None:
        vals = [w for t, w in zip(self.ts, self.w) if t0 <= t <= t1]
        return max(vals) if vals else None

    def integral_between(self, t0: float, t1: float, baseline_w: float = 0.0) -> float | None:
        """∫(P - baseline) dt over [t0, t1] via trapezoid on samples in-window."""
        pts = [(t, w) for t, w in zip(self.ts, self.w) if t0 <= t <= t1]
        if len(pts) < 2:
            return None
        e = 0.0
        for (ta, wa), (tb, wb) in zip(pts, pts[1:]):
            e += ((wa - baseline_w) + (wb - baseline_w)) / 2.0 * (tb - ta)
        return e  # Joules


class Telemetry:
    def __init__(self, sources: list[Source], sample_hz: float = 20.0) -> None:
        self.sources = [s for s in sources if s.available]
        self.power_sources: list[PowerSource] = [s for s in self.sources if isinstance(s, PowerSource)]
        self.counter_sources: list[CounterSource] = [s for s in self.sources if isinstance(s, CounterSource)]
        self.sample_hz = sample_hz
        self.traces: dict[str, PowerTrace] = {s.name: PowerTrace(s.name) for s in self.power_sources}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- power sampling thread ----
    def _loop(self) -> None:
        period = 1.0 / self.sample_hz
        while not self._stop.is_set():
            t = time.monotonic()
            for s in self.power_sources:
                self.traces[s.name].add(t, s.read_power_w())
            dt = period - (time.monotonic() - t)
            if dt > 0:
                self._stop.wait(dt)

    def start(self) -> None:
        for s in self.sources:
            s.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        for s in self.sources:
            s.stop()

    # ---- counter reads (energy/bytes) at window boundaries ----
    def read_counters(self) -> dict[str, float | None]:
        return {s.name: s.read_counter() for s in self.counter_sources}

    def summary(self) -> str:
        avail = ", ".join(s.name for s in self.sources) or "NONE"
        missing = ", ".join(
            f"{s.name}({s.detect_note})"
            for s in (self.power_sources + self.counter_sources)
            if not s.available
        )
        return f"active: {avail}" + (f" | unavailable: {missing}" if missing else "")
