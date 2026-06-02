"""Source interfaces.

Two flavours:
  PowerSource   -> instantaneous watts, polled by the central loop at >=10 Hz,
                   integrated as ∫(P - P_baseline) dt over the op window.
  CounterSource -> monotonically increasing counter (energy in J, or bytes),
                   read once at T_start and once at T_end; we keep the delta.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Source(ABC):
    name: str = "source"
    kind: str = "base"

    def __init__(self) -> None:
        self.available: bool = False
        self.detect_note: str = ""

    def start(self) -> None:  # optional per-run setup
        pass

    def stop(self) -> None:   # optional per-run teardown
        pass


class PowerSource(Source):
    kind = "power_integral"

    @abstractmethod
    def read_power_w(self) -> float | None:
        """Instantaneous power in Watts, or None if momentarily unavailable."""
        ...


class CounterSource(Source):
    # kind set by subclass: "energy_counter" (Joules) or "byte_counter" (bytes)
    @abstractmethod
    def read_counter(self) -> float | None:
        """Monotonic cumulative counter (Joules or bytes), or None."""
        ...
