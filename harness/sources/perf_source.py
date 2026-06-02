"""CPU package energy (cores + IOD) on AMD EPYC.

Preferred path: amd_energy hwmon cumulative counter (microjoules), read as an
energy counter. Fallback: `perf stat -e power/energy-pkg/` over the window.

AMD EPYC exposes ONLY the package domain via RAPL/amd_energy -- NO DRAM domain
(measurability flag 1). DRAM energy is modeled at ~10 pJ/bit elsewhere.
"""
from __future__ import annotations

import glob
import os

from .base import CounterSource


def _find_pkg_energy_files() -> list[str]:
    """amd_energy exposes energy*_input (uJ) under hwmon; socket pkg first."""
    out: list[str] = []
    for hw in glob.glob("/sys/class/hwmon/hwmon*"):
        try:
            name = open(os.path.join(hw, "name")).read().strip()
        except OSError:
            continue
        if name not in ("amd_energy", "power"):
            continue
        for f in sorted(glob.glob(os.path.join(hw, "energy*_input"))):
            label_f = f.replace("_input", "_label")
            label = ""
            try:
                label = open(label_f).read().strip().lower()
            except OSError:
                pass
            # socket/package counters; skip per-core to avoid double counting
            if "core" not in label:
                out.append(f)
    return out


class PerfPkgSource(CounterSource):
    name = "cpu_pkg_energy"
    kind = "energy_counter"

    def __init__(self) -> None:
        super().__init__()
        self._files = _find_pkg_energy_files()
        self.available = len(self._files) > 0
        self.detect_note = (
            f"{len(self._files)} hwmon energy counter(s)"
            if self.available else "no amd_energy hwmon; use perf stat fallback"
        )

    def read_counter(self) -> float | None:
        """Cumulative package energy in Joules (sum across socket counters)."""
        if not self.available:
            return None
        total_uj = 0
        try:
            for f in self._files:
                total_uj += int(open(f).read().strip())
        except (OSError, ValueError):
            return None
        return total_uj / 1e6  # uJ -> J
