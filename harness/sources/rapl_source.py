"""CPU package energy via the RAPL powercap interface.

AMD EPYC exposes package energy through the intel-rapl powercap framework
(/sys/class/powercap/intel-rapl:N/energy_uj, microjoules, ROOT-readable only).
We sum the top-level package domains. This is an ENERGY COUNTER: the delta over an
op window is the ABSOLUTE CPU energy consumed during it (idle + active) -- which
sidesteps the active-vs-idle baseline ambiguity that bites the GPU power-integral.

NOTE: AMD exposes package (and usually 'core') domains but NOT 'dram', so DRAM
energy stays modeled (~10 pJ/bit) -- the known EPYC limitation in the methodology.
"""
from __future__ import annotations

import glob
import os

from .base import CounterSource


def _rapl_package_dirs() -> list[str]:
    out: list[str] = []
    for d in sorted(glob.glob("/sys/class/powercap/intel-rapl:*")):
        base = os.path.basename(d)              # intel-rapl:0 (pkg) vs intel-rapl:0:0 (subdomain)
        if base.count(":") != 1:                # keep only top-level package domains
            continue
        if os.path.exists(os.path.join(d, "energy_uj")):
            out.append(d)
    return out


def _read_int(path: str) -> int | None:
    try:
        return int(open(path).read().strip())
    except (OSError, ValueError):
        return None


class RaplSource(CounterSource):
    """RAPL package energy, unwrapped. energy_uj is a fixed-width counter that rolls
    over at max_energy_range_uj (~65 kJ on AMD); a multi-second op can cross a wrap,
    making a raw delta negative. We unwrap PER DOMAIN across reads (each read gap is
    far shorter than the wrap period, so at most one wrap per gap) and return a
    monotonic cumulative Joule value."""
    name = "cpu_pkg_energy_rapl"
    kind = "energy_counter"

    def __init__(self) -> None:
        super().__init__()
        self._dirs = _rapl_package_dirs()
        self._max = {d: (_read_int(os.path.join(d, "max_energy_range_uj")) or 0)
                     for d in self._dirs}
        self._last: dict[str, int] = {}
        self._off: dict[str, int] = {}
        readable = any(_read_int(os.path.join(d, "energy_uj")) is not None for d in self._dirs)
        self.available = readable and len(self._dirs) > 0
        self.detect_note = (
            f"{len(self._dirs)} RAPL package domain(s), max~{max(self._max.values(), default=0)//10**6}kJ"
            if self.available else "RAPL energy_uj absent or not root-readable"
        )

    def read_counter(self) -> float | None:
        if not self.available:
            return None
        total_uj = 0
        for d in self._dirs:
            raw = _read_int(os.path.join(d, "energy_uj"))
            if raw is None:
                return None
            last = self._last.get(d)
            if last is not None and raw < last and self._max[d]:
                self._off[d] = self._off.get(d, 0) + self._max[d]   # wrapped since last read
            self._last[d] = raw
            total_uj += self._off.get(d, 0) + raw
        return total_uj / 1e6
