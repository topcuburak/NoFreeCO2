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


def _rapl_package_files() -> list[str]:
    out: list[str] = []
    for d in sorted(glob.glob("/sys/class/powercap/intel-rapl:*")):
        base = os.path.basename(d)              # intel-rapl:0  (pkg) vs intel-rapl:0:0 (subdomain)
        if base.count(":") != 1:                # keep only top-level package domains
            continue
        ej = os.path.join(d, "energy_uj")
        if os.path.exists(ej):
            out.append(ej)
    return out


class RaplSource(CounterSource):
    name = "cpu_pkg_energy_rapl"
    kind = "energy_counter"

    def __init__(self) -> None:
        super().__init__()
        self._files = _rapl_package_files()
        readable = False
        for f in self._files:                   # probe (needs root)
            try:
                int(open(f).read().strip())
                readable = True
            except OSError:
                pass
        self.available = readable and len(self._files) > 0
        self.detect_note = (
            f"{len(self._files)} RAPL package domain(s)"
            if self.available else "RAPL energy_uj absent or not root-readable"
        )

    def read_counter(self) -> float | None:
        """Cumulative package energy in Joules (sum across sockets)."""
        if not self.available:
            return None
        total_uj = 0
        try:
            for f in self._files:
                total_uj += int(open(f).read().strip())
        except (OSError, ValueError):
            return None
        # NOTE: energy_uj wraps at max_energy_range_uj; ops here are seconds-scale so
        # a wrap is unlikely, but a long hold could hit it -- revisit if deltas go negative.
        return total_uj / 1e6
