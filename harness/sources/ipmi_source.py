"""Whole-node power via IPMI DCMI (Watts, ~1 Hz).

Captures NIC + NVMe + fans + PSU loss that AMD package RAPL misses -- the
chassis-minus-package difference is how we back out NVMe/NIC energy.
"""
from __future__ import annotations

import re
import shutil
import subprocess

from .base import PowerSource

_RE = re.compile(r"Instantaneous power reading:\s*([\d.]+)\s*Watts", re.I)


class IpmiSource(PowerSource):
    name = "ipmi_chassis"

    def __init__(self) -> None:
        super().__init__()
        self._bin = shutil.which("ipmitool")
        self.available = self._bin is not None
        self.detect_note = "ipmitool found" if self.available else "ipmitool not in PATH"

    def read_power_w(self) -> float | None:
        if not self.available:
            return None
        try:
            out = subprocess.run(
                [self._bin, "dcmi", "power", "reading"],
                capture_output=True, text=True, timeout=2,
            ).stdout
        except (subprocess.SubprocessError, OSError):
            return None
        m = _RE.search(out)
        return float(m.group(1)) if m else None
