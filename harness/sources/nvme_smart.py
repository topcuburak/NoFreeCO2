"""NVMe bytes written/read via SMART (data units * 512000 bytes).

A BYTE counter, not power: used to cross-check the modeled NVMe energy
(bytes * 65 pJ/bit write) against the IPMI-minus-package residual, and to
verify S actually hit the drive.
"""
from __future__ import annotations

import json
import shutil
import subprocess

from .base import CounterSource

_UNIT_BYTES = 512_000  # NVMe "data unit" = 1000 * 512 bytes


class NvmeSmartSource(CounterSource):
    name = "nvme_bytes_written"
    kind = "byte_counter"

    def __init__(self, cfg: dict | None = None) -> None:
        super().__init__()
        self._bin = shutil.which("nvme")
        # Map scratch mount -> device; on ford this is the md0 members. Resolve on
        # testbed; default to nvme0 for off-testbed import.
        self._dev = (cfg or {}).get("storage", {}).get("device", "/dev/nvme0")
        self.available = self._bin is not None
        self.detect_note = "nvme-cli found" if self.available else "nvme-cli not in PATH"

    def read_counter(self) -> float | None:
        """Cumulative bytes WRITTEN to the device."""
        if not self.available:
            return None
        try:
            out = subprocess.run(
                [self._bin, "smart-log", self._dev, "-o", "json"],
                capture_output=True, text=True, timeout=3,
            ).stdout
            d = json.loads(out)
        except (subprocess.SubprocessError, OSError, json.JSONDecodeError):
            return None
        units = d.get("data_units_written")
        return float(units) * _UNIT_BYTES if units is not None else None
