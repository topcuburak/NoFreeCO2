"""NVMe bytes written via sysfs /sys/block/<dev>/stat (no nvme-cli needed).

The block stat's 7th field (0-based index 6) is sectors written; bytes = sectors
* 512. We sum over the NVMe namespace block devices (RAID-0 writes hit both
members). A BYTE counter, used to size the store (host->NVMe) and cross-check the
modeled NVMe energy (bytes * ~65 pJ/bit write).
"""
from __future__ import annotations

import glob
import os

from .base import CounterSource

_SECTOR = 512


class NvmeSmartSource(CounterSource):
    name = "nvme_bytes_written"
    kind = "byte_counter"

    def __init__(self, cfg: dict | None = None) -> None:
        super().__init__()
        devs = (cfg or {}).get("storage", {}).get("nvme_devices")
        if not devs:
            devs = sorted(os.path.basename(p) for p in glob.glob("/sys/block/nvme*n*"))
        self._stat_files = [f"/sys/block/{d}/stat" for d in devs
                            if os.path.exists(f"/sys/block/{d}/stat")]
        self.available = len(self._stat_files) > 0
        self.detect_note = (f"{len(self._stat_files)} nvme block dev(s): {devs}"
                            if self.available else "no nvme block devices")

    def read_counter(self) -> float | None:
        """Cumulative bytes WRITTEN across the NVMe namespaces."""
        if not self.available:
            return None
        total = 0
        try:
            for f in self._stat_files:
                total += int(open(f).read().split()[6]) * _SECTOR
        except (OSError, ValueError, IndexError):
            return None
        return float(total)
