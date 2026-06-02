"""GPU package power via NVML (HBM + SM + on-chip + PCIe controller, lumped).

NOTE (measurability flag 2): NVML reports ONE package number per GPU; SM and HBM
cannot be separated here. HBM energy is attributed downstream as S * 3.5 pJ/bit
and the remainder assigned to SM/quiesce. See measurement plan §C1.
"""
from __future__ import annotations

from .base import PowerSource

try:
    import pynvml  # type: ignore
    _HAVE = True
except Exception:  # pragma: no cover - off-testbed
    _HAVE = False


class NvmlSource(PowerSource):
    name = "nvml_gpu_pkg"

    def __init__(self, gpu_indices: list[int] | None = None) -> None:
        super().__init__()
        self._handles: list = []
        self._indices = gpu_indices
        if not _HAVE:
            self.detect_note = "pynvml not importable"
            return
        try:
            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            idxs = self._indices if self._indices is not None else range(n)
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in idxs]
            self.available = len(self._handles) > 0
            self.detect_note = f"{len(self._handles)} GPU(s)"
        except Exception as e:  # pragma: no cover
            self.detect_note = f"nvmlInit failed: {e}"

    def read_power_w(self) -> float | None:
        if not self.available:
            return None
        try:
            mw = sum(pynvml.nvmlDeviceGetPowerUsage(h) for h in self._handles)
            return mw / 1000.0
        except Exception:  # pragma: no cover
            return None
