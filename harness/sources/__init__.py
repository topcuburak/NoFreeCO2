"""Telemetry sources. Each detects its own availability and degrades gracefully
so the harness imports and partially runs off-testbed (no NVML/perf/IPMI)."""
from .base import Source, PowerSource, CounterSource
from .nvml_source import NvmlSource
from .perf_source import PerfPkgSource
from .rapl_source import RaplSource
from .ipmi_source import IpmiSource
from .nvme_smart import NvmeSmartSource

__all__ = [
    "Source", "PowerSource", "CounterSource",
    "NvmlSource", "PerfPkgSource", "RaplSource", "IpmiSource", "NvmeSmartSource",
]


def default_sources(cfg: dict) -> list[Source]:
    """Build the source set declared in ford.yaml, skipping unavailable ones."""
    want = cfg.get("telemetry", {}).get("sources", {})
    candidates: list[Source] = []
    if want.get("nvml"):       candidates.append(NvmlSource())
    if want.get("rapl", True): candidates.append(RaplSource())      # AMD CPU pkg energy
    if want.get("perf_pkg"):   candidates.append(PerfPkgSource())   # amd_energy hwmon fallback
    if want.get("ipmi"):       candidates.append(IpmiSource())
    if want.get("nvme_smart"): candidates.append(NvmeSmartSource(cfg))
    return candidates
