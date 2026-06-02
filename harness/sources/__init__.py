"""Telemetry sources. Each detects its own availability and degrades gracefully
so the harness imports and partially runs off-testbed (no NVML/perf/IPMI)."""
from .base import Source, PowerSource, CounterSource
from .nvml_source import NvmlSource
from .perf_source import PerfPkgSource
from .ipmi_source import IpmiSource
from .nvme_smart import NvmeSmartSource

__all__ = [
    "Source", "PowerSource", "CounterSource",
    "NvmlSource", "PerfPkgSource", "IpmiSource", "NvmeSmartSource",
]


def default_sources(cfg: dict) -> list[Source]:
    """Build the source set declared in ford.yaml, skipping unavailable ones."""
    want = cfg.get("telemetry", {}).get("sources", {})
    candidates: list[Source] = []
    if want.get("nvml"):       candidates.append(NvmlSource())
    if want.get("perf_pkg"):   candidates.append(PerfPkgSource())
    if want.get("ipmi"):       candidates.append(IpmiSource())
    if want.get("nvme_smart"): candidates.append(NvmeSmartSource(cfg))
    return candidates
