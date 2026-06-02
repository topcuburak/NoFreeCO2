"""Per-run logging schema for mechanism-cost measurements.

One RunRecord per operation (dump / restore / migrate / microbench), matching the
schema in socc-2026-paper-measurement-plan.md. Serializes to a JSON line.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class SourceResult:
    """Energy attributed to one telemetry source over the operation window."""
    name: str
    kind: str                 # "power_integral" | "energy_counter" | "byte_counter"
    energy_j: float | None    # integral of (P - P_baseline) dt, or counter delta
    baseline_w: float | None = None
    peak_w: float | None = None
    bytes_delta: int | None = None   # for byte counters (NVMe SMART)
    available: bool = True
    note: str = ""


@dataclass
class RunRecord:
    run_id: str
    workload: str             # e.g. "A2", "microbench:hbm_copy"
    operation: str            # "dump" | "restore" | "migrate" | "baseline" | "microbench"
    config: dict[str, Any]    # TP / batch / context / model / NUMA pin, etc.
    state_bytes: int          # S — size of state moved
    t_start_mono: float       # CLOCK_MONOTONIC seconds at op API call
    t_end_mono: float         # CLOCK_MONOTONIC seconds at completion
    sources: list[SourceResult] = field(default_factory=list)
    outcome: str = "ok"       # "ok" | "error:<msg>"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def latency_s(self) -> float:
        return self.t_end_mono - self.t_start_mono

    @property
    def total_energy_j(self) -> float:
        return sum(s.energy_j or 0.0 for s in self.sources)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)
