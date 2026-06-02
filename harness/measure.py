"""Operation-agnostic, window-based differential measurement.

measure_operation() is the one reusable entry point every workload/microbench
plugs into. Protocol (matches measurement plan §Telemetry):

  1. Sample power >=10 Hz (Telemetry thread already running).
  2. Establish baseline over `baseline_seconds` under the SAME workload state.
  3. Read energy/byte counters at T_start (CLOCK_MONOTONIC).
  4. Run the operation callable.
  5. Read counters at T_end.
  6. PowerSource:   E_i = ∫(P_i - P_i_baseline) dt over [T_start, T_end].
     EnergyCounter: E_i = counter(T_end) - counter(T_start).
     ByteCounter:   bytes_i = counter(T_end) - counter(T_start).
"""
from __future__ import annotations

import time
from typing import Callable

from .schema import RunRecord, SourceResult
from .telemetry import Telemetry


def _run_id(workload: str, operation: str) -> str:
    return f"{workload}.{operation}.{time.time_ns()}"


def measure_operation(
    tele: Telemetry,
    *,
    workload: str,
    operation: str,
    state_bytes: int,
    op: Callable[[], object],
    config: dict | None = None,
    baseline_seconds: float = 30.0,
    settle_seconds: float = 1.0,
) -> RunRecord:
    config = config or {}

    # --- 1-2. baseline window (workload state held, no operation) ---
    t_base0 = time.monotonic()
    time.sleep(baseline_seconds)
    t_base1 = time.monotonic()
    baselines = {
        ps.name: tele.traces[ps.name].mean_between(t_base0, t_base1)
        for ps in tele.power_sources
    }
    time.sleep(settle_seconds)

    # --- 3. counters at T_start ---
    c_start = tele.read_counters()

    # --- 4. the operation ---
    t_start = time.monotonic()
    outcome = "ok"
    try:
        result = op()
    except Exception as e:  # capture, still record the window we have
        outcome = f"error:{type(e).__name__}:{e}"
        result = None
    t_end = time.monotonic()

    # --- 5. counters at T_end ---
    c_end = tele.read_counters()

    # --- 6. attribute per source ---
    sources: list[SourceResult] = []
    for ps in tele.power_sources:
        base = baselines.get(ps.name)
        e = tele.traces[ps.name].integral_between(t_start, t_end, base or 0.0)
        sources.append(SourceResult(
            name=ps.name, kind="power_integral", energy_j=e,
            baseline_w=base, peak_w=tele.traces[ps.name].peak_between(t_start, t_end),
            note="" if base is not None else "no baseline samples",
        ))
    for cs in tele.counter_sources:
        s0, s1 = c_start.get(cs.name), c_end.get(cs.name)
        delta = (s1 - s0) if (s0 is not None and s1 is not None) else None
        if cs.kind == "byte_counter":
            sources.append(SourceResult(cs.name, cs.kind, energy_j=None, bytes_delta=int(delta) if delta is not None else None))
        else:  # energy_counter (Joules)
            sources.append(SourceResult(cs.name, cs.kind, energy_j=delta))

    rec = RunRecord(
        run_id=_run_id(workload, operation),
        workload=workload, operation=operation, config=config,
        state_bytes=state_bytes, t_start_mono=t_start, t_end_mono=t_end,
        sources=sources, outcome=outcome,
        extra={"op_result": str(result)[:200]} if result is not None else {},
    )
    return rec
