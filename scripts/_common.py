"""Shared runner plumbing: build telemetry, write records, pretty-print."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import Telemetry, default_sources, RunRecord  # noqa: E402
from harness.configload import load  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def build_telemetry(cfg: dict | None = None) -> Telemetry:
    cfg = cfg or load("ford.yaml")
    hz = cfg.get("telemetry", {}).get("power_sample_hz", 20)
    tele = Telemetry(default_sources(cfg), sample_hz=hz)
    print(f"[telemetry] {tele.summary()}")
    return tele


def write_record(rec: RunRecord, run_tag: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{run_tag}.jsonl")
    with open(path, "a") as f:
        f.write(rec.to_json() + "\n")
    return path


def print_record(rec: RunRecord) -> None:
    print(f"\n=== {rec.workload} / {rec.operation}  [{rec.outcome}] ===")
    print(f"  latency: {rec.latency_s:.4f} s   state: {rec.state_bytes/1e9:.3f} GB")
    for s in rec.sources:
        if s.kind == "byte_counter":
            gb = f" ({s.bytes_delta/1e9:.2f} GB)" if s.bytes_delta else ""
            print(f"  {s.name:22} bytes={s.bytes_delta}{gb}")
        else:
            base = f" (baseline {s.baseline_w:.1f} W)" if s.baseline_w else ""
            ej = f"{s.energy_j:.2f} J" if s.energy_j is not None else "n/a"
            abs_s = f" [abs {s.energy_abs_j:.1f} J]" if s.energy_abs_j is not None else ""
            print(f"  {s.name:22} {ej}{base}{abs_s}")
    abs_total = sum(s.energy_abs_j for s in rec.sources if s.energy_abs_j is not None)
    print(f"  TOTAL marginal: {rec.total_energy_j:.2f} J (above baseline)  |  "
          f"TOTAL absolute: {abs_total:.1f} J (measured ∫P dt)")
