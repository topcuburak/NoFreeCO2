"""Run the isolation microbenchmarks under the harness to derive measured
per-component coefficients (ε_HBM, ε_PCIe+DRAM, ε_NVMe_write).

This is the FIRST thing to run on ford: it exercises the full measurement loop
(telemetry -> measure_operation -> record) without needing model weights, and
produces the coefficients that decompose the lumped GPU package power.

    python scripts/run_microbench.py --bytes 8e9
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import measure_operation                      # noqa: E402
from harness.configload import load                         # noqa: E402
from microbench.isolation import hbm_copy, pcie_copy, nvme_write  # noqa: E402
from _common import build_telemetry, write_record, print_record   # noqa: E402


def pj_per_bit(energy_j: float | None, nbytes: int) -> float | None:
    if energy_j is None or nbytes == 0:
        return None
    return energy_j / (nbytes * 8) * 1e12  # J/bit -> pJ/bit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=float, default=4e9, help="bytes per iteration")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--baseline", type=float, default=10.0)
    ap.add_argument("--scratch", default=None,
                    help="dir for nvme_write (must be on the NVMe you want to measure; "
                         "default: ford.yaml storage.scratch)")
    args = ap.parse_args()

    cfg = load("ford.yaml")
    scratch = args.scratch or cfg.get("storage", {}).get("scratch", "/tmp")
    print(f"[microbench] nvme_write target dir: {scratch}")
    nbytes = int(args.bytes)

    tele = build_telemetry(cfg)
    tele.start()
    try:
        benches = [
            ("hbm_copy",   lambda: hbm_copy(nbytes, gpu=args.gpu)),
            ("pcie_copy",  lambda: pcie_copy(nbytes, gpu=args.gpu)),
            ("nvme_write", lambda: nvme_write(nbytes, os.path.join(scratch, "socc_mb.bin"))),
        ]
        for name, op in benches:
            rec = measure_operation(
                tele, workload=f"microbench:{name}", operation="microbench",
                state_bytes=nbytes, op=op, config={"bytes": nbytes, "gpu": args.gpu},
                baseline_seconds=args.baseline,
            )
            print_record(rec)
            moved = int(rec.extra.get("op_result") or nbytes)
            print(f"  --> measured {pj_per_bit(rec.total_energy_j, moved)} pJ/bit (over {moved} B moved)")
            write_record(rec, "microbench")
    finally:
        tele.stop()


if __name__ == "__main__":
    main()
