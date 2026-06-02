#!/usr/bin/env python3
"""Characterize each storage tier: durable WRITE (dump) and cold READ (restore)
latency + energy. Produces the storage-tier sensitivity table.

For each tier (name -> dir from ford.yaml storage.tiers, or --tiers):
  WRITE: nvme_write(nbytes) durable (fsync)  -> dump leg
  READ : drop page cache, then nvme_read(nbytes) -> restore leg (cold, hits device)
Each is wrapped in the harness (CPU RAPL energy, NVMe bytes, GPU). Reports
bandwidth (bytes/latency), marginal energy, and pJ/bit per tier per direction.

Run as root (RAPL energy + drop_caches need it):
    sudo -E $(which python) scripts/characterize_storage.py --bytes 16e9
    sudo -E $(which python) scripts/characterize_storage.py --bytes 16e9 \
        --tiers nvme_raid0=/var/data,sas_ssd=/home/test

NOTE: single PM1733 isn't separately mountable (both in md0) -> model it as
nvme_raid0 bandwidth / 2 in analysis.
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from harness import measure_operation                              # noqa: E402
from harness.configload import load                                # noqa: E402
from microbench.isolation import nvme_write, nvme_read             # noqa: E402
from _common import build_telemetry, write_record, print_record    # noqa: E402


def drop_caches() -> None:
    os.system("sync")
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except OSError as e:
        print(f"[storage] WARN: can't drop caches ({e}); reads may hit page cache")


def main() -> None:
    ap = argparse.ArgumentParser(description="storage-tier write/read latency + energy")
    ap.add_argument("--bytes", type=float, default=16e9, help="bytes per write/read")
    ap.add_argument("--tiers", default=None,
                    help="name=dir,name=dir (default: ford.yaml storage.tiers)")
    ap.add_argument("--baseline", type=float, default=5.0)
    args = ap.parse_args()

    cfg = load("ford.yaml")
    if args.tiers:
        tiers = dict(kv.split("=", 1) for kv in args.tiers.split(","))
    else:
        tiers = cfg.get("storage", {}).get("tiers", {})
    if not tiers:
        raise SystemExit("no tiers (set ford.yaml storage.tiers or --tiers)")
    nbytes = int(args.bytes)
    print(f"[storage] tiers={tiers}  bytes={nbytes/1e9:.0f} GB")

    tele = build_telemetry(cfg)
    tele.start()
    rows = []
    try:
        for name, d in tiers.items():
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, "socc_storage_char.bin")
            print(f"\n[storage] === tier {name} ({d}) ===")

            rec_w = measure_operation(
                tele, workload=f"storage:{name}", operation="write",
                state_bytes=nbytes, baseline_seconds=args.baseline,
                op=lambda p=path: nvme_write(nbytes, p, iters=1),
                config={"tier": name, "dir": d})
            print_record(rec_w); write_record(rec_w, "storage_char")

            drop_caches()
            rec_r = measure_operation(
                tele, workload=f"storage:{name}", operation="read",
                state_bytes=nbytes, baseline_seconds=args.baseline,
                op=lambda p=path: nvme_read(p, nbytes),
                config={"tier": name, "dir": d})
            print_record(rec_r); write_record(rec_r, "storage_char")

            try:
                os.remove(path)
            except OSError:
                pass
            rows.append((name, rec_w, rec_r))
    finally:
        tele.stop()

    def bw(rec):  # GB/s
        return nbytes / rec.latency_s / 1e9 if rec.latency_s else 0.0

    def pjb(rec):  # pJ/bit from marginal energy
        return rec.total_energy_j / (nbytes * 8) * 1e12 if rec.total_energy_j else 0.0

    print("\n=== STORAGE TIER CHARACTERIZATION "
          f"({nbytes/1e9:.0f} GB) ===")
    print(f"{'tier':14}{'op':7}{'GB/s':>8}{'latency_s':>11}{'marginal_J':>12}{'pJ/bit':>11}")
    for name, rw, rr in rows:
        for op, rec in (("write", rw), ("read", rr)):
            print(f"{name:14}{op:7}{bw(rec):8.2f}{rec.latency_s:11.2f}"
                  f"{rec.total_energy_j:12.1f}{pjb(rec):11.1f}")


if __name__ == "__main__":
    main()
