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
import subprocess
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


_BS = 1 << 26  # 64 MB, block-aligned for O_DIRECT


def dd_write(path: str, nbytes: int) -> int:
    """O_DIRECT durable write -> measures DEVICE write bandwidth (bypasses page cache)."""
    count = nbytes // _BS
    subprocess.run(["dd", "if=/dev/zero", f"of={path}", f"bs={_BS}", f"count={count}",
                    "oflag=direct", "conv=fdatasync"], check=True, capture_output=True)
    return count * _BS


def dd_read(path: str, nbytes: int) -> int:
    """O_DIRECT read -> measures DEVICE read bandwidth (bypasses page cache)."""
    count = nbytes // _BS
    subprocess.run(["dd", f"if={path}", "of=/dev/null", f"bs={_BS}", f"count={count}",
                    "iflag=direct"], check=True, capture_output=True)
    return count * _BS


def main() -> None:
    ap = argparse.ArgumentParser(description="storage-tier write/read latency + energy")
    ap.add_argument("--bytes", type=float, default=16e9, help="bytes per write/read")
    ap.add_argument("--tiers", default=None,
                    help="name=dir,name=dir (default: ford.yaml storage.tiers)")
    ap.add_argument("--baseline", type=float, default=5.0)
    ap.add_argument("--method", choices=["direct", "buffered"], default="direct",
                    help="direct = O_DIRECT dd (DEVICE capability); "
                         "buffered = Python page-cache write+fsync (realistic checkpoint)")
    args = ap.parse_args()

    write_op = dd_write if args.method == "direct" else (lambda p, n: nvme_write(n, p, iters=1))
    read_op = dd_read if args.method == "direct" else nvme_read

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
                op=lambda p=path: write_op(p, nbytes),
                config={"tier": name, "dir": d, "method": args.method})
            print_record(rec_w); write_record(rec_w, "storage_char")

            drop_caches()
            rec_r = measure_operation(
                tele, workload=f"storage:{name}", operation="read",
                state_bytes=nbytes, baseline_seconds=args.baseline,
                op=lambda p=path: read_op(p, nbytes),
                config={"tier": name, "dir": d, "method": args.method})
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

    def cpu_marg(rec):  # CPU energy above idle (data-movement cost)
        for s in rec.sources:
            if s.name == "cpu_pkg_energy_rapl":
                return s.energy_j or 0.0
        return 0.0

    def cpu_abs(rec):  # CPU energy incl. idle = data-movement + TIME TERM (dominant)
        for s in rec.sources:
            if s.name == "cpu_pkg_energy_rapl" and s.energy_abs_j is not None:
                return s.energy_abs_j
        return 0.0

    def pjb_abs(rec):  # pJ/bit from the time-term-inclusive CPU energy
        return cpu_abs(rec) / (nbytes * 8) * 1e12 if cpu_abs(rec) else 0.0

    print(f"\n=== STORAGE TIER CHARACTERIZATION ({nbytes/1e9:.0f} GB, method={args.method}) ===")
    print("  cpu_marg = CPU energy above idle (move only) | cpu_abs = incl. idle "
          "(move + time term, the real dump cost)")
    print(f"{'tier':14}{'op':7}{'GB/s':>8}{'latency_s':>11}{'cpu_marg_J':>12}"
          f"{'cpu_abs_J':>12}{'pJ/bit(abs)':>13}")
    for name, rw, rr in rows:
        for op, rec in (("write", rw), ("read", rr)):
            print(f"{name:14}{op:7}{bw(rec):8.2f}{rec.latency_s:11.2f}"
                  f"{cpu_marg(rec):12.1f}{cpu_abs(rec):12.1f}{pjb_abs(rec):13.1f}")


if __name__ == "__main__":
    main()
