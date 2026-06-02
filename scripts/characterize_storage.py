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
import statistics
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


def cpu_abs_j(rec) -> float:
    """CPU package energy incl. idle over the op = move + time term (the real cost)."""
    for s in rec.sources:
        if s.name == "cpu_pkg_energy_rapl" and s.energy_abs_j is not None:
            return s.energy_abs_j
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="storage-tier write/read latency + energy")
    ap.add_argument("--bytes", type=float, default=16e9, help="bytes per write/read")
    ap.add_argument("--tiers", default=None,
                    help="name=dir,name=dir (default: ford.yaml storage.tiers)")
    ap.add_argument("--baseline", type=float, default=5.0)
    ap.add_argument("--method", choices=["direct", "buffered"], default="direct",
                    help="direct = O_DIRECT dd (DEVICE capability); "
                         "buffered = Python page-cache write+fsync (realistic checkpoint)")
    ap.add_argument("--repeat", type=int, default=1, help="measured trials per tier/op")
    ap.add_argument("--warmup", type=int, default=0, help="discard this many warmup trials first")
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
    rows = []                                  # (name, [write_recs], [read_recs])
    n_trials = args.warmup + args.repeat
    try:
        for name, d in tiers.items():
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, "socc_storage_char.bin")
            print(f"\n[storage] === tier {name} ({d})  {args.repeat} trial(s) "
                  f"(+{args.warmup} warmup) ===")
            w_recs, r_recs = [], []
            for i in range(n_trials):
                tag = "warmup" if i < args.warmup else f"t{i - args.warmup + 1}"
                rec_w = measure_operation(
                    tele, workload=f"storage:{name}", operation="write",
                    state_bytes=nbytes, baseline_seconds=args.baseline,
                    op=lambda p=path: write_op(p, nbytes),
                    config={"tier": name, "dir": d, "method": args.method, "trial": tag})
                drop_caches()
                rec_r = measure_operation(
                    tele, workload=f"storage:{name}", operation="read",
                    state_bytes=nbytes, baseline_seconds=args.baseline,
                    op=lambda p=path: read_op(p, nbytes),
                    config={"tier": name, "dir": d, "method": args.method, "trial": tag})
                print(f"  [{name} {tag}] write {rec_w.latency_s:6.2f}s {cpu_abs_j(rec_w):7.0f}J"
                      f"   read {rec_r.latency_s:6.2f}s {cpu_abs_j(rec_r):7.0f}J")
                if i >= args.warmup:
                    w_recs.append(rec_w); r_recs.append(rec_r)
                    write_record(rec_w, "storage_char"); write_record(rec_r, "storage_char")
            try:
                os.remove(path)
            except OSError:
                pass
            rows.append((name, w_recs, r_recs))
    finally:
        tele.stop()

    def msd(vals):
        if not vals:
            return 0.0, 0.0
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    def bw_list(recs):
        return [nbytes / r.latency_s / 1e9 for r in recs if r.latency_s]

    def lat_list(recs):
        return [r.latency_s for r in recs]

    def cpuabs_list(recs):
        return [cpu_abs_j(r) for r in recs]

    print(f"\n=== STORAGE TIER CHARACTERIZATION ({nbytes/1e9:.0f} GB, method={args.method}, "
          f"n={args.repeat}) ===   mean ± std   (cpu_abs = move + time term = real dump cost)")
    print(f"{'tier':12}{'op':6}{'GB/s (mean±std)':>18}{'latency_s':>16}{'cpu_abs_J':>16}")
    for name, wr, rr in rows:
        for op, recs in (("write", wr), ("read", rr)):
            bwm, bws = msd(bw_list(recs))
            lm, ls = msd(lat_list(recs))
            cm, cs = msd(cpuabs_list(recs))
            print(f"{name:12}{op:6}{bwm:9.2f}±{bws:<7.2f}{lm:8.2f}±{ls:<7.2f}{cm:8.0f}±{cs:<7.0f}")


if __name__ == "__main__":
    main()
