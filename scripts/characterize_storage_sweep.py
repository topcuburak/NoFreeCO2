#!/usr/bin/env python3
"""Storage size sweep: store/read a range of sizes (e.g. 2->50 GB) on each tier and
measure latency + power + energy per size, then fit the affine cost model

    E(S)   = a_E + b_E * S       (J;  a_E = fixed overhead, b_E = per-byte energy)
    lat(S) = a_L + b_L * S       (s;  b_L = 1/bandwidth)

Single-size characterization (characterize_storage.py) gives only a slope at one point;
this sweep separates the FIXED overhead (intercept) from the per-byte slope, and shows
whether power is size-independent (it should be ~the node floor).

  WRITE: O_DIRECT dd (durable, fdatasync) -> dump/store leg
  READ : drop page cache, O_DIRECT dd     -> restore/load leg (cold, hits device)

Run as root (RAPL + drop_caches):
    sudo -E $(which python) scripts/characterize_storage_sweep.py \
        --min-gb 2 --max-gb 50 --step-gb 2 --repeat 2 \
        --tiers nvme_raid0=/var/data,sas_ssd=/home/test

NOTE: SAS is ~0.5 GB/s, so the full 2->50 GB sweep on SAS is SLOW (~1-2 h for repeat 2).
Use nohup, or run NVMe and SAS as separate invocations. Needs >= max-gb free on each
tier's filesystem (check `df -h`).
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
from _common import build_telemetry, write_record                 # noqa: E402

_BS = 1 << 26  # 64 MB, O_DIRECT block-aligned


def drop_caches() -> None:
    os.system("sync")
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
    except OSError as e:
        print(f"[sweep] WARN: can't drop caches ({e}); reads may hit page cache")


def dd_write(path: str, nbytes: int) -> int:
    count = nbytes // _BS
    subprocess.run(["dd", "if=/dev/zero", f"of={path}", f"bs={_BS}", f"count={count}",
                    "oflag=direct", "conv=fdatasync"], check=True, capture_output=True)
    return count * _BS


def dd_read(path: str, nbytes: int) -> int:
    count = nbytes // _BS
    subprocess.run(["dd", f"if={path}", "of=/dev/null", f"bs={_BS}", f"count={count}",
                    "iflag=direct"], check=True, capture_output=True)
    return count * _BS


def cpu_abs_j(rec) -> float:
    for s in rec.sources:
        if s.name == "cpu_pkg_energy_rapl" and s.energy_abs_j is not None:
            return s.energy_abs_j
    return 0.0


def lstsq(xs, ys):
    """Fit y = a + b*x -> (intercept a, slope b)."""
    n = len(xs)
    if n < 2:
        return (ys[0] if ys else 0.0, 0.0)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    return (my - b * mx, b)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description="storage size sweep + affine fit per tier")
    ap.add_argument("--min-gb", type=float, default=2.0)
    ap.add_argument("--max-gb", type=float, default=50.0)
    ap.add_argument("--step-gb", type=float, default=2.0)
    ap.add_argument("--tiers", default="nvme_raid0=/var/data,sas_ssd=/home/test",
                    help="name=dir,name=dir")
    ap.add_argument("--baseline", type=float, default=5.0)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=0)
    args = ap.parse_args()

    tiers = dict(kv.split("=", 1) for kv in args.tiers.split(","))
    sizes_gb = []
    g = args.min_gb
    while g <= args.max_gb + 1e-9:
        sizes_gb.append(round(g, 3))
        g += args.step_gb
    print(f"[sweep] tiers={tiers}  sizes={sizes_gb} GB  repeat={args.repeat}")

    tele = build_telemetry()
    tele.start()
    results = {}                                          # name -> list of per-size dicts
    try:
        for name, d in tiers.items():
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, "socc_sizesweep.bin")
            print(f"\n[sweep] ===== tier {name} ({d}) =====")
            per_size = []
            for sg in sizes_gb:
                nbytes = (int(round(sg * 1e9)) // _BS) * _BS  # block-align
                gb = nbytes / 1e9
                wl, wj, rl, rj = [], [], [], []
                for i in range(args.warmup + args.repeat):
                    rec_w = measure_operation(
                        tele, workload=f"storsweep:{name}", operation="write",
                        state_bytes=nbytes, baseline_seconds=args.baseline,
                        op=lambda p=path, n=nbytes: dd_write(p, n),
                        config={"tier": name, "dir": d, "size_gb": gb, "phase": "write"})
                    drop_caches()
                    rec_r = measure_operation(
                        tele, workload=f"storsweep:{name}", operation="read",
                        state_bytes=nbytes, baseline_seconds=args.baseline,
                        op=lambda p=path, n=nbytes: dd_read(p, n),
                        config={"tier": name, "dir": d, "size_gb": gb, "phase": "read"})
                    if i >= args.warmup:
                        wl.append(rec_w.latency_s); wj.append(cpu_abs_j(rec_w))
                        rl.append(rec_r.latency_s); rj.append(cpu_abs_j(rec_r))
                        write_record(rec_w, "storage_sweep"); write_record(rec_r, "storage_sweep")
                try:
                    os.remove(path)
                except OSError:
                    pass
                row = dict(gb=gb,
                           w_lat=statistics.mean(wl), w_j=statistics.mean(wj),
                           r_lat=statistics.mean(rl), r_j=statistics.mean(rj))
                row["w_bw"] = gb / row["w_lat"] if row["w_lat"] else 0.0
                row["r_bw"] = gb / row["r_lat"] if row["r_lat"] else 0.0
                row["w_pow"] = row["w_j"] / row["w_lat"] if row["w_lat"] else 0.0
                row["r_pow"] = row["r_j"] / row["r_lat"] if row["r_lat"] else 0.0
                per_size.append(row)
                print(f"  [{gb:5.1f} GB] write {row['w_lat']:7.2f}s {row['w_bw']:5.2f}GB/s "
                      f"{row['w_pow']:5.0f}W {row['w_j']:7.0f}J | read {row['r_lat']:7.2f}s "
                      f"{row['r_bw']:5.2f}GB/s {row['r_pow']:5.0f}W {row['r_j']:7.0f}J")
            results[name] = per_size
    finally:
        tele.stop()

    # --- affine fits: E = a + b*S and lat = a + b*S, per tier per direction ---
    print(f"\n=== AFFINE FITS  y = a + b*S   (S in GB) ===")
    print(f"{'tier':12}{'op':6}{'E: a(J)':>9}{'b(J/GB)':>9}{'lat: a(s)':>11}{'b(s/GB)':>9}"
          f"{'BW(GB/s)':>10}{'avgPow(W)':>11}")
    for name, rows in results.items():
        S = [r["gb"] for r in rows]
        for op, lk, jk, pk in (("write", "w_lat", "w_j", "w_pow"), ("read", "r_lat", "r_j", "r_pow")):
            aE, bE = lstsq(S, [r[jk] for r in rows])
            aL, bL = lstsq(S, [r[lk] for r in rows])
            bw = 1.0 / bL if bL else 0.0
            pw = statistics.mean([r[pk] for r in rows])
            print(f"{name:12}{op:6}{aE:9.0f}{bE:9.1f}{aL:11.3f}{bL:9.4f}{bw:10.2f}{pw:11.0f}")
    print("\nE: a = fixed-overhead energy (J), b = per-byte energy (J/GB). "
          "lat: b = 1/bandwidth. avgPow ~ const across sizes => energy is the time term.")
    print("Raw per-op records -> data/storage_sweep.jsonl")


if __name__ == "__main__":
    main()
