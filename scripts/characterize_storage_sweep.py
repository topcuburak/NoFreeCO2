#!/usr/bin/env python3
"""Storage size sweep: store/read a range of sizes (e.g. 2->50 GB) on each tier and
measure latency + CPU power + energy per size, plus a MODELED drive-power term, then
fit the affine cost model

    E(S)   = a_E + b_E * S       (J;  a_E = fixed overhead, b_E = per-byte energy)
    lat(S) = a_L + b_L * S       (s;  b_L = 1/bandwidth)

Single-size characterize_storage.py gives only a slope at one point; this sweep
separates the FIXED overhead (intercept) from the per-byte slope, and shows whether
power is size-independent (it should be ~the node floor).

Energy domains:
  cpu_W  = MEASURED CPU package power (RAPL energy_abs / latency)
  drive  = MODELED: n_drives * P_active * latency. P_active from `nvme id-ctrl` power
           state mp (rated ceiling; actual draw <= this); SATA from datasheet (no
           live power telemetry). NOT measured -- ford has no BMC/PDU/wall meter.
  DRAM   = not included here (modeled elsewhere; no DRAM RAPL on EPYC).

  WRITE: O_DIRECT dd (durable, fdatasync) -> dump/store leg
  READ : drop page cache, O_DIRECT dd     -> restore/load leg (cold, hits device)

Run as root (RAPL + drop_caches):
    sudo -E $(which python) scripts/characterize_storage_sweep.py \
        --min-gb 2 --max-gb 50 --step-gb 2 --repeat 2 \
        --tiers nvme_raid0=/var/data,sas_ssd=/home/test \
        --drive nvme_raid0=25:2,sas_ssd=3:1

SAS is ~0.5 GB/s -> the full sweep on SAS is SLOW (~45 min at repeat 1). Use nohup or
separate invocations. Needs >= max-gb free on each tier (`df -h`).
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
    ap.add_argument("--drive", default="nvme_raid0=25:2,sas_ssd=3:1",
                    help="MODELED drive power per tier: name=activeW:nDrives "
                         "(NVMe activeW = `nvme id-ctrl` PS0 mp, RAID0 nDrives=2; "
                         "SATA from datasheet). Rated upper bound, not measured.")
    ap.add_argument("--baseline", type=float, default=5.0)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=0)
    args = ap.parse_args()

    tiers = dict(kv.split("=", 1) for kv in args.tiers.split(","))
    drive = {}
    for kv in args.drive.split(","):
        nm, val = kv.split("=", 1)
        p, _, n = val.partition(":")
        drive[nm] = (float(p), int(n or 1))

    sizes_gb = []
    g = args.min_gb
    while g <= args.max_gb + 1e-9:
        sizes_gb.append(round(g, 3))
        g += args.step_gb
    print(f"[sweep] tiers={tiers}  drive(model)={drive}  sizes={sizes_gb} GB  repeat={args.repeat}")

    tele = build_telemetry()
    tele.start()
    results = {}
    try:
        for name, d in tiers.items():
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, "socc_sizesweep.bin")
            P_drv, n_drv = drive.get(name, (0.0, 1))
            drive_w = P_drv * n_drv                              # modeled drive power (const)
            print(f"\n[sweep] ===== tier {name} ({d})  drive {n_drv}x{P_drv:.0f}W={drive_w:.0f}W (modeled) =====")
            per_size = []
            for sg in sizes_gb:
                nbytes = (int(round(sg * 1e9)) // _BS) * _BS
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
                r = dict(gb=gb, drive_w=drive_w,
                         w_lat=statistics.mean(wl), w_cpu=statistics.mean(wj),
                         r_lat=statistics.mean(rl), r_cpu=statistics.mean(rj))
                r["w_bw"] = gb / r["w_lat"] if r["w_lat"] else 0.0
                r["r_bw"] = gb / r["r_lat"] if r["r_lat"] else 0.0
                r["w_drv"] = drive_w * r["w_lat"]                # modeled drive energy
                r["r_drv"] = drive_w * r["r_lat"]
                r["w_tot"] = r["w_cpu"] + r["w_drv"]
                r["r_tot"] = r["r_cpu"] + r["r_drv"]
                r["w_cpu_w"] = r["w_cpu"] / r["w_lat"] if r["w_lat"] else 0.0
                r["r_cpu_w"] = r["r_cpu"] / r["r_lat"] if r["r_lat"] else 0.0
                per_size.append(r)
                print(f"  [{gb:5.1f}GB] W {r['w_lat']:6.2f}s {r['w_bw']:5.2f}GB/s "
                      f"cpu{r['w_cpu']:5.0f}J(+drv{r['w_drv']:4.0f})=tot{r['w_tot']:5.0f}J | "
                      f"R {r['r_lat']:6.2f}s {r['r_bw']:5.2f}GB/s "
                      f"cpu{r['r_cpu']:5.0f}J(+drv{r['r_drv']:4.0f})=tot{r['r_tot']:5.0f}J")
            results[name] = per_size
    finally:
        tele.stop()

    # --- affine fits: cpu-only AND cpu+drive total, plus latency, per tier/dir ---
    print(f"\n=== AFFINE FITS  y = a + b*S  (S in GB) ===")
    print(f"{'tier':12}{'op':6}{'cpuE a(J)':>10}{'b(J/GB)':>9}{'totE a(J)':>10}{'b(J/GB)':>9}"
          f"{'b_lat(s/GB)':>12}{'BW(GB/s)':>10}{'cpu_W':>8}")
    for name, rows in results.items():
        S = [x["gb"] for x in rows]
        for op, lk, ck, tk, pk in (("write", "w_lat", "w_cpu", "w_tot", "w_cpu_w"),
                                    ("read", "r_lat", "r_cpu", "r_tot", "r_cpu_w")):
            aC, bC = lstsq(S, [x[ck] for x in rows])
            aT, bT = lstsq(S, [x[tk] for x in rows])
            aL, bL = lstsq(S, [x[lk] for x in rows])
            bw = 1.0 / bL if bL else 0.0
            cw = statistics.mean([x[pk] for x in rows])
            print(f"{name:12}{op:6}{aC:10.0f}{bC:9.1f}{aT:10.0f}{bT:9.1f}{bL:12.4f}{bw:10.2f}{cw:8.0f}")
    print("\ncpuE = MEASURED CPU pkg (a=fixed overhead, b=per-byte). totE = + MODELED drive")
    print("(n*Pactive*t). b_lat = 1/bandwidth. Drive power is a rated upper bound, not measured.")
    print("Raw per-op records -> data/storage_sweep.jsonl")


if __name__ == "__main__":
    main()
