#!/usr/bin/env python3
"""Concurrent multi-GPU HBM<->host PCIe: measure AGGREGATE bandwidth + energy when
N GPUs transfer at the same time -- the TP=N suspend/resume contention scenario.

Sweeps the number of concurrent GPUs (1..4) so you see whether aggregate bandwidth
scales linearly (independent PCIe links) or CONTENDS on the shared host-DRAM write /
IO-die path. The per-GPU bandwidth at N=4 is the rate to feed into a TP=4 dump model.

Each GPU runs its own chunked copy in a thread; a barrier starts them together so the
measured window is genuinely concurrent. Energy = sum over the N involved GPUs (NVML
scoped to gpus[:N]) + CPU package (RAPL), absolute (incl. idle floor).

    sudo -E $(which python) scripts/characterize_bw_multigpu.py \
        --bytes 16e9 --gpus 0,1,2,3 --counts 1,2,3,4 --dir both --repeat 3 --warmup 1

bytes is PER GPU. Run as root for RAPL. GPUs must be free.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from harness import measure_operation                              # noqa: E402
from _common import build_telemetry, write_record                 # noqa: E402

_CHUNK = 128 << 20  # 128 MB per chunk


def make_bufs(gpus, direction):
    """Pre-allocate one chunk buffer per GPU (device + pinned host), reused across
    trials so allocation isn't in the timed window."""
    import torch
    n_elem = _CHUNK // 2
    bufs = {}
    for g in gpus:
        dev = torch.device(f"cuda:{g}")
        host = torch.empty(n_elem, dtype=torch.float16, pin_memory=True)
        gbuf = torch.empty(n_elem, dtype=torch.float16, device=dev)
        torch.cuda.synchronize(dev)
        bufs[g] = (host, gbuf, dev)
    return bufs


def _worker(g, bufs, nchunks, direction, barrier, out, idx):
    import torch
    host, gbuf, dev = bufs[g]
    do = (lambda: gbuf.copy_(host)) if direction == "h2d" else (lambda: host.copy_(gbuf))
    barrier.wait()                                   # all GPUs start together
    for _ in range(nchunks):
        do()
        torch.cuda.synchronize(dev)
    out[idx] = nchunks * _CHUNK


def concurrent_copy(gpus, bufs, nbytes, direction):
    nchunks = max(1, nbytes // _CHUNK)
    barrier = threading.Barrier(len(gpus))
    out = [0] * len(gpus)
    threads = [threading.Thread(target=_worker,
                                args=(g, bufs, nchunks, direction, barrier, out, i))
               for i, g in enumerate(gpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return sum(out)


def gpu_abs(rec) -> float:
    for s in rec.sources:
        if s.name == "nvml_gpu_pkg" and getattr(s, "energy_abs_j", None) is not None:
            return s.energy_abs_j
    return 0.0


def cpu_abs(rec) -> float:
    for s in rec.sources:
        if s.name == "cpu_pkg_energy_rapl" and getattr(s, "energy_abs_j", None) is not None:
            return s.energy_abs_j
    return 0.0


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description="concurrent N-GPU HBM<->host bandwidth + energy")
    ap.add_argument("--bytes", type=float, default=16e9, help="bytes moved PER GPU")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--counts", default="1,2,3,4", help="how many concurrent GPUs to sweep")
    ap.add_argument("--dir", choices=["d2h", "h2d", "both"], default="both")
    ap.add_argument("--baseline", type=float, default=5.0)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    all_gpus = [int(x) for x in args.gpus.split(",")]
    counts = [int(x) for x in args.counts.split(",")]
    dirs = ["d2h", "h2d"] if args.dir == "both" else [args.dir]
    nbytes = int(args.bytes)
    S_gb = nbytes / 1e9

    results = {}                                     # dir -> list of per-count dicts
    for d in dirs:
        leg = "suspend/extract (HBM->host)" if d == "d2h" else "restore (host->HBM)"
        print(f"\n[multi] ===== {d}  {leg}  ({S_gb:.0f} GB/GPU) =====")
        rows = []
        for n in counts:
            gpus_n = all_gpus[:n]
            bufs = make_bufs(gpus_n, d)
            tele = build_telemetry(nvml_gpus=gpus_n)     # scope to the N GPUs in use
            tele.start()
            lat, gj, cj = [], [], []
            try:
                for i in range(args.warmup + args.repeat):
                    rec = measure_operation(
                        tele, workload=f"multigpu:{d}", operation=f"pcie_{d}_x{n}",
                        state_bytes=nbytes * n, baseline_seconds=args.baseline,
                        op=lambda: concurrent_copy(gpus_n, bufs, nbytes, d),
                        config={"dir": d, "gpus": gpus_n, "n": n})
                    if i >= args.warmup:
                        lat.append(rec.latency_s); gj.append(gpu_abs(rec)); cj.append(cpu_abs(rec))
                        write_record(rec, "bw_multigpu")
            finally:
                tele.stop()
            del bufs
            lm = statistics.mean(lat)
            agg_bw = (nbytes * n) / lm / 1e9              # aggregate GB/s
            per_bw = agg_bw / n                           # per-GPU GB/s
            gjm, cjm = statistics.mean(gj), statistics.mean(cj)
            tot = gjm + cjm
            rows.append(dict(n=n, lat=lm, agg_bw=agg_bw, per_bw=per_bw,
                             gpu=gjm, cpu=cjm, tot=tot, pw=tot / lm if lm else 0.0))
            print(f"  [n={n}] lat {lm:5.2f}s  per-GPU {per_bw:5.1f}GB/s  agg {agg_bw:6.1f}GB/s  "
                  f"gpu {gjm:5.0f}J  cpu {cjm:5.0f}J  tot {tot:5.0f}J  {tot/lm if lm else 0:4.0f}W")
        results[d] = rows

    # --- scaling summary: agg vs ideal n*per_bw(n=1) ---
    print(f"\n=== MULTI-GPU SCALING  (per-GPU & aggregate BW, scaling efficiency) ===")
    print(f"{'dir':5}{'n':>3}{'per_GB/s':>10}{'agg_GB/s':>10}{'scaling':>9}{'tot_J':>8}"
          f"{'J/GB':>8}{'tot_W':>8}")
    for d, rows in results.items():
        base = rows[0]["agg_bw"] if rows else 0.0        # n=1 aggregate = single-GPU BW
        for r in rows:
            ideal = base * r["n"]
            eff = r["agg_bw"] / ideal if ideal else 0.0
            jpg = r["tot"] / (S_gb * r["n"]) if r["n"] else 0.0
            print(f"{d:5}{r['n']:>3}{r['per_bw']:>10.1f}{r['agg_bw']:>10.1f}{eff:>8.0%}"
                  f"{r['tot']:>8.0f}{jpg:>8.1f}{r['pw']:>8.0f}")
    print("\nscaling = agg_bw(n) / (n * single-GPU agg_bw). 100% = no contention (independent")
    print("PCIe links); <100% = host-DRAM/IO-die contention. per-GPU BW at n=4 feeds TP=4 model.")
    print("J/GB = total energy / total bytes moved. Raw -> data/bw_multigpu.jsonl")


if __name__ == "__main__":
    main()
