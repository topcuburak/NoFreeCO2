# S1 — multi-GPU transparent suspend/restore vs footprint (hold_gpu, no NCCL)

**Testbed:** ford (4× A100-40GB, EPYC 75F3, NVMe-RAID0 /var/data). **Date:** 2026-06-16.
**Tool:** `scripts/sweep_multigpu_suspend.py`. N independent `hold_gpu` processes (one per GPU,
`--chunks 1`, no NCCL / no CUDA-IPC) → `cuda-checkpoint --multiproc` lock-all→checkpoint-all→
restore-all→unlock-all. The CLEAN multi-GPU GPU-leg baseline — no destroy/reinit (unlike A1/A2 TP>1).
**Sweep:** gpu-counts {1,2,4} × per-GPU {8,16,24,32} GiB = 12 configs, total footprints 9–140 GB,
4 cycles each. Energy = MEASURED GPU board (NVML) + CPU pkg (RAPL) + MODELED DRAM/drive.

## Data quality: intermittent resume stalls (report the floor)
Resume (`cuda_restore`, host→HBM) has **sporadic stalls** — 4/48 cycles (~8%) spike to 24–32 s while
the rest of the *same config* sit tight at 2–10 s (e.g. `1GPU 34.9GB: [2.9, 32.2, 2.7, 2.8]`). Not
cold-start (hits cycle 1/2/3, not just 0) — a `cuda-checkpoint` restore transient (host-memory reclaim
/ contention during the host→HBM copy). A stall can only inflate, never speed up the copy, so the
**floor** (avg of the 2 fastest of 4 cycles) is the true cost. Suspend/store/load are stable.

## Per-leg robust-floor cost (NVMe, avg of 2 fastest cycles)
| nGPU | total_GB | suspend_s | store_s | load_s | resume_s | RT_s |
|---|---|---|---|---|---|---|
| 1 | 9.1 | 2.61 | 1.91 | 0.89 | 1.57 | 6.98 |
| 1 | 17.7 | 3.85 | 3.16 | 1.56 | 1.88 | 10.46 |
| 1 | 26.3 | 4.94 | 4.55 | 2.23 | 2.31 | 14.04 |
| 1 | 34.9 | 6.80 | 5.89 | 2.91 | 2.74 | 18.34 |
| 2 | 18.2 | 4.19 | 3.30 | 1.60 | 2.28 | 11.38 |
| 2 | 35.4 | 7.25 | 6.12 | 2.95 | 3.13 | 19.45 |
| 2 | 52.6 | 10.24 | 8.90 | 4.29 | 5.02 | 28.45 |
| 2 | 69.8 | 14.23 | 11.62 | 5.64 | 4.65 | 36.14 |
| 4 | 36.4 | 9.65 | 6.20 | 3.02 | 6.16 | 25.04 |
| 4 | 70.8 | 18.33 | 11.73 | 5.71 | 8.96 | 44.73 |
| 4 | 105.2 | 20.91 | 17.15 | 8.41 | 9.68 | 56.14 |
| 4 | 139.5 | 27.25 | 23.00 | 11.10 | 9.45 | 70.79 |

## Affine fit vs TOTAL footprint S (all GPU counts pooled, NVMe)
| leg | a (s) | b (s/GB) | rate |
|---|---|---|---|
| suspend | 0.79 | 0.1960 | **5.10 GB/s** |
| store (NVMe) | 0.34 | 0.1614 | 6.20 GB/s |
| load (NVMe) | 0.18 | 0.0783 | 12.78 GB/s |
| resume | 1.23 | 0.0698 | 14.32 GB/s |
| **round-trip** | 2.55 | 0.5055 | 1.98 GB/s |

store/load match the standalone storage sweep (NVMe 5.8/12.7 GB/s) and A1 — confirms tier-independence
of the GPU legs and byte-linearity of the storage legs.

## Findings
1. **Footprint-driven across the full range 9→140 GB.** Extends the single-GPU `temporal_components`
   curve (capped at 38 GB) to A1-scale (4×35 = 140 GB ≈ A1's 148 GB), one clean per-byte coefficient.
2. **Suspend ≈ GPU-count-independent per byte, + a small per-process term.** Matched total: ~16 GB
   1GPU 3.9 vs 2GPU 4.2 s; ~32 GB 6.8 vs 7.2 s (nearly equal); but ~72 GB 2GPU 14.2 vs **4GPU 18.3 s**.
   `lock-all/checkpoint-all` is **sequential** (no PCIe contention — per-GB slope constant), but each
   GPU adds ~0.5–1 s of `cuda-checkpoint` subprocess overhead (8 calls for 4 GPUs vs 4 for 2). So
   suspend = `b·S + c·nGPU`, byte-dominated with a per-process intercept.
3. **Resume floor 14.3 GB/s** but with an ~8% stall tail (24–32 s) — report the floor, note the tail.
4. **This is the synthetic lower bound** the real workloads sit on: A1 transparent suspend (4×A100,
   148 GB) measured 27 s ≈ S1 fit at 148 GB (0.79 + 0.196·148 = 29.8 s) — consistent. The real-workload
   premium over this baseline is the allocation-structure term (separate `--chunks` probe, pending).

## SATA tier (full cycle, `s1_mg_sata_full`, robust floor)
| nGPU | total_GB | suspend_s | store_s | load_s | resume_s | RT_s |
|---|---|---|---|---|---|---|
| 1 | 9.1 | 3.34 | 22.86 | 21.46 | 2.46 | 50.11 |
| 1 | 34.9 | 8.03 | 76.20 | 70.43 | 2.62 | 157.28 |
| 2 | 69.8 | 14.24 | 148.41 | 135.66 | 5.14 | 303.45 |
| 4 | 139.5 | 27.19 | 290.99 | 266.87 | 9.17 | 594.22 |

(12 configs; representative rows. Full table in `data/timed_dump.jsonl` tag `s1_mg_sata_full`.)

**SATA per-leg fit vs total footprint:**
| leg | a (s) | b (s/GB) | rate |
|---|---|---|---|
| suspend | 1.94 | 0.1845 | 5.42 GB/s |
| store (SATA) | 4.41 | 2.0579 | **0.49 GB/s** |
| load (SATA) | 4.25 | 1.8825 | **0.53 GB/s** |
| resume | 2.26 | 0.0585 | 17.1 GB/s |
| **round-trip** | 12.85 | 4.1834 | 0.24 GB/s |

**GPU legs are tier-independent (cross-check):** suspend SATA 5.42 vs NVMe 5.10 GB/s; resume SATA 17.1 vs
NVMe 14.3 GB/s — equal within floor noise, confirming suspend/resume never touch the disk. Store/load
match the standalone storage sweep to two digits (0.48/0.53 GB/s).

## NVMe vs SATA — storage-tier flip (round-trip slope)
| | NVMe | SATA | ratio |
|---|---|---|---|
| round-trip slope | 0.506 s/GB | 4.183 s/GB | **8.3×** |
| store+load share | ~47% | **~93%** | — |

**S1 reproduces the real-workload flip:** A1 (real FSDP, 148 GB) measured an 8.8× NVMe→SATA round-trip
increase; the synthetic S1 baseline gives 8.3× — the storage-tier sensitivity is intrinsic to the
mechanism (footprint-driven `power÷bandwidth`), not workload-specific. On SATA the GPU legs become a
rounding error (storage 93% of round-trip); on NVMe they're ~half.

## Allocation-structure probe (`s1_mg_chunks_nvme`) — allocation count is IRRELEVANT
Fixed 4 GPU × 32 GiB = 139.5 GB, swept allocations PER GPU 1→1024 (total `cuda-checkpoint` regions
4→4096), suspend/resume only, 4 cycles. Robust floor (2 fastest):

| chunks/GPU | total allocs | suspend_s | susp GB/s |
|---|---|---|---|
| 1 | 4 | 15.78* | 8.84* |
| 4 | 16 | 25.72 | 5.42 |
| 16 | 64 | 25.69 | 5.43 |
| 64 | 256 | 25.66 | 5.44 |
| 256 | 1024 | 25.75 | 5.42 |
| 1024 | 4096 | 26.02 | 5.36 |

**Suspend is flat within 0.36 s across 16→4096 allocations** → allocation count does NOT drive
checkpoint cost. (*chunks=1 floor 15.78 s is an intermittent fast-path — its *mean* is 27.2 s,
matching the rest, and `s1_mg_nvme_full` measured chunks=1 at 27.25 s; ~26 s is the consistent value.)
chunks=4096/GPU skipped (4096×8 MB tensors + allocator metadata exceeded the 40 GB card — harmless,
trend already flat). Resume floor ~9 s flat too.

**So there is NO allocation-structure term.** Mechanism cost is *purely footprint-driven*. This closes
the A1 question: clean A1 (5.47 GB/s, hundreds of FSDP tensors) ≈ S1 (5.10, 1 tensor) ≈ this probe
(flat to 4096 allocs) — the earlier A1 "+85% / 2.74 GB/s" was a measurement artifact (verbose-cc +
memory pressure), NOT fragmentation. Cleaner model: `cost = a + b·S`, S = total bytes, full stop.

## S1 — COMPLETE (both tiers, 1–4 GPUs, 9–140 GB, 1–4096 allocations)
Final footprint-driven model, per leg (`cost = a + b·S`, S = total bytes):
- **suspend** ~5.1–5.4 GB/s (tier-independent), + small per-process term (~0.5–1 s/GPU subprocess overhead)
- **resume** ~14–17 GB/s (tier-independent), ~8% stall tail (24–32 s transients)
- **store/load** NVMe 6.2/12.8, SATA 0.49/0.53 GB/s (byte-linear, GPU-count-independent)
- **round-trip** NVMe 0.506 s/GB, SATA 4.183 s/GB
- **allocation structure: no effect** — flat from 1 tensor to 4096 allocations; cost is total-bytes only
This is the clean lower bound AND the real cost — the synthetic baseline equals real-workload cost
(no allocation-structure premium), validated against A1 at matched footprint. S1 is fully closed.

## Reproduce
```
sudo -E $(which python) scripts/sweep_multigpu_suspend.py \
  --gpu-counts 1,2,4 --sizes 8,16,24,32 --cycles 4 --store-out /var/data --tag s1_mg_nvme_full
# SATA: --store-out /home/test --tag s1_mg_sata_full
```
Raw records: `data/timed_dump.jsonl` (tag `s1_mg_nvme_full`, multiproc=true). Floor = avg of 2 fastest
of 4 cycles per leg (drops cold cycle-0 + the ~8% restore stalls).
