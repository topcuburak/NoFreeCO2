# Concurrent multi-GPU HBM↔host PCIe — bandwidth contention + power (standalone)

**Testbed:** ford (4× A100-40GB, EPYC 75F3). **Date:** 2026-06-08.
**Tool:** `scripts/characterize_bw_multigpu.py` (N GPUs each run a chunked copy in a
thread, barrier-started so the window is genuinely concurrent). **16 GB/GPU**, 3 trials
+ 1 warmup. **Energy = `∫P dt`:** `nvml_gpu_pkg` summed over the N GPUs + `cpu_pkg_energy_rapl`.
DRAM modeled (no DRAM RAPL on EPYC); rest-of-node not captured (no BMC/PDU).

Scope note: standalone experiment + its model; no cross-reference to other results.

## Scaling (1→4 concurrent GPUs)

| dir | n | per-GPU GB/s | aggregate GB/s | scaling | lat (s) | gpu J | cpu J | tot J | J/GB | tot W |
|---|---|---|---|---|---|---|---|---|---|---|
| d2h | 1 | 24.4 | 24.4 | 100% | 0.65 | 55 | 96 | 151 | 9.4 | 230 |
| d2h | 2 | 23.3 | 46.7 | 96% | 0.69 | 115 | 107 | 222 | 6.9 | 324 |
| d2h | 3 | 15.7 | 47.2 | 64% | 1.02 | 287 | 164 | 451 | 9.4 | 443 |
| d2h | 4 | 12.0 | 47.9 | 49% | 1.34 | 533 | 226 | 759 | 11.9 | 567 |
| h2d | 1 | 24.5 | 24.5 | 100% | 0.65 | 53 | 94 | 147 | 9.2 | 225 |
| h2d | 2 | 24.5 | 49.0 | 100% | 0.65 | 111 | 100 | 211 | 6.6 | 323 |
| h2d | 3 | 16.4 | 49.2 | 67% | 0.98 | 277 | 154 | 431 | 9.0 | 442 |
| h2d | 4 | 12.4 | 49.6 | 51% | 1.29 | 514 | 214 | 728 | 11.4 | 564 |

`d2h` = HBM→host (extract), `h2d` = host→HBM. Raw → `data/bw_multigpu.jsonl`.

## Findings

1. **Aggregate HBM↔host bandwidth caps at ~48–50 GB/s.** Scales perfectly to 2 GPUs
   (~47–49 GB/s, ~96–100%), then **plateaus** — n=3 and n=4 give the *same* aggregate, so
   the extra GPUs just split a fixed pie. Per-GPU drops 24 → 12 GB/s at n=4. The PCIe
   links are **not independent** for host-bound DMA; ~48 GB/s (~2× a single x16 Gen4) is
   the host PCIe-root / IO-die ceiling (DRAM channels ~200 GB/s are far above, so not the
   limit). Both directions hit the same wall.

2. **Per-GPU rate at n=4 ≈ 12 GB/s** — half the solo 24. This is the contended
   HBM↔host coefficient for a 4-GPU concurrent suspend/resume.

3. **Contention raises per-GPU POWER while lowering throughput** — doubly wasteful.
   Per-GPU draw climbs 85 W (n=1) → 100 W (n=4) even as per-GPU BW halves: the DMA
   engines are powered and stalling on the contended host path. Per-byte energy rises
   9.4 → 11.9 J/GB.

4. **n=2 is the sweet spot** (full parallelism, no contention): 2× bytes in the same
   time amortizes the CPU floor → lowest per-byte energy (6.6–6.9 J/GB).

## Power breakdown (per the measured components, + modeled DRAM)

| component | model | value |
|---|---|---|
| **GPU** (measured, ×n) | idle floor + DMA | ~55 W floor + ~27 W DMA (full speed) → ~45 W (contended) ⇒ **85→100 W/GPU** |
| **CPU pkg** (measured) | floor + coordination | ~140 W floor + 8 W (n=1) → 29 W (n=4) ⇒ **148→169 W** |
| **DRAM** (MODELED, not in tot_W) | dynamic at aggregate BW | ~**30 W** [20–40] at the ~48 GB/s aggregate (≈0.6 W per GB/s; DDR4 dynamic + DIMM idle) |

```
P_total(n) ≈ n·(55 W GPU floor + P_dma) + (140 W CPU floor + coord) + DRAM(~30 W, modeled)
```
So the true total at n=4 d2h ≈ 398 (GPU) + 169 (CPU) + ~30 (DRAM) ≈ **~600 W**
(vs the 567 W measured, which excludes DRAM + platform).

## Use
Per-GPU HBM↔host bandwidth at n=4 (~12 GB/s) and the ~48 GB/s aggregate ceiling are the
contended-rate inputs for a TP=4 suspend/resume model (the transparent TP>1 checkpoint
itself being infeasible). Contention makes suspend time grow super-linearly in GPU count
past 2.

## Reproduce
```
sudo -E $(which python) scripts/characterize_bw_multigpu.py \
  --bytes 16e9 --gpus 0,1,2,3 --counts 1,2,3,4 --dir both --repeat 3 --warmup 1
```
