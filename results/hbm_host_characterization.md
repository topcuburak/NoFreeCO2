# HBM <-> host (PCIe) characterization (appendix data)

**Testbed:** ford (A100 SXM4 40GB, PCIe Gen4 x16). **Date:** 2026-06-02.
**Tool:** `scripts/characterize_hbm_host.py`
**Config:** `--bytes 16e9 --gpu 1 --repeat 8 --warmup 2` (raw cudaMemcpy, pinned host, iters=1).
**Energy:** `gpu_marg` = GPU package energy above idle (NVML, the DMA work); `cpu_abs` = CPU package energy incl. idle over the op (RAPL, time term). DRAM modeled.

## Summary (mean ± std, n=8, 2 warmup discarded)

| direction | leg | GB/s | latency_s | gpu_marg_J | cpu_abs_J |
|---|---|---|---|---|---|
| hbm_to_host (D2H) | dump / extract | 25.42 ± 0.03 | 0.630 ± 0.001 | 5 ± 1 | 87 ± 1 |
| host_to_hbm (H2D) | restore | 24.57 ± 0.01 | 0.651 ± 0.000 | 2 ± 3 | 89 ± 1 |

**~25 GB/s both directions — near PCIe Gen4 x16 peak, and ~symmetric (no dump/restore asymmetry on this leg).**

## Raw per-trial (16 GB each)

### hbm_to_host (D2H)
| trial | latency_s | gpu_J | cpu_abs_J |
|---|---|---|---|
| warmup | **12.046** | 184 | 1636 |
| warmup | 0.627 | 6 | 87 |
| t1 | 0.628 | 6 | 86 |
| t2 | 0.629 | 5 | 88 |
| t3 | 0.629 | 6 | 87 |
| t4 | 0.630 | 7 | 87 |
| t5 | 0.630 | 4 | 88 |
| t6 | 0.630 | 5 | 87 |
| t7 | 0.630 | 3 | 87 |
| t8 | 0.630 | 5 | 88 |

### host_to_hbm (H2D)
| trial | latency_s | gpu_J | cpu_abs_J |
|---|---|---|---|
| warmup | 0.651 | 7 | 89 |
| warmup | 0.651 | 4 | 89 |
| t1 | 0.651 | -0 | 91 |
| t2 | 0.651 | 0 | 89 |
| t3 | 0.651 | -1 | 89 |
| t4 | 0.651 | 2 | 90 |
| t5 | 0.651 | 4 | 89 |
| t6 | 0.651 | 6 | 89 |
| t7 | 0.651 | 4 | 90 |
| t8 | 0.651 | 6 | 89 |

## Observations

- **First D2H trial was 12.05 s (vs 0.63 s steady)** — one-time CUDA context init / pinned-buffer registration / first-touch page faults. Correctly discarded by `--warmup` (this is *why* warmup matters).
- **GPU marginal is tiny (~2–5 J) and near noise**: a pure DMA copy barely raises GPU package power above idle (the SMs are idle; only the copy engine + PCIe PHY work, and the PCIe controller is already in the idle baseline).
- **`cpu_abs` (~87–89 J) dominates the measured energy** and is time-term: ~138 W over 0.63 s ≈ idle 121 W + ~17 W marginal for driving the DMA. So even here the time term carries the cost.
- **Symmetric**: D2H and H2D nearly identical in BW *and* energy. So the dump/restore asymmetry in the full path comes entirely from the **storage** leg (write 2x slower than read), NOT PCIe.

## Caveat: capability vs real path
This is the **raw PCIe DMA ceiling** (~25 GB/s). The real `cuda-checkpoint` extract is ~5 GB/s (~4-5x slower) due to lock/quiesce + per-allocation free + non-optimal copies. Real-path numbers: `timed_dump` (cuda-checkpoint suspend/resume); clean absolute energy still to be extracted.

## Reproduce
```
sudo -E $(which python) scripts/characterize_hbm_host.py --bytes 16e9 --gpu 1 --repeat 8 --warmup 2
```
Raw per-op JSON: `data/hbm_host_char.jsonl`.
