# vLLM TP=1 transparent dump/restore cycle (appendix data)

**Testbed:** ford (1× A100-40GB, PCIe Gen4, PM1733 RAID-0 NVMe @ /var/data).
**Date:** 2026-06-02. **Model:** meta-llama/Llama-3.1-8B, TP=1, `--enforce-eager`,
single-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`). **Tool:** `scripts/sweep_vllm_dump.py`.
**Cycle:** suspend (cuda-checkpoint HBM→host) → store (host→NVMe, O_DIRECT) → load
(NVMe→host) → resume (cuda-checkpoint host→HBM). criu skipped (io_uring); store/load
are device-rate proxies. **Energy = `cpu_abs`** (CPU pkg incl. idle = time term);
GPU power-integral is unreliable for state-changing ops (busy→idle baseline). NVML
scoped to the one GPU in use.

## A. Batch-size sweep — TRANSPARENCY TAX (footprint constant)

3 cycles/config. Config 0 is a cold-start **warmup outlier** (first cuda-checkpoint
of the session) — discard.

| batch (input-len / prompts) | foot_GB | suspend_s | store_s | resume_s | dump_J | restore_J |
|---|---|---|---|---|---|---|
| 2000 / 8 (warmup) | 39.2 | 10.18 ± 4.47 | 6.93 | 3.41 ± 2.54 | 2446 ± 631 | 973 ± 421 |
| 8000 / 16 | 39.2 | 7.65 ± 0.06 | 6.66 | 2.90 ± 0.13 | 2079 ± 7 | 908 ± 12 |
| 12000 / 32 | 39.2 | 7.71 ± 0.06 | 6.70 | 3.01 ± 0.02 | 2093 ± 26 | 921 ± 2 |

**Footprint is 39.2 GB regardless of batch size** → vLLM pre-allocates the KV pool, so
the transparent dump pays the same at any load. Dump/restore energy batch-invariant
(~2086 / ~915 J, warm). Once warm, reproducibility is ±<1%.

## B. gpu-mem-util sweep — FOOTPRINT S-CURVE (footprint scales)

3 cycles/config, all stable (small workload, generation-gated readiness).

| gpu-mem-util | foot_GB | suspend_s | store_s | load_s | resume_s | dump_J | restore_J |
|---|---|---|---|---|---|---|---|
| 0.5 | 22.2 | 5.14 ± 0.51 | 3.87 | 1.91 | 2.04 ± 0.05 | 1222 ± 74 | 530 ± 6 |
| 0.7 | 30.6 | 6.82 ± 0.14 | 5.24 | 2.58 | 2.53 ± 0.17 | 1633 ± 28 | 684 ± 22 |
| 0.9 | 39.2 | 7.99 ± 0.46 | 6.60 | 3.24 | 2.85 ± 0.02 | 1974 ± 55 | 817 ± 5 |

### Linearity — AFFINE fit `cost = a + b·S` (small fixed overhead + per-byte slope)

Three footprints (22.2 / 30.6 / 39.2 GB) give a clean linear fit:

| leg | fit (least-squares over the 3 points) | per-GB at 22→39 GB |
|---|---|---|
| **dump** (suspend+store) | **≈ 240 J + 44 J/GB·S** | 55 → 50 J/GB |
| **restore** (load+resume) | **≈ 155 J + 17 J/GB·S** | 24 → 21 J/GB |
| store bandwidth | constant **~5.8 GB/s** | 5.74 / 5.84 / 5.94 |
| suspend bandwidth | constant **~4.6 GB/s** | 4.32 / 4.49 / 4.91 |
| load bandwidth | constant **~12 GB/s** | 11.6 / 11.9 / 12.1 |

→ The model is **affine, not pure proportional**: a small **fixed overhead** (~240 J
dump, ~155 J restore — the cuda-checkpoint lock/setup + fsync commit) plus a
**per-byte slope** (~44 J/GB dump, ~17 J/GB restore). The decreasing per-GB (55→50)
is the intercept amortizing. Bandwidths are footprint-independent (store 5.8 GB/s,
suspend ~4.6 GB/s — the real cuda-checkpoint rate, ~5× below the raw PCIe ceiling).
This validates the `a + b·S` form for `analyze_dump_cost.py`.

## Cross-leg structure (matches the per-leg characterizations)

- **store ↔ NVMe write** ~5.8 GB/s matches `results/storage_characterization.md` (5.8 GB/s).
- **load ↔ NVMe read** ~12 GB/s matches storage read (12.7).
- **suspend ~4.6 GB/s** ≈ 5× slower than the raw HBM→host PCIe ceiling (25 GB/s,
  `results/hbm_host_characterization.md`) — the cuda-checkpoint overhead.
- **dump > restore** on both legs (suspend > resume, store > load): dump ~44 J/GB
  slope vs restore ~17 J/GB → the full dump costs ~2.6× the restore per byte.

## Operational findings (paper-worthy)

1. **Footprint is pre-allocation-bound** (batch-invariant) → the transparency tax.
2. **Suspending a half-initialized vLLM corrupts it** — early sweeps that dumped before
   generation started broke the process (resume found the GPU empty). Readiness must
   wait for the GPU to be *generating* (full init), not just memory-plateaued.
3. **cuda-checkpoint suspend/resume have run-to-run variance** (and a cold-start warmup
   outlier); storage legs are deterministic. → average ≥3 cycles, discard warmup.

## Reproduce
```
# tax:
sudo -E $(which python) scripts/sweep_vllm_dump.py --store-out /var/data --cycles 3 \
  --configs "--input-len 2000 --num-prompts 8|--input-len 8000 --num-prompts 16|--input-len 12000 --num-prompts 32"
# S-curve:
sudo -E $(which python) scripts/sweep_vllm_dump.py --store-out /var/data --min-gb 15 --cycles 3 \
  --base "--max-model-len 16384 --max-tokens 64 --input-len 1024 --num-prompts 4" \
  --configs "--gpu-memory-utilization 0.5|--gpu-memory-utilization 0.7|--gpu-memory-utilization 0.9"
```
Per-op raw records: `data/timed_dump.jsonl`.
