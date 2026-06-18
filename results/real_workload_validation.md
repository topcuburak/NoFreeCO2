# Real-workload validation (A3–A6): mechanism cost vs the footprint-driven model

The S1/S2 synthetic baselines + A1/A2 gave the mechanism-cost model (footprint-driven, `a + b·S`,
no allocation-structure term). A3–A6 are REAL workloads run end-to-end through the same suspend/restore
harnesses to confirm each lands on the model at its footprint. Methodology identical: transparent
cuda-checkpoint (GPU, single-process) or criu (CPU); 5 cycles, robust floor; FULL energy (measured
GPU+CPU + modeled DRAM/drive); both tiers.

## A3 — ViT-Huge training (single GPU, transparent cuda-checkpoint)
`workloads/a3_vit/vit_train.py` (torchvision vit_h_14, 632M params, batch 64, bf16). Real training loop,
suspended in place via cuda-checkpoint (no NCCL). **Footprint 35.2 GB** = the caching-allocator RESERVED
high-water (activation peak), NOT the 7.72 GB live-between-steps — cuda-checkpoint dumps reserved segments.

| leg | NVMe | SATA |
|---|---|---|
| suspend | 6.95 s (5.07 GB/s) | 6.96 s (tier-indep ✓) |
| store | 5.96 s (5.91 GB/s) | 77.4 s (0.46 GB/s) |
| load | 2.93 s (12.0 GB/s) | 70.3 s (0.50 GB/s) |
| resume | 2.71 s | 2.68 s (tier-indep ✓) |
| **round-trip** | **18.5 s / 4.79 kJ** (136 J/GB) | **157 s / 38.8 kJ** (1100 J/GB) |

**Validates the model:** S1 fit predicts suspend 0.79 + 0.196×35 = **7.7 s**, A3 measured **6.9 s** (≈10%);
suspend 5.07 GB/s = S1 5.1 = A1 5.47; store/load match the tier sweep; NVMe→SATA flip **8.5×/8.1×**
(consistent with A1 8.8, A2-TP1 8.6, S1 8.3). A real vision-transformer training workload sits on the
footprint curve — no workload-specific premium. Tags `a3_vit_nvme`/`a3_vit_sata`.

Note worth keeping: `gpu_alloc` (PyTorch live, 7.72 GB) ≠ dumped footprint (NVML reserved, 35.2 GB) —
the activation high-water is reserved-but-mostly-free between steps, yet cuda-checkpoint dumps all of it.
Same effect seen in A1 (training reserves the backward-activation peak).

## A4 — DLRMv2 training (memory-extreme, embedding-dominated) — PENDING
## A5 — HACC cosmology — PENDING
## A6 — gem5 SPEC CPU2017 (CPU/criu) — PENDING
