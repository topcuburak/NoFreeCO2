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

## A4 — DLRM training (memory-extreme, embedding-dominated, transparent cuda-checkpoint)
`workloads/a4_dlrm/dlrm_train.py` (sparse embedding 30 GiB + bottom/top MLP, batch 2048, plain SGD).
Single process, no NCCL. **Footprint 32.8 GB** = 32.2 GB resident embedding + ~0.6 GB CUDA context.
5 cycles, variance <1%, no drift (the cold cycle is not even an outlier here).

| leg | NVMe | SATA |
|---|---|---|
| suspend | 6.41 s (5.13 GB/s) | 6.39 s (tier-indep ✓) |
| store | 6.34 s (5.18 GB/s) | 72.0 s (0.46 GB/s) |
| load | 2.81 s (11.7 GB/s) | 66.0 s (0.50 GB/s) |
| resume | 2.56 s | 2.57 s (tier-indep ✓) |
| **round-trip** | **18.1 s / 4.70 kJ** (143 J/GB) | **147 s / 36.4 kJ** (1107 J/GB) |

**Validates the model:** S1 fit predicts suspend 0.79 + 0.196×32.8 = **7.2 s**, A4 measured **6.4 s**;
suspend 5.13 GB/s = S1 5.1 = A3 5.07; store/load match the tier sweep; NVMe→SATA flip **8.1×/7.7×**;
energy/byte 143 J/GB ≈ A3's 136 (workload-independent). Tags `a4_dlrm_nvme`/`a4_dlrm_sata`.

**The A4 contrast (why DLRM was worth running):** the dumped image (32.8 GB) ≈ the resident embedding
(32.2 GB) — the gap is just the CUDA context, with NO reserved-activation slack. This is the MIRROR of
A3, where 7.72 GB live ballooned to a 35.2 GB dumped image (27 GB of reserved activation peak). So the
dumped footprint spans ≈live (embedding-bound DLRM) to ≫live (activation-bound ViT), and the mechanism
cost tracks the DUMPED bytes either way — confirming the cost driver is what cuda-checkpoint actually
serializes (caching-allocator reserved high-water), not the framework's live-tensor accounting.

## A5 — HACC cosmology — PENDING
## A5 — HACC cosmology — PENDING
## A6 — gem5 SPEC CPU2017 (CPU/criu) — PENDING
