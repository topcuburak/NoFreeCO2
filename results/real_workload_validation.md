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
## A5 — graph analytics (GAPBS PageRank), memory-extreme CPU, criu
`gapbs/pr` (Kronecker graph, OpenMP across all 64 threads / 4 NUMA nodes), checkpointed with criu
via `scripts/sweep_criu_dump.py --launch`. The CPU/host-domain counterpart to the GPU mechanisms,
and the paper's memory-extreme + real-CPU validation point (backs the synthetic S2). Footprint dialed
by graph scale: RSS ≈ 8 bytes × (2^scale × degree). Two footprints; energy = measured CPU pkg (RAPL)
+ modeled DRAM (0.3 W/GB) + modeled drive (NVMe 50 / SATA 3 W). 5 cycles, robust floor (drop cold).

**Tier comparison at 71 GB** (scale 29; SATA-safe — image < 140 GB free on `/`):
| leg | NVMe | SATA | flip |
|---|---|---|---|
| dump | 36.9 s / 8.18 kJ (1.92 GB/s) | 157.6 s / 35.2 kJ (0.45 GB/s) | 4.3× |
| restore | 13.4 s / 2.98 kJ (5.29 GB/s) | 154.1 s / 24.6 kJ (0.46 GB/s) | 11.5× |
| **round-trip** | **50.3 s / 11.16 kJ** (157 J/GB) | **311.7 s / 59.8 kJ** (843 J/GB) | **6.2× / 5.4×** |

**Memory-extreme, NVMe only at 279 GB** (scale 30, degree 32): dump 167.4 s / 46.2 kJ (1.67 GB/s),
restore 85.9 s / 23.7 kJ (3.25 GB/s); **round-trip 253 s / 69.9 kJ** (250 J/GB). A quarter-terabyte
64-thread process checkpointed and restored 5×.

**The criu regime, measured (the cross-cutting contrast with A1–A4):**
- **NVMe = overhead-bound:** dump 1.92, restore 5.29 GB/s — both well under raw NVMe (5.8 / 12.7).
  The criu page-walk + thread-freeze caps throughput, not the disk.
- **SATA = bandwidth-bound:** 0.45 / 0.46 GB/s = raw SATA, and dump ≈ restore (symmetric — the drive
  sets the time).
- **Tier flip is MUTED: 5.4× energy / 6.2× latency** vs the GPU mechanisms' 8–11×. Because criu cannot
  exploit NVMe's full bandwidth, the NVMe leg is "slow," shrinking the gap to bandwidth-bound SATA.
  The dump leg shows it cleanest (4.3×). This validates S2's synthetic muted-flip finding with a real
  64-thread graph workload, at 9× the footprint S2 reached.
- **Energy split inverts vs GPU:** CPU pkg dominates (idle/compute hold), DRAM grows with footprint×time
  (9% at 71 GB → 28% at 279 GB), drive is small. The GPU mechanisms are GPU-board-dominated; criu is
  CPU-hold-dominated. Same "slow-tier cost = the processor waiting, not the disk" pattern as A1.

Tags `a5_graph_nvme` (2×5 cyc), `a5_graph_sata2`, `a5_graph_big_nvme`.

## A8 — DuckDB in-memory analytics (multi-THREAD, latency-insensitive batch), criu
`work_duck.py` (one process, 64 threads, uncompressed in-memory table, looping GROUP BY).
The single-address-space / many-TID criu case. Footprint dialed by table rows (force_compression
off -> RSS tracks the target). 25->150 GB sweep, both tiers, 4 cycles, robust floor.

| footprint | NVMe dump | NVMe restore | SATA dump | SATA restore |
|---|---|---|---|---|
| 25 GB | 27.1 s / 5.51 kJ | 10.9 s / 2.20 kJ | 80.2 s / 17.7 kJ | 74.0 s / 10.8 kJ |
| 100 GB | 51.3 s / 11.68 kJ | 20.4 s / 4.71 kJ | 247 s / 55.2 kJ | 249 s / 41.5 kJ |
| 150 GB | 81.3 s / 19.72 kJ | 41.7 s / 10.17 kJ | (>140 GB free, NVMe only) | |

- **NVMe overhead-bound** (1.4-2.0 GB/s), **SATA bandwidth-bound** (0.40-0.45 GB/s = raw SATA).
- **Footprint-linear:** dump ≈ 1.3 kJ + 122 J/GB across 25->150 GB; restore ~half.
- **Tier flip at 100 GB: 6.9× latency / 5.9× energy** -- matches A5 (6.2×/5.4×), the muted criu flip.
- A real latency-insensitive batch-analytics workload lands on the same criu curve as the synthetic
  S2 and the GAPBS A5: the multi-thread mechanism cost is footprint-driven, tier-modulated, CPU-hold
  dominated. Tags `a8_duck_nvme` (6 pts), `a8_duck_sata` (4 pts).

## A7 — DuckDB analytics (multi-PROCESS, latency-insensitive batch), criu
`duck_mp.py` (16 independent DuckDB instances, separate address spaces, each ~gb/16 uncompressed
table + batch GROUP BY). The process-tree counterpart to A8's thread pool: SAME engine, footprints,
tiers -- the only variable is structure. 25->150 GB sweep, both tiers, 4 cycles, robust floor.

**A7 (multi-process) vs A8 (multi-thread), NVMe:**
| footprint | A7 dump | A8 dump | A7 restore | A8 restore |
|---|---|---|---|---|
| 100 GB | 75.4 s / 19.3 kJ | 51.3 s / 11.7 kJ | 13.0 s / 3.3 kJ | 20.4 s / 4.7 kJ |
| 150 GB | 106.5 s / 28.1 kJ | 81.3 s / 19.7 kJ | 27.4 s / 7.1 kJ | 41.7 s / 10.2 kJ |
| round-trip 150 GB | **134 s / 35.2 kJ** | **123 s / 29.9 kJ** | | |

**Structural asymmetry (the MP-vs-MT result):**
- Multi-process **dump is ~30% slower**: criu seizes/serializes 16 separate address spaces + writes
  16 image sets -> more per-process overhead than one process with 64 threads.
- Multi-process **restore is ~35% faster**: 16 independent processes fault their pages back in
  parallel, vs one process restoring more serially.
- Net round-trip: MP slightly costlier (dominated by the slower dump).
- **SATA: the structure effect washes out** -- A7 ≈ A8 (e.g. 75 GB dump 239 s vs 195 s, both ~0.4 GB/s,
  dump≈restore). The disk sets the time. So **process structure affects mechanism cost only in the
  overhead-bound (fast-storage) regime; on slow storage the tier dominates and MP-vs-MT disappears.**
- Both footprint-linear, muted ~6× criu tier flip. Tags `a7_duck_nvme` (6 pts), `a7_duck_sata` (4 pts).
  (SATA A7 100 GB dump energy is a RAPL artifact -- 29.5 kJ vs the ~75 kJ trend; latency is clean.)

**A7 was originally NPB-MPI (HPC batch) -- a reported boundary result:** criu DUMPS a live MPI job
(measured: 1.79 GB / 1.61 s / 315 J on ford, after disabling launcher GPU/NVML VMAs and dumping with
--tcp-established) but CANNOT transparently RESTORE it: the mpirun OOB listener port cannot be re-bound
(TIME_WAIT), and --tcp-close does not help (it is a listening, not established, socket). This marks the
boundary of transparent C/R -- restore must RE-ACQUIRE external/scarce resources (ports, PIDs, locks)
that dump only READS -- alongside vLLM-TP and HACC. A7 thus pivoted to DuckDB-multi-process (criu-safe).

## A6 — gem5 (big-RSS single process), CPU/criu — PENDING
