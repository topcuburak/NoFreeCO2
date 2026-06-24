# Complete statistics inventory (SoCC 2026 mechanism-cost paper)

Testbed **ford**: 4x A100-40GB SXM4, EPYC 75F3 (32c/64t, 4 NUMA), 755 GB RAM, PM1733 RAID-0 NVMe
(/var/data, 3.9 TB free) + HPE SATA SSD (/, 140 GB free). Energy = MEASURED GPU board (NVML) + CPU
pkg (RAPL) + MODELED DRAM (0.3 W/GB) + drive (NVMe 50 W / SATA 3 W).

## 1. Temporal mechanism cost -- per-suspend dump+restore (round-trip), both tiers
| WL | workload | footprint | NVMe E / T | SATA E / T |
|---|---|---|---|---|
| A1 | Llama-3.1-8B FSDP train (4 GPU) | 148 GB | 36.0 kJ / 71.7 s | 264.7 kJ / 630.6 s |
| A2 | vLLM serving TP=1 | 41 GB | 5.30 kJ / 20.2 s | 35.7 kJ / 173.0 s |
| A3 | ViT-Huge train (1 GPU) | 35 GB | 4.79 kJ / 18.5 s | 38.8 kJ / 157.3 s |
| A4 | DLRM train (1 GPU) | 33 GB | 4.70 kJ / 18.0 s | 36.3 kJ / 146.9 s |
| A5 | GAPBS PageRank (criu) | 71 GB | 12.4 kJ / 55.5 s | 59.8 kJ / 311.7 s |
| A5b| GAPBS memory-extreme | 279 GB | 69.9 kJ / 253.3 s | (>140 GB, NVMe only) |
| A6 | gem5 sim (criu) | 50 GB | 14.7 kJ / 66.7 s | 63.4 kJ / 383.6 s |
| A7 | DuckDB multi-process (criu) | 100 GB | 22.6 kJ / 88.4 s | 74.4 kJ / 573.1 s |
| A8 | DuckDB multi-thread (criu) | 100 GB | 16.4 kJ / 72.0 s | 96.7 kJ / 495.6 s |

Per-leg rates: GPU transparent **suspend ~5.0-5.5 GB/s** (tier-independent), restore ~13 GB/s.
NVMe store 5.9 / load 12 GB/s; SATA store 0.46 / load 0.50 GB/s. criu NVMe **1.4-2.0 GB/s
(overhead-bound, < raw 5.8)**, SATA **0.40-0.46 GB/s (bandwidth-bound)**.

## 2. Regime findings (cross-cutting)
- **GPU mechanisms bandwidth-bound -> NVMe->SATA flip 8-11x** (A1 8.8, A2 8.6, A3 8.5, A4 8.1).
- **criu overhead-bound on NVMe / bandwidth-bound on SATA -> muted flip ~5-6x** (A5/A6/A7/A8).
- **Reserved-vs-live footprint**: cuda-checkpoint dumps caching-allocator RESERVED high-water.
  A3 ViT 7.72 GB live -> 35.2 GB dumped (27 GB activation slack); A4 DLRM ~live (embedding-bound).
- **MP vs MT (A7 vs A8, same DuckDB)**: multi-process dump ~30% slower, restore ~35% faster than
  multi-thread; net RT MP slightly costlier; structure effect washes out on SATA (disk-bound).
- **Transparent RESTORE boundary**: MPI/TP runtime sockets block it (NPB-MPI dumps 1.79 GB/1.61 s
  but can't re-bind the mpirun OOB listener; vLLM-TP _PP assertion; HACC). Dump reads, restore must
  re-acquire scarce external resources (ports/PIDs/locks).
- gem5 host RSS = guest + SE bloat; checkpointable with --remote-gdb-port=0.

## 3. Dump-free running power (steady-state, per leg)
| WL | GPU W | CPU W | DRAM W | total | character |
|---|---|---|---|---|---|
| A1 FSDP (4 GPU) | 1311 | 159 | 0 | **1471 W** | ~328 W/GPU |
| A3 ViT (1 GPU) | 389 | 137 | 0 | **526 W** | compute-bound |
| A2 vLLM (1 GPU) | 355 | 142 | 0 | **497 W** | saturated batch |
| A4 DLRM (1 GPU) | 169 | 143 | 0 | **312 W** | embedding/launch-bound |
| A7 DuckDB-MP | 0 | 281 | 30 | **311 W** | 64-thread |
| A8 DuckDB-MT | 0 | 272 | 30 | **302 W** | 64-thread |
| A5 graph | 0 | 234 | 21 | **255 W** | memory-bound (cores stall) |
| A6 gem5 | 0 | 143 | 5 | **148 W** | single-thread |
~10x spread. Linear in job size (per-step/query/iter). RAPL pkg counter wraps ~224 s at full load
(keep windows < 180 s); GPU NVML unaffected.

## 4. Idle floor / duty cycle
- Bare A100 idle (holding mem): **73 W** board (~19% of active). CPU node idle **125 W**.
- vLLM engine idle (resident, no requests): **209 W** (GPU 78 + CPU 131).
- A2 serving range: E = 209 + 288*util W (saturated 497 W -> 295 W at util 0.3).

## 5. Overhead ratios (mechanism / running energy)
One suspend/resume = N s of running, and % of a job's energy:
| WL | run W | NVMe = Ns / %1h | SATA = Ns / %1h |
|---|---|---|---|
| A1 | 1471 | 24 s / 0.7% | 180 s / 5.0% |
| A2 | 497 | 11 s / 0.30% | 72 s / 2.0% |
| A3 | 526 | 9 s / 0.25% | 74 s / 2.0% |
| A4 | 312 | 15 s / 0.4% | 117 s / 3.2% |
| A5 | 255 | 44 s / 1.2% | 235 s / 6.5% |
| A6 | 148 | 99 s / 2.8% | 428 s / 11.9% |
| A7 | 311 | 73 s / 2.0% | 389 s / 10.8% |
| A8 | 302 | 54 s / 1.5% | 320 s / 8.9% |
NVMe 0.2-2.8% of a 1 h job; SATA 1.3-12%. Lowest-power workloads pay highest %.

## 6. Carbon temporal scheduling (~50 grids, Electricity Maps 2023, H=24)
Deadline-budget shifting net of suspend/restore. C (compute h, research-grounded):
A1=4 A2=2 A3=12 A4=1 A5=1 A6=8 A7=1 A8=1.
| WL | C | gross savings | NVMe net | SATA net | K | SATA ovh (of savings) |
|---|---|---|---|---|---|---|
| A4/A5/A7/A8 | 1 h | 21.3% | 21.3% | 21.3% | 0.00 | 0% (fit one hour) |
| A2 | 2 h | 20.4% | 20.4% | 20.2% | 0.27 | 1.1% |
| A1 | 4 h | 18.9% | 18.8% | 18.3% | 0.56 | 3.1% |
| A6 | 8 h | 15.8% | 15.5% | 14.5% | 0.96 | **8.0%** |
| A3 | 12 h | 11.9% | 11.8% | 11.6% | 1.34 | 1.9% |
- Savings ~12-25% at H=24 (up to ~33% volatile grids); grow with horizon, shrink with C.
- Short jobs (C=1 h): K=0, zero overhead. Overhead is a long-job phenomenon, worst for A6 gem5.
- 10-17% of long-job instances net-negative at H=24 (more at H<=8): suspend not worth it in the
  low-slack / SATA / low-power / flat-grid corner.

## 7. Workload durations + resource counts (see workload_durations_refs.md)
A1 8B FT ~1-4 h/8 GPU; A2 batch ~1.5 h/1 GPU; A3 ViT-H days (2500 TPUv3-core-days); A4 DLRM ~15 min/
8 GPU; A5 PR ~3 min/64c; A6 gem5 hours-days/1c; A7/A8 DuckDB suite ~3 min/64c. (Sourced, MLPerf/papers.)

## 8. Spatial (NOT yet computed -- coefficients only)
Migrate leg = image_GB x per-byte + RTT; energy 3.6 kJ/GB fabric (band 0.3-6) + endpoint hold.
WAN single-stream 8-25 s/GB (RTT-scaled), parallel ~0.5-1 s/GB. (network_coefficients_lit.md.)
3-leg spatial = dump (measured) + migrate (modeled) + restore (measured) -- PENDING analysis.
