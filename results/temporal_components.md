# Temporal mechanism cost — per-component, vs footprint (measured, hold_gpu)

**Testbed:** ford (A100-40GB, EPYC 75F3, NVMe-RAID0 /var/data, SATA SSD /home). **Date:** 2026-06-15.
**Tool:** `scripts/sweep_checkpoint_size.py` (hold_gpu target → suspend→store→load→resume), single-GPU
(TP=1), 3 cycles/footprint, footprints 2.7–39.2 GB. Energy = `∫P dt` (GPU NVML + CPU RAPL).
criu skipped (io_uring); store/load = O_DIRECT dd device-rate proxy.

Temporal mechanism = `suspend → store → [hold] → load → resume`. Two domains: HBM↔host
(tier-independent) and host↔storage (NVMe vs SATA).

## Per-component cost (cost = a + b·S), host+GPU energy

| component | tier | bandwidth | b_lat (s/GB) | b_E (J/GB) | a_E (J) |
|---|---|---|---|---|---|
| HBM→host (suspend) | — | 5.9 GB/s | 0.170 | 41.3 | 155 |
| host→HBM (resume) | — | 20 GB/s | 0.050 | 13.1 | 177 |
| host→NVMe (store) | NVMe | 5.9 GB/s | 0.170 | 33.3 | 61 |
| NVMe→host (load) | NVMe | 12.8 GB/s | 0.078 | 15.3 | 31 |
| host→SATA (store) | SATA | 0.48 GB/s | 2.064 | 383.6 | 791 |
| SATA→host (load) | SATA | 0.53 GB/s | 1.881 | 351.4 | 752 |

All bandwidths match the standalone leg sweeps (NVMe 5.8/12.7, SATA 0.48/0.53). The 34 GB resume
transient is excluded from the resume fit.

## Composed temporal round-trip (suspend + store + load + resume)
```
Temporal-NVMe:  latency = 2.17 + 0.468·S  s   |   energy = 423  + 103·S  J
Temporal-SATA:  latency = 9.98 + 4.166·S  s   |   energy = 1875 + 789·S  J
```

## Power decomposition per component (at ~40 GB, NVMe)
| component | GPU (implied W) | CPU (implied W) |
|---|---|---|
| HBM→host (suspend) | 678 J (**91 W active DMA**) | 1108 J (148 W) |
| host→HBM (resume) | 247 J (**87 W active**) | 429 J (151 W) |
| host→NVMe (store) | 376 J (**55 W idle** — freed GPU) | 955 J (140 W floor) |
| NVMe→host (load) | 177 J (**55 W idle**) | 455 J (140 W floor) |

- **host↔storage** = CPU floor (~140 W, disk I/O) + **GPU idle ~55 W** (freed-but-powered; charge
  only if GPU reserved — committed-vs-released choice).
- **host↔HBM** = GPU **active** (~88–91 W = 55 idle + ~35 DMA) + CPU floor (~148 W).

## Key structural finding (GPU-mechanism vs storage share)
| tier | HBM↔host (tier-indep) | storage | storage share |
|---|---|---|---|
| NVMe | 54.4 J/GB | 48.6 J/GB | **47%** — balanced |
| SATA | 54.4 J/GB | 735 J/GB | **93%** — storage dominates |

→ On fast storage the GPU mechanism is ~half the cost; on slow storage it's a rounding error
(storage ~14× the HBM↔host term). Full round-trip **~7.7× costlier on SATA** (789 vs 103 J/GB).

## Notes / data-quality
- The saved records (`data/timed_dump.jsonl`) lack a clean workload/tier/chunks label
  (`workload`="timed_dump" always; `mark_min` overloaded; `load` leg omits the tier). Tier was
  recovered by **bandwidth classification** and vLLM `mark_min` 0/1 rows excluded. Solid, but
  motivates adding explicit `workload`/`tier`/`chunks`/`footprint` tags to records.

## Reproduce
```
sudo -E $(which python) scripts/sweep_checkpoint_size.py --gpu 0 \
  --sizes 2,4,8,16,24,30,36 --store-out /var/data --cycles 3 --chunks 1     # NVMe
sudo -E $(which python) scripts/sweep_checkpoint_size.py --gpu 0 \
  --sizes 2,4,8,16,24,30,36 --store-out /home/test --cycles 3 --chunks 1    # SATA
```
