# Temporal scheduling overhead: mechanism cost vs workload running energy

Demystifies the suspend/resume overhead by expressing the measured mechanism cost (dump+restore)
against each workload's own **dump-free running energy**. Both measured the same way (GPU board via
NVML + CPU pkg via RAPL + modeled DRAM), so the ratio is apples-to-apples.

## Method
- **Running energy**: `scripts/job_energy.py` runs a FIXED job (handshake excludes setup/model-load/
  table-build), integrates full power over the job-only window. Multiple job sizes per workload ->
  linear fit -> steady power. Records in `data/job_energy.jsonl`.
- **Dropped points** (measurement artifacts, all have clean replacements): A4 s50-s400 (<1.5 s window,
  launch-overhead-bound), A5c i10 (333 s > RAPL wrap -> 55 W), A8 q500 (536 s > wrap -> 54 W). The
  EPYC RAPL pkg counter wraps ~every 224 s at full load; under 64-thread saturation the sampler
  starves and the single-wrap unwrap misses wraps -> CPU under-count. Keep job windows < ~180 s.
- **Power is the per-window time-average** (∫P dt / window); within-window ripple is captured.
  Extrapolation to long runtimes assumes the same utilization/phase mix -- sound for the
  continuously-busy loops here; A2 serving is SATURATED offline-batch (upper bound; real online
  serving has duty cycle < 1).

## Steady running power, per leg
| WL | workload | GPU W | CPU W | DRAM W | total |
|---|---|---|---|---|---|
| A6 | gem5 (1 thread) | — | 143 | 5 | **148 W** |
| A5 | graph PageRank (mem-bound) | — | 234 | 21 | **255 W** |
| A8 | DuckDB multi-thread | — | 272 | 30 | **302 W** |
| A7 | DuckDB multi-process | — | 281 | 30 | **311 W** |
| A4 | DLRM (1 GPU, embed-bound) | 169 | 143 | 0 | **312 W** |
| A2 | vLLM serving (1 GPU, saturated) | 355 | 142 | 0 | **497 W** |
| A3 | ViT training (1 GPU) | 389 | 137 | 0 | **526 W** |
| A1 | FSDP training (4 GPU) | 1311 | 159 | 0 | **1471 W** |

~10x spread -> a single global "running power" would be wrong; per-workload baseline matters.
GPU-workload DRAM = 0 (run with --dram-gb 0; HBM is in the GPU-board leg). Workload character shows
in the power: memory-bound code (A5 graph, A4 DLRM) draws less than dense compute (A3 ViT, A1 FSDP).

## Overhead: mechanism cost / running power
"=N s run" = equivalent seconds of running; "%1h" = mechanism energy / (running power x 3600 s).

| WL | run W | NVMe mech | =N s | %1h | SATA mech | =N s | %1h |
|---|---|---|---|---|---|---|---|
| A1 (148 GB) | 1471 | 36.0 kJ | 24 s | 0.7% | 264.7 kJ | 180 s | 5.0% |
| A2 (41 GB) | 497 | 5.30 kJ | 11 s | 0.30% | 35.68 kJ | 72 s | 2.0% |
| A3 (35 GB) | 526 | 4.79 kJ | 9 s | 0.25% | 38.8 kJ | 74 s | 2.0% |
| A4 (33 GB) | 312 | 4.70 kJ | 15 s | 0.4% | 36.4 kJ | 117 s | 3.2% |
| A5 (71 GB) | 255 | 11.2 kJ | 44 s | 1.2% | 59.8 kJ | 235 s | 6.5% |
| A6 (86 GB) | 148 | 14.7 kJ | 99 s | 2.8% | 63.4 kJ | 428 s | 11.9% |
| A7 (100 GB) | 311 | 22.6 kJ | 73 s | 2.0% | 121 kJ | 389 s | 10.8% |
| A8 (100 GB) | 302 | 16.4 kJ | 54 s | 1.5% | 96.7 kJ | 320 s | 8.9% |

A2 firmed up from the 10-cycle TP=1 records (full GPU+CPU+DRAM+drive, robust floor): NVMe 20.2 s /
5.30 kJ, SATA 173.1 s / 35.68 kJ.

## Result
One suspend/resume = **0.2-2.8% of a 1-hour job's energy on NVMe, 1.3-12% on SATA** (= 6-100 s of
running on NVMe, 48-428 s on SATA). Temporal suspend is cheap in energy; the **storage tier** moves
it from negligible to noticeable. The **lowest-power workloads pay the highest percentage** (small
denominator): gem5 (148 W) at 2.8%/11.9% vs FSDP (1471 W) at 0.7%/5.0%. Scale: 30 min -> 2x these %,
2 h -> 0.5x. Using saturated active power makes these % a LOWER bound (real idle -> larger fraction).

## Per-suspend cost card (energy + LATENCY, both tiers) -- inputs for the carbon model
Round-trip dump+restore, robust floor. Both E_mech and T_mech are saved per leg in the timed_dump
records (t_start_mono/t_end_mono + full_total_j).
| WL | footprint | NVMe E / T | SATA E / T | run power |
|---|---|---|---|---|
| A1 | 148 GB | 36.0 kJ / 71.7 s | 264.7 kJ / 630.6 s | 1471 W |
| A2 | 41 GB | 5.30 kJ / 20.2 s | 35.7 kJ / 173.0 s | 497 W |
| A3 | 35 GB | 4.79 kJ / 18.5 s | 38.8 kJ / 157.3 s | 526 W |
| A4 | 33 GB | 4.70 kJ / 18.0 s | 36.3 kJ / 146.9 s | 312 W |
| A5 | 71 GB | 12.4 kJ / 55.5 s | 59.8 kJ / 311.7 s | 255 W |
| A6 | 50 GB | 14.7 kJ / 66.7 s | 63.4 kJ / 383.6 s | 148 W |
| A7 | 100 GB | 22.6 kJ / 88.4 s | 74.4 kJ / 573.1 s | 311 W |
| A8 | 100 GB | 16.4 kJ / 72.0 s | 96.7 kJ / 495.6 s | 302 W |

## Multi-suspend carbon model (the K dependence)
A job interrupted K times pays K independent (E_mech, T_mech) pairs:
- carbon overhead   = K · E_mech · CI            (gCO2 from the dump/restore work)
- added latency     = K · T_mech                 (completion-time stretch)
- carbon saved      = E_shifted · (CI_dirty - CI_clean)
- NET = saved - overhead;  break-even is in K.
More suspends -> finer carbon tracking (more saved) but more mechanism overhead. On SATA the
per-suspend cost is ~8x NVMe, so the optimal K -- and whether suspending pays at all -- flips with
the storage tier. (C2 analysis quantifies this against real CI traces.)

## Idle floor (resident-but-not-computing) and duty cycle
Measured with `steady_power.py` / `job_energy.py --no-handshake`:
- **Bare GPU idle** (one A100 holding 20 GB, no compute): **73 W board** + 125 W CPU node baseline.
  A parked GPU is NOT free -- ~19% of its active power. CPU node idle ~125 W (static floor).
- **vLLM engine idle** (model + KV pool resident, scheduler threads, zero requests): **209 W**
  (GPU 78 + CPU 131). Tag `a2_idle`.

**A2 serving is bursty**, so its running power is a range, not the saturated 497 W:
`E_serve(util) = 497·util + 209·(1-util) = 209 + 288·util`.
| util | A2 power | NVMe %1h (5.30 kJ) | SATA %1h (35.68 kJ) |
|---|---|---|---|
| 1.0 saturated | 497 W | 0.30% | 2.0% |
| 0.5 | 353 W | 0.42% | 2.8% |
| 0.3 typical online | 295 W | 0.50% | 3.4% |

The idle floor also frames the SCHEDULING decision: suspending frees the resource (can power down /
repurpose, saving the idle floor), whereas leaving a workload parked still burns it (73 W/GPU, 209 W
for a resident vLLM engine). The other 7 workloads run continuously busy (no idle phase), so their
active power = running average; only serving needs the duty-cycle treatment.
