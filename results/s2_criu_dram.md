# S2 — criu dump/restore vs DRAM footprint (CPU domain), measured energy, both tiers

**Testbed:** ford (EPYC 75F3, NVMe-RAID0 /var/data, SATA SSD /home). **Date:** 2026-06-18.
**Tool:** `scripts/sweep_criu_dump.py` + `scripts/work_dram.py` (ACTIVE configurable-DRAM workload:
full-array increment every iteration, so memory is resident/dirty/hot — criu checkpoints a LIVE
compute process). Footprints 4/8/16/32 GiB, 4 cycles, both tiers, robust floor (drop cold cycle-0).
**criu confirmed WORKING on ford** (first time — it was only ever io_uring-blocked inside vLLM; a
plain anon-memory process dumps fine). Energy = MEASURED CPU pkg (RAPL ∫P dt) + MODELED DRAM
(0.3 W/GB) + drive (NVMe 50 / SATA 3 W). **No GPU** (CPU workload). Dump `sync`s the image
(durable); restore reads cold (drop_caches) — both genuinely hit the tier.

## Per-footprint cost
| GB | NVMe dump (GB/s, kJ) | NVMe restore | SATA dump (GB/s, kJ) | SATA restore |
|---|---|---|---|---|
| 4 | 2.57 s (1.56, 0.51) | 1.35 s (2.96, 0.26) | 12.20 s (0.33, 1.86) | 9.87 s (0.41, 1.38) |
| 8 | 4.99 s (1.60, 0.99) | 3.45 s (2.32, 0.68) | 24.22 s (0.33, 3.76) | 18.74 s (0.43, 2.65) |
| 16 | 9.83 s (1.63, 1.99) | 3.33 s (4.80, 0.68) | 45.20 s (0.35, 7.14) | 37.64 s (0.43, 5.41) |
| 32 | 19.36 s (1.65, 4.04) | 9.57 s (3.34, 1.97) | 77.52 s (0.41, 12.55) | 74.24 s (0.43, 11.09) |

**Affine fits (cost = a + b·S):**
- NVMe: dump **0.20 s + 0.599 s/GB (1.67 GB/s)**, restore **0.28 s + 0.276 s/GB (3.62 GB/s)**
- SATA: dump **5.24 s + 2.303 s/GB (0.43 GB/s)**, restore **0.56 s + 2.304 s/GB (0.43 GB/s)**

Round-trip @32 GB: NVMe ~29 s / 6.0 kJ (188 J/GB), SATA ~152 s / 23.6 kJ (739 J/GB).

## Findings
1. **On NVMe, criu is OVERHEAD-bound, not disk-bound.** dump 1.65, restore 3.6 GB/s — both far below
   raw NVMe (5.8 write / 12.7 read), with the CPU near-idle (e.g. 32 GB restore 145 W vs 128 baseline).
   criu's per-page image serialization (dump) and process recreation + paged image read (restore) cap
   the rate, not the device.
2. **On SATA, criu is DISK-bound** — both legs ~0.43 GB/s = the SATA ceiling.
3. **Tier flip is MUTED: dump 4.0×, restore 7.8×** (NVMe→SATA at 32 GB) vs the GPU mechanisms' 8–11×.
   Because criu's overhead caps the fast tier, the NVMe↔SATA gap shrinks — **criu's per-page cost
   compresses storage-tier sensitivity**. A clean regime contrast: the GPU dump is bandwidth-bound (so
   the tier dominates), criu is overhead-bound on fast storage (so the tier matters less).
4. **dump > restore cost** (1.65 < 3.6 GB/s on NVMe) — the dump pays for walking + serializing every
   page and the `sync`; restore's cold image read is comparatively cheaper per byte.
5. **SATA energy is CPU idle-hold** — 32 GB SATA dump 12.55 kJ is mostly the CPU sitting at ~idle for
   77 s of disk I/O wait + modeled DRAM, drive negligible (3 W). Same idle-hold regime as the GPU SATA.

## CPU-domain vs GPU-domain (the regime contrast)
- **GPU (A1/A2/S1):** suspend HBM→host (PCIe, bandwidth-bound, parallel) + store host→disk. Two legs;
  the GPU↔host leg is fast (parallel PCIe), storage dominates; tier flip 8–11×.
- **CPU (S2 criu):** the state is already in DRAM, so criu does DRAM→disk in ONE leg — but
  overhead-bound (per-page), ~1.6–3.6 GB/s on NVMe regardless of the 12.7 GB/s device. No HBM/PCIe.
  Tier flip muted (4–8×).
→ Mechanism cost is still footprint-linear in both domains, but the *bottleneck differs*: bandwidth
(GPU) vs per-page checkpoint overhead (CPU/criu).

## Remaining
A6 (gem5 SPEC CPU2017, multi-process ~10–100 MB each) validates this on a real CPU workload + the
multi-process regime (per-process criu fixed overhead dominates at small RSS). S2 gives the per-byte
criu coefficient; A6 the small-footprint / many-process behavior.

## Reproduce
```
sudo -E python scripts/sweep_criu_dump.py --sizes 4,8,16,32 --cycles 4 \
  --store-out /var/data --tag s2_criu_nvme_v3        # SATA: --store-out /home/test --tag s2_criu_sata_v3
# --target work_dram.py (active, default) | hold_dram.py (idle)
```
Raw: `data/timed_dump.jsonl` tags `s2_criu_nvme_v3` / `s2_criu_sata_v3` (CPU domain, no GPU).
