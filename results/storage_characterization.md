# Storage tier characterization (appendix data)

**Testbed:** ford. **Date:** 2026-06-02. **Tool:** `scripts/characterize_storage.py`
**Config:** `--bytes 16e9 --repeat 8 --warmup 2 --method direct` (O_DIRECT `dd`, device capability).
**Telemetry:** `nvml_gpu_pkg`, `cpu_pkg_energy_rapl`, `nvme_bytes_written`.
**Energy metric:** `cpu_abs` = CPU package energy incl. idle over the op = data-movement + time term (the real dump/restore cost). GPU ≈ 0 (storage doesn't touch GPU). DRAM modeled (no DRAM RAPL on EPYC).

**Tiers:**
- `nvme_raid0` = 2× Samsung PM1733 RAID-0 (md0, xfs) at `/var/data`
- `sas_ssd` = HPE VK000480GXNZA (sda2, ext4) at `/home`

## Summary (mean ± std, n=8, 2 warmup discarded)

| tier | op | GB/s | latency_s | cpu_abs_J |
|---|---|---|---|---|
| nvme_raid0 | write (dump)    | 5.80 ± 0.46 | 2.78 ± 0.23 | 371 ± 33 |
| nvme_raid0 | read (restore)  | 12.69 ± 0.22 | 1.26 ± 0.02 | 170 ± 5 |
| sas_ssd | write (dump)    | 0.48 ± 0.00 | 33.37 ± 0.20 | 4202 ± 33 |
| sas_ssd | read (restore)  | 0.53 ± 0.00 | 30.27 ± 0.11 | 3815 ± 27 |

**Cross-tier (dump/write):** SAS is ~12× the latency and ~11× the energy of NVMe-RAID0 for the same 16 GB.
**Dump/restore asymmetry:** read ~2× faster/cheaper than write (NVMe: 1.26 vs 2.78 s).

## Raw per-trial data

### nvme_raid0 (/var/data)
| trial | write_s | write_J | read_s | read_J |
|---|---|---|---|---|
| warmup | 2.58 | 346 | 1.25 | 165 |
| warmup | 2.70 | 358 | 1.27 | 167 |
| t1 | 2.59 | 344 | 1.26 | 168 |
| t2 | 2.75 | 362 | 1.26 | 168 |
| t3 | 2.57 | 341 | 1.26 | 167 |
| t4 | 2.62 | 352 | 1.27 | 168 |
| t5 | 2.60 | 345 | 1.26 | 167 |
| t6 | 3.22 | 431 | 1.29 | 182 |
| t7 | 2.92 | 393 | 1.27 | 174 |
| t8 | 2.95 | 397 | 1.22 | 168 |

### sas_ssd (/home/test)
| trial | write_s | write_J | read_s | read_J |
|---|---|---|---|---|
| warmup | 32.97 | 4188 | 30.22 | 3812 |
| warmup | 33.15 | 4172 | 30.38 | 3823 |
| t1 | 33.46 | 4205 | 30.21 | 3820 |
| t2 | 33.14 | 4171 | 30.35 | 3834 |
| t3 | 33.26 | 4186 | 30.24 | 3814 |
| t4 | 33.25 | 4202 | 30.20 | 3805 |
| t5 | 33.23 | 4170 | 30.46 | 3847 |
| t6 | 33.49 | 4243 | 30.31 | 3846 |
| t7 | 33.77 | 4259 | 30.29 | 3783 |
| t8 | 33.36 | 4180 | 30.10 | 3774 |

## Observations

- **SAS is extremely stable** (write CV <1%, ±0.00 GB/s) — device-bandwidth-bound, deterministic.
- **NVMe-RAID0 drifts up on later trials** (t6–t8: 2.92–3.22 s / 393–431 J vs t1–t5: ~2.6 s / ~350 J). Repeated O_DIRECT rewrites of the same file → likely **thermal / NVMe GC / write-amplification** at sustained load. This is the mild **super-linear-at-scale** behavior expected from the step breakdown; the ±0.46 GB/s std captures it. Worth flagging when arguing the linear cost model holds in the mid-range but degrades under sustained/large writes.
- Energy tracks latency tightly (energy = node power × time), confirming `cpu_abs` (time term) is the right, reproducible dump-cost metric vs the noisy marginal.

## Reproduce
```
git clone https://github.com/topcuburak/NoFreeCO2 && cd NoFreeCO2
conda env create -f environment.yml && conda activate socc-bench
sudo -E $(which python) scripts/characterize_storage.py --bytes 16e9 --repeat 8 --warmup 2
```
Raw per-op JSON records: `data/storage_char.jsonl`.
