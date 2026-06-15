# Storage size sweep — affine cost model per tier (standalone experiment)

**Testbed:** ford (EPYC 75F3). **Date:** 2026-06-08. **Tool:** `scripts/characterize_storage_sweep.py`.
**Sizes:** 2→50 GB in 2 GB steps (block-aligned to 64 MB), O_DIRECT `dd`, cold reads
(drop_caches). NVMe `repeat=2`, SAS `repeat=1`.
**Energy:** `cpu` = MEASURED CPU package (`∫P dt` via RAPL); `drive` = MODELED
(`n_drives · P_active · latency`); DRAM/rest-of-node not captured (no BMC/PDU on ford).
**Drive power (modeled):** NVMe = `nvme id-ctrl` PS0 `mp` **25 W × 2** (RAID-0); SATA =
**3 W** datasheet (`smartctl` exposes no live power). Rated upper bound, not measured.

Scope note: this is the raw size-sweep experiment and its fit only; it does not
cross-reference or reinterpret other results.

## Tiers
- `nvme_raid0` = 2× Samsung PM1733 RAID-0 (md0) at `/var/data`
- `sas_ssd` = HPE VK000480GXNZA SATA SSD at `/home` (sda2)

## Affine fit  `y = a + b·S`  (S in GB)

| tier | op | cpuE: a (J) | b (J/GB) | totE: a (J) | b (J/GB) | 1/BW (s/GB) | BW (GB/s) | cpu_W |
|---|---|---|---|---|---|---|---|---|
| nvme_raid0 | write | 13 | 23.6 | 18 | 31.9 | 0.1664 | 6.01 | 141 |
| nvme_raid0 | read | 3 | 10.9 | 4 | 14.8 | 0.0778 | 12.85 | 141 |
| sas_ssd | write | −53 | 273.1 | −54 | 279.3 | 2.0636 | 0.48 | 132 |
| sas_ssd | read | −11 | 249.1 | −11 | 254.7 | 1.8772 | 0.53 | 132 |

(25 points/fit. Representative rows below; full per-op data in `data/storage_sweep.jsonl`.)

| size | NVMe W (s / GB/s / cpuJ / totJ) | SAS W (s / GB/s / cpuJ / totJ) |
|---|---|---|
| 2 GB | 0.42 / 4.60 / 60 / 81 | 3.95 / 0.49 / 519 / 531 |
| 16 GB | 2.90 / 5.50 / 410 / 555 | 32.31 / 0.49 / 4253 / 4350 |
| 50 GB | 8.67 / 5.76 / 1231 / 1665 | 102.71 / 0.49 / 13639 / 13947 |

## Findings (intrinsic to this experiment)

1. **Proportional, negligible fixed overhead.** Intercepts are within noise of zero
   (NVMe tiny-positive, SAS tiny-negative — a fit artifact); <0.5% of the cost at 50 GB.
   So storage cost is **`b·S`**, no meaningful per-op fixed term over this range.

2. **`b_E = (P_cpu + P_drive) / BW`**, exactly, every tier/direction:
   - NVMe write 191 W / 6.01 = 31.8 ≈ 31.9 J/GB; SAS write 135 W / 0.48 = 281 ≈ 279 J/GB.
   The whole cost reduces to **power ÷ bandwidth**.

3. **`cpu_W` ≈ constant (132–141 W)** across tiers, sizes, and directions — the node
   CPU floor. The ~12–23× energy/byte gap between tiers is **entirely bandwidth** (time),
   at essentially equal power.

4. **Drive-share flips with tier:**
   - NVMe (fast): modeled drive = **+35%** of energy (50 W over a short transfer).
   - SAS (slow): modeled drive = **+2%** (3 W over a long transfer — negligible).

5. **Direction:** write costs ~2.1× read per byte on NVMe (BW 6.0 vs 12.85), ~1.1× on
   SAS (0.48 vs 0.53) — purely the bandwidth ratio at equal power.

## Model
```
E_store(S, tier, dir) ≈ (P_cpu + P_drive) · S / BW          (intercept ≈ 0)
  P_cpu ≈ 132–141 W (measured floor) ; P_drive modeled (NVMe 50 W / SATA 3 W)
  BW: NVMe 6.01 write / 12.85 read ; SAS 0.48 / 0.53 GB/s
```

## Reproduce
```
sudo -E $(which python) scripts/characterize_storage_sweep.py \
  --min-gb 2 --max-gb 50 --step-gb 2 --repeat 2 --tiers nvme_raid0=/var/data
nohup sudo -E $(which python) scripts/characterize_storage_sweep.py \
  --min-gb 2 --max-gb 50 --step-gb 2 --repeat 1 --tiers sas_ssd=/home/test &
```
