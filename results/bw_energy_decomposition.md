# PCIe bandwidth sweep — energy vs bandwidth (standalone experiment)

**Testbed:** ford (1× A100-40GB, PCIe Gen4, EPYC 75F3). **Date:** 2026-06-08.
**Tool:** `scripts/characterize_bw_sweep.py` (`microbench.pcie_copy_rate`, chunk+pace to
a target GB/s). **S = 16 GB/trial**, 3 trials + 1 warmup per rate. **Energy = `∫P dt`**
(`energy_abs_j`): `nvml_gpu_pkg` (GPU) + `cpu_pkg_energy_rapl` (CPU pkg), NVML scoped to
the one GPU. DRAM modeled.

Scope note: this file is the raw bandwidth-sweep experiment and its fit only. It does not
cross-reference or reinterpret other results.

## Method
Move a fixed `S` over PCIe at a CONTROLLED effective bandwidth (transfer in chunks,
sleep to pace to a target rate). Sweeping BW sweeps the transfer time `t = S/BW`. Fit
**E vs t**: slope and intercept.

## Measured (mean, n=3)

| dir | BW (GB/s) | lat (s) | gpu_abs (J) | cpu_abs (J) | total (J) |
|---|---|---|---|---|---|
| d2h | 2.0 | 7.99 | 583 | 1026 | 1608 |
| d2h | 4.0 | 3.99 | 292 | 521 | 813 |
| d2h | 6.0 | 2.66 | 194 | 353 | 548 |
| d2h | 8.0 | 2.00 | 145 | 269 | 415 |
| d2h | 12.0 | 1.33 | 123 | 184 | 307 |
| d2h | 16.0 | 1.00 | 88 | 141 | 229 |
| d2h | 24.9 (ceiling) | 0.64 | 56 | 95 | 151 |
| h2d | 2.0 | 7.99 | 588 | 1030 | 1618 |
| h2d | 4.0 | 3.99 | 293 | 523 | 817 |
| h2d | 6.0 | 2.66 | 193 | 350 | 543 |
| h2d | 8.0 | 2.00 | 145 | 266 | 411 |
| h2d | 12.0 | 1.33 | 124 | 182 | 306 |
| h2d | 16.0 | 1.00 | 90 | 140 | 230 |
| h2d | 24.5 (ceiling) | 0.65 | 54 | 95 | 149 |

`d2h` = HBM→host (extract direction); `h2d` = host→HBM. Raw records in `data/bw_sweep.jsonl`.

## Fit: `E = intercept + slope·t`  (linear least squares, R² ≈ 1)

| dir | domain | slope (W) | intercept (J) | intercept/S (J/GB) |
|---|---|---|---|---|
| d2h | gpu | 70.8 | 13 | 0.8 |
| d2h | cpu | 126.6 | 15 | 0.9 |
| d2h | **total** | **197.4** | **29** | **1.8** |
| h2d | gpu | 71.5 | 12 | 0.8 |
| h2d | cpu | 127.6 | 12 | 0.7 |
| h2d | **total** | **199.1** | **24** | **1.5** |

Predicted vs measured (d2h total): t=0.64→155/151, t=2.00→423/415, t=7.99→1603/1608.

## Observations (intrinsic to this experiment)

- Energy is **linear in transfer time** over 2–25 GB/s (R²≈1): a small bandwidth-
  independent intercept (~24–29 J for 16 GB) plus a slope of ~198 W.
- The slope splits as **CPU ~127 W + GPU ~71 W**; the intercept (per-byte term) is
  small in both domains (~0.7–0.9 J/GB each).
- Share of total energy in the slope term: **~83% at 25 GB/s, ~98% at 2 GB/s.**
- `d2h` and `h2d` fits agree to within ~1% — the raw transfer is **direction-symmetric**
  in both bandwidth and energy.

## Reproduce
```
sudo -E $(which python) scripts/characterize_bw_sweep.py --bytes 16e9 --gpu 0 \
  --rates 2,4,6,8,12,16,0 --dir both --repeat 3 --warmup 1
```
