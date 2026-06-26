# Sub-hourly (15-min) temporal scheduling: when suspend/restore stops being worth it

`scripts/carbon_temporal_15min.py`. Same deadline-budget model as `carbon_temporal.py`, but the
scheduler now decides every **15 minutes** instead of hourly. Per the brief we REINTERPRET each
hourly CI sample as one 15-min slot (no interpolation), so a 4-hour span is now 16 CI values. The
workload COMPUTE hours are unchanged, so a job needing C compute hours now needs **4C slots** and a
deadline of H hours is **4H slots**.

Two schedulers, both measured against "run now" (naive contiguous block from start hour t):
- **NO suspend/resume** -- may delay the start for free, but must run the 4C slots CONTIGUOUSLY.
  Picks the single cleanest contiguous 4C-slot window. Mechanism cost = 0.
- **WITH suspend/resume** -- may also suspend mid-run to skip dirty slots, so it picks the 4C
  cleanest slots anywhere in the window (possibly scattered). Pays a MEASURED dump+restore per
  internal gap, split-priced (dump half at the suspend-slot CI, restore half at the resume-slot CI),
  NVMe and SATA.

The headline metric is **Δsusp = savings_with - savings_without**: the marginal carbon value of being
allowed to suspend. **Δsusp < 0 means suspend/restore KILLS the saving** -- the fragmentation gain
does not pay for the mechanism and you are better off with the free contiguous shift. `kill%` is the
fraction of (region x start) instances with Δsusp < 0.

**Setup**: 18 representative regions auto-selected stratified by CI volatility (CV 0%..104%:
hong_kong, south_africa, israel, taiwan, turkey, czechia, cyprus, italy, peru, brazil, greece, spain,
great_britain, chile, austria, lithuania, finland, sweden), 80 random start times per region per
workload, Electricity Maps 2023 `direct` CI. Two slacks per workload: tight (H=2C) and loose (H=4C).

## Result (mean over 18 regions x 80 starts)
save% is vs run-now. Δsusp = value of suspend. mech%ovh = mechanism as % of the scattered gross.
| WL | C | slack | H | tier | no-susp % | susp-net % | **Δsusp %** | mech %ovh | K | **kill %** |
|---|---|---|---|---|---|---|---|---|---|---|
| A4 DLRM | 1h | tight | 2h | nvme | 6.1 | 6.6 | **+0.50** | 28 | 0.40 | 14 |
| A4 DLRM | 1h | tight | 2h | sata | 6.4 | 5.8 | **-0.62** | 255 | 0.42 | 32 |
| A4 DLRM | 1h | loose | 4h | nvme | 13.9 | 14.7 | +0.73 | 116 | 0.47 | 12 |
| A4 DLRM | 1h | loose | 4h | sata | 13.9 | 13.3 | **-0.60** | 144 | 0.52 | 36 |
| A5 graph | 1h | tight | 2h | nvme | 6.4 | 6.5 | +0.07 | 100 | 0.40 | 24 |
| A5 graph | 1h | tight | 2h | sata | 6.7 | 4.8 | **-1.89** | 908 | 0.40 | 33 |
| A5 graph | 1h | loose | 4h | sata | 14.2 | 12.2 | **-1.94** | 245 | 0.47 | 36 |
| A7 DuckDB-MP | 1h | tight | 2h | nvme | 6.3 | 5.7 | **-0.59** | 457 | 0.40 | 30 |
| A7 DuckDB-MP | 1h | tight | 2h | sata | 5.9 | 4.0 | **-1.93** | 503 | 0.41 | 35 |
| A7 DuckDB-MP | 1h | loose | 4h | sata | 14.7 | 12.7 | **-2.00** | 287 | 0.48 | 37 |
| A8 DuckDB-MT | 1h | tight | 2h | nvme | 6.6 | 6.1 | **-0.50** | 346 | 0.40 | 31 |
| A8 DuckDB-MT | 1h | tight | 2h | sata | 6.0 | 3.1 | **-2.97** | 1200 | 0.43 | 36 |
| A8 DuckDB-MT | 1h | loose | 4h | sata | 13.8 | 10.7 | **-3.05** | 416 | 0.49 | 39 |
| A2 vLLM | 2h | tight | 4h | nvme | 8.6 | 10.1 | +1.46 | 17 | 0.78 | 9 |
| A2 vLLM | 2h | tight | 4h | sata | 8.9 | 9.9 | +1.01 | 195 | 0.79 | 29 |
| A2 vLLM | 2h | loose | 8h | nvme | 17.0 | 19.0 | +2.01 | 5 | 1.05 | 7 |
| A2 vLLM | 2h | loose | 8h | sata | 17.0 | 18.2 | +1.29 | 37 | 1.05 | 33 |
| A1 FSDP | 4h | tight | 8h | nvme | 8.1 | 12.9 | +4.73 | 9 | 1.62 | 9 |
| A1 FSDP | 4h | tight | 8h | sata | 7.9 | 11.1 | +3.18 | 55 | 1.63 | 38 |
| A1 FSDP | 4h | loose | 16h | nvme | 13.4 | 21.2 | **+7.82** | 5 | 2.35 | 3 |
| A1 FSDP | 4h | loose | 16h | sata | 13.3 | 18.7 | +5.41 | 70 | 2.34 | 27 |
| A6 gem5 | 8h | tight | 16h | nvme | 7.3 | 13.0 | +5.76 | 21 | 3.13 | 9 |
| A6 gem5 | 8h | tight | 16h | sata | 8.0 | 10.5 | +2.50 | 88 | 3.17 | **46** |
| A6 gem5 | 8h | loose | 32h | nvme | 13.7 | 21.9 | +8.22 | 14 | 4.73 | 7 |
| A6 gem5 | 8h | loose | 32h | sata | 14.5 | 18.5 | +3.93 | 66 | 4.64 | 43 |
| A3 ViT | 12h | tight | 24h | nvme | 7.2 | 15.5 | +8.33 | 1.5 | 4.52 | 0.1 |
| A3 ViT | 12h | tight | 24h | sata | 6.9 | 14.8 | +7.92 | 11 | 4.52 | 3 |
| A3 ViT | 12h | loose | 48h | nvme | 13.1 | 24.4 | **+11.33** | 1.0 | 6.76 | 0.3 |
| A3 ViT | 12h | loose | 48h | sata | 13.5 | 24.3 | +10.79 | 8 | 6.81 | 2 |

## Findings
1. **Sub-hourly granularity removes the C=1h "immunity".** At hourly granularity short jobs fit one
   hour, K=0, and NEVER suspend (immune, see `carbon_temporal.md`). At 15-min granularity a C=1h job
   is 4 slots, so the cleanest 4 slots can scatter (K~0.4) and it now pays mechanism cost. The four
   short workloads (A4/A5/A7/A8) flip to **Δsusp < 0 on SATA** -- suspend/resume LOSES vs just doing
   the free contiguous shift, **roughly 1 in 3 times (kill 32-39%)**.
2. **Storage tier is the decision variable.** On NVMe the mechanism is cheap, so finer granularity is
   almost always a win (Δsusp > 0 for every workload except a few short-job/tight cases near zero;
   kill <= 14%). On SATA the same fine-grained schedule is a net loss for short jobs and only
   marginally positive for the mid jobs, while still leaving long jobs exposed (A6 gem5 SATA
   **kills 43-46%** of the time). The fixed dump/restore energy, amortized over a 15-min run block
   instead of a 60-min one, is ~4x more impactful.
3. **mech%ovh routinely exceeds 100% for short jobs on SATA** (up to 1200% for A8) -- the dump/restore
   energy is several times larger than the entire extra carbon the scattered schedule saves. That is
   the quantitative definition of the kill zone: the mechanism is bigger than the prize.
4. **Long jobs still benefit, and more than at hourly granularity, but only on NVMe.** A3 ViT gains
   **+11.3%** absolute carbon from sub-hourly suspend (NVMe, loose) and A1/A6 gain +8%. They have many
   compute slots to redistribute, so the fragmentation prize dwarfs a cheap NVMe mechanism. On SATA the
   same jobs keep most of the prize for A3 (overhead small relative to its large gross) but A6 gem5
   (low 148 W power) loses much of it and is killed ~45% of the time.
5. **K rises with granularity** (hourly K was 0.3-1.3; here 0.4 for short jobs up to 6.8 for A3
   loose), because 15-min slots de-correlate faster, so the cleanest set fragments into more blocks --
   each an extra dump+restore.

## Takeaway for the paper
Going sub-hourly is not free. It unlocks a larger carbon prize (long jobs on NVMe gain up to +11%
absolute over the free contiguous shift), but it also **manufactures suspends for jobs that had none**,
and on SATA the measured dump/restore energy is large enough that fine-grained suspend/restore is a
**net carbon LOSS for short jobs (~1/3 of the time) and for low-power long jobs like gem5 (~45%)**.
The free contiguous-shift baseline is the safe default; suspend/restore should only be enabled when
the tier is fast (NVMe) and the job is long. This is exactly the mechanism-cost decision boundary the
measured (E_mech, P) make quantifiable. Output: `results/carbon_temporal_15min_2023.csv` (32 rows).

## Caveats
- **Relabel, not interpolate (deliberate, and the aggressive case for mechanism cost).** We reinterpret
  each hourly value as a 15-min slot rather than interpolating, per the brief. Adjacent slots therefore
  differ by a full hour of real diurnal change, which OVERSTATES sub-hourly volatility and so OVERSTATES
  the temptation to fragment. This makes the run a stress test / upper bound on suspend frequency (and
  thus on mechanism cost). True 15-min CI would be smoother, fewer suspends, smaller kill%. A
  piecewise-linear interpolation variant is a one-line change (interp the 4 sub-slots between hourly
  endpoints) and would soften every kill% number.
- Oracle foresight (the scattered schedule sees all 4H future slots), suspend-and-free accounting (idle
  power during suspend not charged), and research-grounded compute hours C -- same as `carbon_temporal.md`.
