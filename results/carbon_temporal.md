# Carbon benefit of temporal (suspend-in-place) scheduling, net of suspend/restore cost

`scripts/carbon_temporal.py`. Deadline-budget shifting: a workload kicked at hour t needs C hours of
COMPUTE and has a deadline window H (∈ 4,6,8,12,16,24,36,48 h), seeing H hourly CI values. Carbon-aware
runs the C *cleanest* hours in [t,t+H], suspends the rest; each run<->suspend gap is a dump+restore =
measured (E_mech, T_mech). Monte-Carlo over start hours, per workload, per tier, per country.

- **CI data**: Electricity Maps hourly `direct` gCO2eq/kWh, ~50 countries, 2023 (`carbon_data/`).
- **Workload inputs**: measured running power P + per-suspend E_mech (NVMe/SATA) from the campaign;
  **compute time C is research-grounded** (A1=4 A2=2 A3=12 A4=1 A5=1 A6=8 A7=1 A8=1 h; see
  `results/workload_durations_refs.md` for sources/resources). Short jobs (A4/A5/A7/A8) fit one hour.
- net = naive - (cleanest-C-hours compute + mech). gross = naive - compute (no mechanism).
- **Split mechanism pricing**: per gap, dump half (suspend+store) is charged at the suspend-hour CI,
  restore half (load+resume) at the resume-hour CI (not their average). Dump/restore energies measured
  per workload/tier (A7 notably restore>dump on SATA). Result ~unchanged vs averaging (gap is short ->
  CI_suspend ~ CI_resume), but more precise.

## Headline (H=24, avg ~50 grids)
| WL | C | power | gross savings | NVMe overhead | SATA overhead | K |
|---|---|---|---|---|---|---|
| A4 DLRM | 1 h | 312 W | 21.3% | 0% | 0% | 0.00 |
| A5 graph | 1 h | 255 W | 21.3% | 0% | 0% | 0.00 |
| A7 DuckDB-MP | 1 h | 311 W | 21.3% | 0% | 0% | 0.00 |
| A8 DuckDB-MT | 1 h | 302 W | 21.3% | 0% | 0% | 0.00 |
| A2 vLLM batch | 2 h | 497 W | 20.4% | 0.2% | 1.1% | 0.27 |
| A1 FSDP FT | 4 h | 1471 W | 18.9% | 0.4% | 3.1% | 0.56 |
| A6 gem5 sim | 8 h | 148 W | 15.8% | 1.7% | **8.0%** | 0.96 |
| A3 ViT train | 12 h | 526 W | 11.9% | 0.5% | 1.9% | 1.34 |
(overhead = fraction of gross savings lost to suspend/restore.)

## Findings
1. **Job duration decides everything.** Short jobs (A4/A5/A7/A8, C=1 h) fit inside one hour, so they
   shift to the single cleanest hour with **K=0 suspends and ZERO mechanism overhead** -- and get the
   highest gross savings (~21%, they can pick the very best hour). Only **multi-hour jobs (A1, A2, A3,
   A6) ever suspend** and pay overhead. So the suspend/restore overhead is a *long-job* phenomenon.
2. **Temporal shifting saves ~12-25% carbon** at H=24 (more with horizon; up to ~33% in volatile grids
   like DE/CA). Longer jobs save *less* (they must occupy more hours incl. less-clean ones: A3 C=12
   fills half of H=24 -> only 11.9%).
3. **Mechanism overhead is small, tier- and power-dependent, and worst for low-power long jobs on
   SATA.** NVMe < 2% of savings (negligible) everywhere. SATA: A2 1.1%, A1 3.1%, A6 gem5 **8.0%** --
   the fixed E_mech is a bigger slice of gem5's tiny 148 W draw over a long 8 h job.
4. **K stays low (0.3-1.3).** Grid CI is temporally autocorrelated (diurnal), so cleanest hours
   cluster and the optimal schedule rarely fragments; the feared "many suspends" doesn't materialize.
5. **When suspending is NOT worth it:** at short horizons / flat grids the overhead exceeds the saving
   for the long-job workloads (10-17% of instances net-negative at H=24, more at H<=8). The danger
   corner is **long job + low power + slow tier + low slack + flat grid** (A6 gem5 is the poster child).

## Takeaway for the paper
Mechanism cost is the **decision-flipping factor in the low-slack / SATA / low-power corner**, while
negligible in the high-slack / NVMe / high-power corner. So whether to suspend is not "always yes" --
it depends on (deadline slack, storage tier, workload power, grid volatility), which the measured
(E_mech, P) make quantifiable. Output: `results/carbon_temporal_2023.csv` (126 rows: wl x tier x H).

## Caveats / next
- Compute times C are assumed; the curves shift with C (bigger C -> needs bigger H for slack).
- Hourly granularity + greedy-cleanest (carbon-optimal, mechanism-agnostic) -> a mechanism-aware
  scheduler would trade a little carbon for fewer suspends, lowering K further.
- Initial deferral (t -> first clean hour) modeled as a free delayed start (idle-wait), not a suspend.
- 2023 only; multi-year and the absolute-gCO2 view are easy extensions.
