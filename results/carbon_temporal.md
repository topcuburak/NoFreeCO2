# Carbon benefit of temporal (suspend-in-place) scheduling, net of suspend/restore cost

`scripts/carbon_temporal.py`. Deadline-budget shifting: a workload kicked at hour t needs C hours of
COMPUTE and has a deadline window H (∈ 4,6,8,12,16,24,36,48 h), seeing H hourly CI values. Carbon-aware
runs the C *cleanest* hours in [t,t+H], suspends the rest; each run<->suspend gap is a dump+restore =
measured (E_mech, T_mech). Monte-Carlo over start hours, per workload, per tier, per country.

- **CI data**: Electricity Maps hourly `direct` gCO2eq/kWh, ~50 countries, 2023 (`carbon_data/`).
- **Workload inputs**: measured running power P + per-suspend E_mech (NVMe/SATA) from the campaign;
  **compute time C is ASSUMED** (A1=4 A2=2 A3=4 A4=3 A5=2 A6=6 A7=3 A8=3 h) -- adjustable.
- net = naive - (cleanest-C-hours compute + K·E_mech·CI). gross = naive - compute (no mechanism).

## Headline (H=24, avg ~50 grids)
| WL | power | gross savings | NVMe overhead (of savings) | SATA overhead | SATA net-positive |
|---|---|---|---|---|---|
| A1 FSDP | 1471 W | 18.9% | 0.4% | 3.1% | 87% |
| A2 vLLM | 497 W | 20.4% | 0.2% | 1.1% | 88% |
| A3 ViT | 526 W | 18.9% | 0.2% | 1.3% | 90% |
| A4 DLRM | 312 W | 19.6% | 0.3% | 2.0% | 88% |
| A5 graph | 255 W | 20.4% | 0.7% | 3.4% | 85% |
| A7 DuckDB-MP | 311 W | 19.6% | 1.2% | 4.0% | 85% |
| A8 DuckDB-MT | 302 W | 19.6% | 0.9% | 5.4% | 83% |
| A6 gem5 | 148 W | 17.5% | 1.7% | 7.4% | 83% |

## Findings
1. **Temporal shifting saves ~17-25% carbon** (H=24->48, avg; up to ~33-36% in volatile grids like
   DE/CA), growing monotonically with deadline slack (H-C). Savings %% track *relative* CI swing, so a
   flat-but-low grid (France) can show high %% at low absolute gCO2; high-CI grids save more grams.
2. **Mechanism overhead is small but tier- and power-dependent.** NVMe < 2% of the gross savings
   (negligible); SATA 1-7%. It scales **inversely with workload power**: the fixed E_mech is a larger
   slice of a low-power job's energy, so A6 gem5 (148 W) loses 7.4% of its savings to SATA suspends vs
   A1 FSDP (1471 W) at 3.1%.
3. **K stays low (0.3-0.8 suspends/instance).** Grid CI is temporally autocorrelated (diurnal solar/
   wind), so the cleanest hours cluster and the optimal schedule rarely fragments -- the feared
   "many suspends" does not materialize for deadline-budget shifting at hourly granularity.
4. **When suspending is NOT worth it (mechanism cost flips the decision):** at short horizons / flat
   grids the overhead exceeds the saving. Fraction of instances NET-NEGATIVE on SATA: H=6 ~47-100%,
   H=12 ~29-37%, H=24 ~13-17%, H=48 ~10-14%. The danger corner is **low slack + slow tier + low-power
   workload + flat grid**.

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
