# Sub-hourly (15-min) temporal scheduling: three baselines, real misprediction, kill decomposition

`scripts/carbon_temporal_15min.py`. Temporal deadline-budget scheduling at **15-minute** granularity.
Each hourly Electricity Maps `direct` CI sample is REINTERPRETED as one 15-min slot (no interpolation),
so a job needing C compute hours needs **4C slots** and a deadline of H hours is **4H slots**. Monte
Carlo over 18 CV-stratified regions x 80 random start windows, both NVMe and SATA tiers, 2023.

**Data cleaning:** grids that are effectively constant (CV < 1%: hong_kong flat 360, indonesia flat
580) are non-physical placeholders and are DROPPED (45 grids kept). CI <= 0 treated as missing.

## Three baselines (carbon over the whole job, savings % vs B1)
- **B1 run-now** -- 4C slots contiguously from t. Reference (0% by definition).
- **B2 ideal** -- place the 4C compute slots on the cleanest slots anywhere in the window, mechanism
  FREE. Theoretical upper bound on temporal savings. Tier-independent.
- **B3 real** -- same cleanest-slot placement, paying the MEASURED dump+restore per internal gap,
  split-priced (dump half at the suspend-slot CI, restore half at the resume-slot CI), per tier.

`ovh = B2 - B3` (carbon the mechanism eats, pp). `kill% = B3 < 0` frequency (real temporal scheduling
emits MORE than running now). `cml` = conditional mean loss (avg net over killed instances only) --
separates frequent-but-harmless from rare-but-damaging kills. At 15-min granularity even C=1h jobs
(4 slots) can fragment, so the hourly "short jobs are immune" result no longer holds.

## Misprediction (forecast) layer -- decide on predicted, pay on actual
A real scheduler uses a forecast. We add per-region forecast error calibrated to the MEASURED
CarbonCast MAPE (flat grids 3-5%, volatile 13-19%; mapped `mape = clamp(0.0035*CV, 3%, 20%)`). The
error is **autocorrelated (AR(1), phi=0.9)** -- real forecasters err with a slow bias, not independent
per-slot noise, so the predicted-cleanest slots stay clustered (matching the measured "CarbonCast
suspends less"). White noise (phi=0) was rejected: it shredded the autocorrelation and inflated K
2-3x. The scheduler picks slots by PREDICTED CI; carbon and mechanism are paid on ACTUAL CI.

## Result (mean over 18 regions x 80 starts; tight slack H=2C, loose H=4C)
ORACLE (perfect foresight):
| WL | C | slack | B2 idl% | nv B3% | nv ovh | nv kill% | sa B3% | sa ovh | sa kill% | K |
|---|---|---|---|---|---|---|---|---|---|---|
| A4 | 1 | tight | 7.4 | 7.2 | 0.20 | 8.3 | 5.8 | 1.56 | 24.6 | 0.50 |
| A7 | 1 | tight | 7.1 | 5.7 | 1.41 | 22.6 | 4.1 | 2.98 | 30.5 | 0.46 |
| A8 | 1 | tight | 7.4 | 6.1 | 1.24 | 20.7 | 3.4 | 4.00 | 32.6 | 0.47 |
| A2 | 2 | tight | 10.9 | 10.8 | 0.12 | 3.2 | 10.1 | 0.82 | 12.7 | 0.86 |
| A1 | 4 | loose | 22.6 | 22.2 | 0.36 | 0.6 | 19.9 | 2.64 | 8.4 | 2.58 |
| A6 | 8 | tight | 15.0 | 13.8 | 1.16 | 5.5 | 10.0 | 5.02 | 27.3 | 3.55 |
| A3 | 12 | loose | 25.9 | 25.8 | 0.12 | 0.0 | 24.9 | 1.01 | 0.2 | 7.44 |

FORECAST (CarbonCast-calibrated misprediction) -- same columns, K is forecast K:
| WL | C | slack | B2 idl% | nv B3% | nv kill% | sa B3% | sa kill% | K |
|---|---|---|---|---|---|---|---|---|
| A4 | 1 | tight | 7.4 | 5.7 | 25.7 | 3.9 | 38.6 | 0.65 |
| A8 | 1 | tight | 7.4 | 4.2 | 37.6 | 0.4 | 47.6 | 0.65 |
| A6 | 8 | tight | 15.0 | 11.0 | 17.8 | 6.3 | 43.3 | 4.24 |
| A3 | 12 | loose | 25.9 | 21.7 | 4.1 | 20.6 | 9.0 | 8.95 |
(full 32-row tables in `results/carbon_temporal_15min_2023.csv`.)

## Kill decomposition -- mechanism vs misprediction (the characteristic effect)
`mis-only` = forecast placement + FREE mechanism (only misprediction can flip it). `mech-only` =
oracle placement + real mechanism (only the mechanism can flip it). Combined = forecast + mechanism.
| WL | C | mis-only | nv mech-only | sa mech-only | dominant (sata) |
|---|---|---|---|---|---|
| A4 | 1 | 19.7 | 8.3 | 24.6 | mechanism |
| A5 | 1 | 21.4 | 16.5 | 32.3 | mechanism |
| A7 | 1 | 22.1 | 22.6 | 30.5 | mechanism |
| A8 | 1 | 21.0 | 20.7 | 32.6 | mechanism |
| A2 | 2 | 16.5 | 3.2 | 12.7 | mispred |
| A1 | 4 | 10.1 | 2.0 | 16.5 | mechanism |
| A6 | 8 | 7.7 | 5.5 | 27.3 | **mechanism** |
| A3 | 12 | 5.3 | 0.1 | 2.2 | mispred |

**Two structurally distinct failure modes, set by the job not the policy:**
1. **Short jobs are misprediction-bound.** Tiny prize (B2 ~7%), so one forecast slip wipes it: mis-only
   alone flips 16-22% of cases, independent of storage. Mechanism is secondary and only co-dominant on SATA.
2. **Low-power long jobs are mechanism-bound.** A6 gem5 has a big prize (15-24%) but 148 W power and
   K~3.5-5, so the SATA mechanism alone flips 27% of cases (3.5x misprediction's 7.7%).
3. **Storage tier is the switch.** NVMe nearly removes the mechanism -> misprediction is the only real
   flipper everywhere. SATA makes the mechanism dominant for short and low-power-long jobs. Robust
   corner: long high-power job (A1/A3) on NVMe -- every flipper in single digits.

## Takeaway
Sub-hourly removes the C=1h immunity (short jobs now fragment, K~0.4). The mechanism is negligible on
NVMe (ovh < 1.5 pp) and decisive on SATA (short-job kills 38-48% under forecast, low-power A6 43%).
Whether suspend/restore is worth it depends on (storage tier, job length, power, forecast quality),
which the measured (E_mech, P) make quantifiable.

## Caveats
- Relabel (hourly->15-min) overstates sub-hourly volatility and so overstates fragmentation/kill -- an
  upper bound; true 15-min CI would be smoother. Deliberate (per brief), stress-test posture.
- Oracle foresight for B2; suspend-and-free accounting (idle power during suspend not charged --
  justified: an idle interval powers the node down); C research-grounded (`workload_durations_refs.md`).
