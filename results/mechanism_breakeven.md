# Mechanism break-even: the decision rule that prevents misleading carbon savings

`scripts/mechanism_breakeven.py`. The capstone: both legs (temporal suspend/restore, spatial
suspend/migrate/restore) reduce to ONE decision inequality, and the threshold is a MEASURED,
CI-independent per-job constant `r*`. This is the quantity a carbon-aware scheduler must subtract at
decision time; counting the carbon prize while ignoring `r*` reports savings that are actually losses.

## Derivation
A shift pays off only if the carbon prize exceeds the measured mechanism carbon:
```
prize  >=  N_cycles * E_mech * CI_mech
```
Write the prize as a relative CI improvement f = dCI/CI on the job compute energy E_compute = P*C:
```
prize = f * CI * E_compute
```
Cancel CI -> a dimensionless, CI-INDEPENDENT, MEASURED break-even constant (the mechanism's carbon tax rate):
```
f  >=  r*  =  N_cycles * E_mech / E_compute
```
- Temporal: E_mech = dump + restore (local).            N = K suspends (>= 1 if you suspend at all).
- Spatial : E_mech = dump + e_net*S + restore (+WAN).   N = M migrations (~1).

`r*` is the fraction of clean-energy advantage the job must capture to break even on ONE mechanism
cycle. The scheduler rule: **shift only if predicted f > N * r*; otherwise run-now / stay-home.**

## Break-even r* (per job, per tier)
| WL | C | P W | S GB | E_comp kJ | temporal r* nvme | sata | spatial r* nvme | sata |
|---|---|---|---|---|---|---|---|---|
| A1 | 4 | 1471 | 148 | 21182 | 0.17% | 1.25% | 2.69% | 3.76% |
| A2 | 2 | 497 | 41 | 3578 | 0.15% | 1.00% | 4.27% | 5.12% |
| A3 | 12 | 526 | 35 | 22723 | 0.02% | 0.17% | 0.58% | 0.73% |
| A4 | 1 | 312 | 33 | 1123 | 0.41% | 3.23% | 10.99% | 13.81% |
| A5 | 1 | 255 | 71 | 918 | 1.36% | 6.52% | 29.20% | 34.36% |
| A6 | 8 | 148 | 50 | 4262 | 0.34% | 1.49% | 4.57% | 5.71% |
| A7 | 1 | 311 | 100 | 1120 | 3.14% | 6.65% | 35.30% | 38.80% |
| A8 | 1 | 302 | 100 | 1087 | 2.75% | 8.89% | 35.86% | 42.01% |

Absolute break-even CI gap dCI* = r* * CI at a 300 gCO2/kWh grid (below this the mechanism LOSES):
| WL | temporal sata dCI* | spatial sata dCI* |
|---|---|---|
| A3 | 0.5 | 2.2 |
| A1 | 3.7 | 11.3 |
| A6 | 4.5 | 17.1 |
| A4 | 9.7 | 41.4 |
| A5 | 19.5 | 103.1 |
| A7 | 19.9 | 116.4 |
| A8 | 26.7 | 126.0 |

## r* predicts every measured kill (validation against carbon_temporal_15min / carbon_spatial_zones)
- A7 spatial SATA r* = 38.8%. EAST_ASIA in-zone spread ~8% << 38.8% -> killed (measured net -18.8%,
  kill 71%). EUROPE spread 71% >> 38.8% -> safe (kill 6%). The threshold IS the empirical boundary.
- A3 spatial r* = 0.73%. Any zone with >1% spread clears it -> A3 safe everywhere (EAST_ASIA 8% >> 0.73%).
- A6 spatial r* = 5.71%. EAST_ASIA spread ~9.8% is just above it -> borderline (net +3%, 38% of instances lose).
- Temporal A7 SATA r* = 6.65%/suspend; with K suspends the bar is K*6.65% -> matches the SATA short-job kills.

## Two regimes
- **Temporal r* tiny (0.02-6.6%):** a small diurnal swing pays for a suspend, EXCEPT short/large-state
  jobs on SATA. Suspend-in-place is usually safe.
- **Spatial r* large (0.6-42%):** e_net*S inflates E_mech ~10-28x, so a migration needs a big in-zone
  CI spread; it fails wherever the reachable zone is tight (East Asia, Middle East) for short,
  large-state, or low-power jobs. A7 needs a 116 gCO2/kWh gap (CI=300) -- more than East Asia's entire spread.

## The proposal
`r*` is the mechanism cost expressed in the same units as the carbon signal the scheduler already
uses. The apparent-minus-real savings gap is exactly N*r*. A correct carbon-aware scheduler runs a
one-line admission test from (a) the measured per-workload-class r* (a constant from our campaign) and
(b) the predicted relative CI improvement f -- NO absolute CI needed: shift iff f > N*r*. This converts
"the mechanism cost" from a post-hoc caveat into a decision input that prevents misleading carbon savings.
