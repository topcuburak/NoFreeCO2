# Per-job mechanism profile: carbon AND latency overhead -> smart per-job scheduling

`scripts/mechanism_profile.py`. The scheduler does not apply one global carbon-aware policy. For EACH
job it computes TWO per-job overheads from measured constants and decides individually, because every
job experiences a different overhead on each axis:

```
CARBON axis    r* = E_mech / E_compute     break-even relative CI gain per cycle (mechanism_breakeven.md)
LATENCY axis   l* = T_mech / runtime       stall added as a fraction of runtime per cycle
```
- Temporal: E_mech = dump+restore (kJ), T_mech = dump+restore wall time (s, MEASURED), N = K suspends.
- Spatial : E_mech += e_net*S, T_mech += S / BW_net (WAN transfer stall), N = M migrations (~1).

The scheduler's decision (per job, two objectives):
```
net carbon    = f - N*r*              shift only if > 0           (f = predicted relative CI prize)
added latency = N*l* * runtime        shift only if <= deadline slack
```
Choose run-now / suspend / migrate to maximize net carbon subject to latency <= slack.

## Profile (SATA tier shown; WAN 15 s/GB single tuned stream)
| WL | C | S | temporal r* | temporal l* | spatial r* | spatial l* | xfer |
|---|---|---|---|---|---|---|---|
| A3 | 12 | 35 | 0.17% | 0.4% | 0.73% | 1.6% | 522 s |
| A6 | 8 | 50 | 1.49% | 1.3% | 5.71% | 3.9% | 746 s |
| A1 | 4 | 148 | 1.25% | 4.4% | 3.76% | 19.7% | 2209 s |
| A2 | 2 | 41 | 1.00% | 2.4% | 5.12% | 10.9% | 612 s |
| A4 | 1 | 33 | 3.23% | 4.1% | 13.81% | 17.8% | 493 s |
| A5 | 1 | 71 | 6.52% | 8.7% | 34.36% | 38.1% | 1060 s |
| A7 | 1 | 100 | 6.65% | 15.9% | 38.80% | 57.4% | 1493 s |
| A8 | 1 | 100 | 8.89% | 13.8% | 42.01% | 55.2% | 1493 s |
(NVMe l* is ~3-4x smaller; full table both tiers from the script.)

## Why per-job (the axes are NOT correlated)
1. **A1 vs A6 -- the axes disagree.** A1 is carbon-CHEAP to migrate (r* 3.8%: high 1471 W power dilutes
   the mechanism) but latency-EXPENSIVE (l* ~20%: 148 GB takes ~37 min to ship). A6 is the opposite:
   latency-cheap (l* 3.9%: long 8 h runtime amortizes the stall) but carbon-moderate (r* 5.7%). A
   carbon-only policy migrates A1 and eats 20% latency; a latency-only policy migrates A6 and may lose
   carbon in a tight zone. You need BOTH numbers, per job.
2. **Huge spread.** A3 cheap on both (shift freely); A7/A8 brutal on both -- spatial stall is 55-57% of
   the ENTIRE runtime (migrating a 1 h job adds ~25 min of transfer). No global policy fits.
3. **Temporal vs spatial differ for the same job.** A7 stall is 16% temporal vs 57% spatial -- same job,
   different leg, different decision. The scheduler also chooses the LEG per job.

## Contribution
The mechanism cost is a per-job, two-objective (carbon r*, latency l*) overhead the scheduler computes
from measured constants and trades against the predicted prize and the deadline slack -- not a flat
policy. Each job experiences a different overhead, so each is scheduled individually. Companion to
`mechanism_breakeven.md` (carbon axis) and `wan_migration_overhead.md` (latency model + boundaries).
