# Spatial (region-migration) carbon scheduling, net of the WAN migrate-leg cost

`scripts/carbon_spatial.py` (global oracle) and `scripts/carbon_spatial_zones.py` (zone-limited +
outlier-trimmed). Spatial analogue of `carbon_temporal.py`. A job needs C contiguous compute hours and
runs continuously (no suspend-in-place), but each hour it MAY relocate to a cleaner region. Migration =
stage-and-forward: dump at source + ship the S GB state over the WAN + restore at dest. One-way
(results stream back, negligible); infinite capacity assumed (reachability, not capacity -- see Caveats).

## Mechanism: migrate = suspend/resume + the network leg
Migration is the temporal suspend/resume PLUS shipping the dumped state, `e_net*S`:
```
migrate_carbon = (E_dump + e_net*S) * CI_source  +  E_restore * CI_dest      (split-priced)
```
- `e_net = 3.6 kJ/GB` (full-path marginal inter-DC; `network_coefficients_lit.md`). Endpoints-only is
  ~50-100 J, negligible. The network leg is priced at the SOURCE grid CI (egress; a conservative choice
  since you migrate dirty->clean, so source CI is the high end -- see Caveat on attribution).
- `E_dump`/`E_restore` are the MEASURED per-tier dump/restore halves; `S` is the measured image size.

The network leg DOMINATES the mechanism. Per workload, `e_net*S` vs the local suspend/resume energy:
| WL | S GB | E_mech NVMe | E_mech SATA | net = e_net*S | net/NVMe | net/SATA |
|---|---|---|---|---|---|---|
| A1 | 148 | 36 kJ | 265 kJ | 533 kJ | 14.8x | 2.0x |
| A2 | 41 | 5.3 | 35.7 | 148 | 27.8x | 4.1x |
| A4 | 33 | 4.7 | 36.3 | 119 | 25.5x | 3.3x |
| A6 | 50 | 14.7 | 63.4 | 180 | 12.3x | 2.8x |
| A7 | 100 | 35.2 | 74.4 | 360 | 10.2x | 4.8x |
| A8 | 100 | 29.9 | 96.7 | 360 | 12.0x | 3.7x |
So migrating a dumped workload costs ~10-28x a local NVMe suspend/resume, ~2-5x a SATA one (energy),
and ~10-50x in stall (WAN ~8-25 s/GB vs local ~0.2-2.2 s/GB). **Temporal is mechanism-energy-limited;
spatial is latency-limited and, once zone-bounded, prize-limited.**

## Global oracle: huge prize, M~1, kill ~0
Four schedules vs B1 (stay home): B2 ideal (cleanest region/hour, free migration), B3 greedy (chase +
pay), B4 DP-optimal (migration-aware). 18 home regions x 60 starts, all 45 grids reachable.
- **B2 ideal ~= 92-95%** -- cross-region CI spread (sweden ~3 vs poland ~714) dwarfs diurnal (~2-3x).
- **M ~= 1.** The "chase A->B->C" barely happens: a few grids (sweden/norway) are PERSISTENTLY clean,
  so optimal = migrate once and stay. Even 12-hour A3 averages 1.3 migrations. (Independently matches
  EuroSys'24's "a single migration captures most of the reduction.")
- **Migrate energy overhead = state / job-length:** 0.6% (A3, 35 GB / 12 h) to ~35% (A7, 100 GB / 1 h).
- **kill ~0% everywhere.** The ~92% prize swamps even a 35 pp migrate overhead. Opposite of temporal.
- Tier barely matters (1-3 pp): `e_net*S` dominates the local dump/restore.

## Zone-limited + outlier-trimmed: the spatial kill zone
The "spatial is free" result is an ARTIFACT of one or two outlier-clean grids. Real migration is
reachability-bounded (residency / provider footprint). We TRIM the 5 greenest (sweden 3 ... switzerland
54) and 5 dirtiest (australia 546 ... cyprus 752) grids globally, then bound migration to a REGION
(continent) or finer SUBREGION. `e_net*S` is geography-INVARIANT, so as the in-zone CI spread shrinks
the fixed migrate overhead eventually exceeds the prize.

**Prize tracks the in-zone CI spread and collapses; the A7 overhead stays ~24-34 pp in every zone:**
| zone | gran | #reg | CI range | prize (idl%) | A7 net% | A7 ovh | A7 kill% |
|---|---|---|---|---|---|---|---|
| EAST_ASIA (jp,kr,tw) | sub | 3 | 377-450 | **8.3** | **-15.6** | 23.9 | **60.8** |
| MIDEAST | sub | 2 | 373-463 | 10.3 | -11.7 | 21.4 | 53.8 |
| BALTIC | sub | 3 | 134-216 | 30.1 | 3.8 | 22.5 | 28.3 |
| EAST_EU | sub | 6 | 208-472 | 35.4 | 4.9 | 30.4 | 31.2 |
| SOUTH_EU | sub | 6 | 106-339 | 44.5 | 16.5 | 30.5 | 19.2 |
| WEST_EU | sub | 6 | 122-329 | 47.1 | 16.0 | 30.6 | 20.0 |
| LATAM | region | 4 | 56-221 | 49.2 | 22.1 | 27.4 | 3.8 |
| ASIA_PACIFIC | region | 6 | 61-515 | 71.1 | 41.1 | 29.5 | 0.0 |
| EUROPE | region | 23 | 88-472 | 71.3 | 36.8 | 34.4 | 6.5 |

**Crossover principle:** migration kills exactly when **in-zone prize < the fixed migrate overhead**.
A7's overhead is ~30 pp regardless of zone, so it survives wide zones (EUROPE, WEST_EU, LATAM) and dies
in tight ones (EAST_ASIA, MIDEAST, EAST_EU).

**EAST_ASIA (Japan/Korea/Taiwan, tightest realistic cluster), full kill table, SATA:**
| WL | C | S | prize% | net% | ovh | kill% |
|---|---|---|---|---|---|---|
| A4 | 1 | 33 | 8.0 | -0.8 | 8.9 | 41.7 |
| A5 | 1 | 71 | 9.6 | -10.8 | 20.4 | 54.2 |
| A7 | 1 | 100 | 8.6 | **-18.8** | 27.5 | **70.8** |
| A8 | 1 | 100 | 9.3 | -17.7 | 26.9 | 62.5 |
| A6 | 8 | 50 | 9.8 | 3.1 | 6.7 | 38.3 |
| A1 | 4 | 148 | 8.1 | 4.7 | 3.4 | 23.3 |
| A3 | 12 | 35 | 8.9 | 7.8 | 1.1 | 11.7 |
In a tight dirty cluster, migrating a short large-state job (A7/A8) emits MORE than running at home
60-71% of the time (~18 pp net loss). Even low-power A6 is killed ~38%. Only long+small-state (A3) and
high-power (A1) stay safely positive (overhead ~1-3 pp).

## Temporal <-> spatial symmetry (for the paper)
| | per-job mechanism (MEASURED) | victim job | what removes the prize |
|---|---|---|---|
| Temporal | suspend + restore | short / low-power | flat grid + tight slack |
| Spatial | migrate (+ e_net*S) | short / **large-state** / low-power | **tight/dirty zone (no clean target)** |
Same structural victims (short, low-power), one new one (large state), and "flat grid" replaced by
"tight/dirty reachable zone." Both reduce to: the mechanism kills when its cost exceeds the carbon prize.

## Caveats
- **Infinite capacity (reachability, not capacity).** B2 ideal matches EuroSys'24's ideal (~96%). At
  fleet scale, finite per-region capacity, load-induced marginal-CI rise, and synchronized resume
  (thundering herd) shrink the prize [Sukprasert EuroSys'24; CarbonFlex; Jiang e-Energy'26; Radovanovic
  VCC]. These only shrink the benefit, not the mechanism cost, so our findings upper-bound the prize and
  lower-bound where the mechanism flips the decision. Joint capacity-mechanism modeling left to future work.
- **Network-leg CI attribution.** `e_net*S` priced at the SOURCE grid (egress). The diffuse path crosses
  many grids; source CI is the conservative (high) end since you migrate dirty->clean. A `--net-ci`
  bracket (source/dest/mean/global) is the stated sensitivity.
- Average (not marginal) direct CI; one-way migration; 2023; C research-grounded.
