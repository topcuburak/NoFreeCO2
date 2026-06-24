# CI forecasting for carbon-aware scheduling: methods, accuracy, oracle gap, forecaster overhead

Our carbon analysis uses an ORACLE (perfect-foresight) CI. This documents what real forecasters
achieve and what they cost -- to bound the oracle and to frame the forecaster's own footprint.
(Research-backed, sources verified against primary PDFs.)

## 1. Methods / tools
| tool | type | inputs | horizon | accuracy | source |
|---|---|---|---|---|---|
| Electricity Maps Forecast API | commercial ML | flow-traced grid data | 6/24/48/72 h | RE-share err "<10%"; no public CI MAPE | app.electricitymaps.com/developer-hub |
| WattTime MOER + forecast | commercial ML (MARGINAL) | MOER hist, interchange, curtailment | +72 h, 5-min | MAE 1-9% most regions; wind harder | watttime.org/data-science/methodology-validation |
| UK NESO Carbon Intensity API | free public, ML+power-flow | gen x emission factors, Met Office wx | 96+ h, 30-min, 14 regions | indep tracker MAE ~24.5 gCO2/kWh, 5-31% | carbonintensity.org.uk |
| ENTSO-E Transparency | EU data platform (INPUT, not CI) | day-ahead wind/solar/gen/load | day-ahead | n/a (feeds models) | transparency.entsoe.eu |
| **CarbonCast** | two-tier ANN -> CNN-LSTM (open) | gen forecasts + hist CI + GFS weather | **96 h hourly, 13 regions** | **96h MAPE 4.8-13.9%, avg ~10%** | BuildSys 2022 (lass.cs.umass.edu/papers/pdf/buildsys2022-carboncast.pdf) |
| DACF | ML source-decomp + ARIMAX | EIA/ENTSO-E gen, GFS, solar/wind | 24 h | NRMSE 0.095-0.183 | e-Energy 2022 |
| LSTM/CNN-LSTM/Transformer/N-BEATS | DL families | hist CI + mix + weather | 24-96 h | CNN-LSTM dominant; ensembles (EnsembleCI/CarbonX) | arXiv 2505.01959 / 2510.01521 |
| persistence / diurnal avg | naive baselines | CI series only | <=1 h best | "hard to beat <=1 h"; the benchmark | CarbonCast PDF |

## 2. Accuracy vs horizon and grid
- **Grows with horizon**: CarbonCast avg ~7% (24h) -> ~12.5% (96h); roughly DOUBLES 24->96 h
  (EnsembleCI: DE 10.45% -> 24.7%). [arXiv 2505.01959]
- **Volatile grids much worse**: CISO ~13%, Germany 14-25%, **Denmark up to 49%** (LiteCast
  2511.06187); flat/dispatchable **PJM 5-7%** (lowest). Wind-dominated harder than solar.
- Gap: no citable per-horizon MAPE for France/Ontario, UK NESO, Electricity Maps, WattTime.

## 3. Forecast vs ORACLE in scheduling (the key bound)
- **Sukprasert/Irwin, EuroSys 2024** (the central cite): ~14% MAPE -> only **~3% carbon increase**
  vs oracle; even **50% error -> only ~10-12%** loss. Scheduling is REMARKABLY robust to CI error.
- **Wiesner "Let's Wait Awhile," Middleware 2021**: oracle adds only **1-2 percentage points** over a
  5%-error forecast (CA, DE). So a real forecaster captures ~**90-97% of oracle savings**.
- **BUT forecast error -> more interruptions (the mechanism-cost link)**: Wiesner -- forecast error has
  "almost no impact" on the Non-Interrupting strategy but "considerable impact" on the Interrupting one,
  which "is more susceptible to optimize for negative spikes" (chases noise). So error hurts the
  SUSPEND-heavy policy specifically -> more dump/restore cycles -> more mechanism overhead.
- **Hanafy, HotCarbon 2023** ("War of the Efficiencies"): even with an oracle, larger slack -> more
  checkpoints -> "carbon footprint of energy overheads [can] overweigh the reduction in carbon
  savings." Directly supports our mechanism-cost thesis.

## 4. Forecaster's OWN energy/carbon overhead (under-studied gap = opportunity)
- **No source measures it** (CarbonCast/EnsembleCI/CarbonX report no compute cost). The field ASSUMES
  it negligible.
- CarbonCast is tiny: ANN 3 dense (50/34/24) + CNN-LSTM (4,16 filters, 24 LSTM units), ~8760 rows/
  region. Training **~0.04-4.5 kWh/run** (Carbontracker bracket). Inference **uJ-mJ** (LSTM cell 3.8
  uJ; CNN <1 uJ/op). vs a GPU job at ~250-400 W/GPU -- the neural footprint is **9-12 orders smaller**.
- **Non-obvious**: the **data pipeline dominates, not the net** -- HTTP API ~25-55 J/call (3-5 orders >
  uJ inference) + GFS weather fetch/decode + retraining. Per-job: negligible. **Fleet-scale**
  (1000s of regions x hourly x retraining x ingest) is **unquantified by anyone** -- the only adjacent
  cost-benefit study is solar nowcasting (2210.04554), not grid-CI.

## Implications for our paper
1. **Our oracle savings are ~achievable** (real forecasters capture ~90-97%); the oracle is ~3%
   optimistic at realistic 14% MAPE. So the savings numbers stand, labeled as an upper bound.
2. **Forecast error AMPLIFIES our mechanism cost**: it specifically inflates the interrupt count K
   (chasing noisy CI dips), and we measured exactly what each extra suspend costs (E_mech, both tiers).
   So error -> more suspends -> more overhead -- the coupling Hanafy posited, now quantifiable with our
   numbers. A threshold/forecast policy variant in carbon_temporal.py would show K (and overhead) rise.
3. **Forecaster overhead is a measurable novel angle**: per-job negligible (uJ inference), but the
   amortized fleet-scale pipeline+retraining footprint is unmeasured -- and our energy methodology
   (NVML+RAPL, the same harness) could measure a CarbonCast-style forecaster end-to-end.
- Note: WattTime is MARGINAL (MOER), Electricity Maps/CarbonCast are AVERAGE CI; the two correlate
  negatively in ~55% of regions -- state our oracle uses AVERAGE (direct) CI.
