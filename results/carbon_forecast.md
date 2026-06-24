# Forecast-driven temporal scheduling: real CarbonCast predictor vs oracle

`scripts/carbon_temporal_forecast.py`. A real trained predictor (CarbonCast, open-source pretrained
CNN-LSTM) in the loop: the scheduler picks the C cleanest hours by PREDICTED CI but pays carbon (and
prices mechanism overhead) on ACTUAL CI. Compared head-to-head with the ORACLE (pick by actual).

- **Predictor**: CarbonCast Tier-2 CNN-LSTM, pretrained models run locally on ford (CPU). 13 regions,
  Jul-Dec 2021 test set. Output `[datetime, actual, predicted]`, 181 daily 96h forecast blocks/region
  (lead time 0..95 within a block; accuracy degrades along the block). data: carboncast_forecasts/.
- **Per-region MAPE** tracks volatility: FPL 3.4% / PJM 4.7% / PL 4.6% (flat) -> CISO 13% / DE 14% /
  BPAT 15% / ES 19% (volatile). Whole-file.
- Each daily block = one scheduling decision at forecast-refresh (window = first H of the block).

## Result (H=24, 13 regions, 2353 blocks)
| job | oracle | forecast | lost (pp) | K oracle->forecast | capture |
|---|---|---|---|---|---|
| short (C=1: A4/A5/A7/A8) | 15.0% | 11.8% | 3.18 | 0.00 -> 0.00 | 79% |
| A1 (C=4) | 13.5% | 11.2% | 2.3 | 0.43 -> 0.14 | 83% |
| A2 (C=2) | 14.6% | 11.8% | 2.8 | 0.22 -> 0.07 | 81% |
| A6 (C=8) | 10.8% | 9.2% | 1.6 | 0.83 -> 0.50 | 85% |
| A3 (C=12) | 7.8% | 6.4% | 1.4 | 1.24 -> 0.77 | 82% |

## Findings
1. **CarbonCast captures ~79-87% of oracle savings** (loses 1.3-3.2 pp). As total carbon: forecast
   emits **~3% more than oracle** -- matches EuroSys'24 (14% MAPE -> ~3% carbon increase). Validates
   the model against the literature.
2. **Forecasting SUSPENDS LESS, not more (K drops).** CarbonCast smooths the CI signal, so the
   predicted-cleanest hours cluster MORE than the noisy actual-cleanest -> schedule fragments less ->
   fewer dump/restore cycles -> LOWER mechanism overhead. Opposite of a reactive threshold policy that
   chases forecast spikes (the regime the literature flags). For optimal deadline-budget selection,
   misprediction's cost is almost entirely LOST SAVINGS (~3 pp), and it shrinks the mechanism term.
3. So with a real predictor: temporal scheduling still saves ~10-12% net; mechanism overhead is even
   smaller than the oracle case. The decision-flipping corner (low-power/SATA/long-job) persists but
   is governed by lost savings, not extra suspends.

## TODO
- Add the measured PREDICTION COST (CarbonCast inference energy+latency, per decision) to the
  forecast-aware total -- pending the inference-energy measurement on ford.
