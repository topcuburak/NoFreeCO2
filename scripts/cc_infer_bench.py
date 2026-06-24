#!/usr/bin/env python3
"""Measure CarbonCast inference cost: latency + energy per forecast.

Loads a pretrained CarbonCast Tier-2 CNN-LSTM (.h5), builds a representative (1, 24, F) input, and
runs model.predict in a tight loop for --seconds. Uses the RUNJOB_READY/trigger handshake so the
energy harness (job_energy.py) measures ONLY the inference loop (model load + TF import excluded).
Reports per-predict latency and the loop's iteration count; job_energy supplies the per-leg energy.

Run inside the carboncast conda env, measured from socc-bench's harness:
  sudo -E python scripts/job_energy.py --gpus none --dram-gb 0 --tag cc_infer -- \
    /home/test/miniconda3/envs/carboncast/bin/python scripts/cc_infer_bench.py \
    --model /home/test/NoFreeCO2/CarbonCast/saved_second_tier_models/direct/CISO.h5 --seconds 60
One CarbonCast "decision" = a 96h forecast = 4 sequential 24h predicts, so report per-decision = 4x.
"""
from __future__ import annotations

import argparse
import os
import time


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to a pretrained Tier-2 .h5")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--predicts-per-forecast", type=int, default=4, help="24h x4 = 96h decision")
    a = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")            # force CPU (clean RAPL measurement)
    import numpy as np
    from tensorflow import keras

    model = keras.models.load_model(a.model, compile=False)
    shp = model.input_shape                                      # (None, 24, F)
    F = shp[-1]; T = shp[-2] if len(shp) == 3 else 24
    x = np.random.rand(1, T, F).astype("float32")
    model.predict(x, verbose=0)                                  # warm up (graph build, threads)
    print(f"[cc-infer] model={os.path.basename(a.model)} input=(1,{T},{F}) -- warmed up", flush=True)

    print("RUNJOB_READY", flush=True)                            # setup (import+load+warmup) done
    trig = os.environ.get("RUNJOB_TRIGGER")
    while trig and not os.path.exists(trig):
        time.sleep(0.02)

    n = 0
    t0 = time.perf_counter()
    end = t0 + a.seconds
    while time.perf_counter() < end:
        model.predict(x, verbose=0)
        n += 1
    dt = time.perf_counter() - t0

    per = dt / n
    print(f"RUNJOB_DONE predicts={n}", flush=True)
    print(f"[cc-infer] {n} predicts in {dt:.1f}s -> {per*1000:.3f} ms/predict "
          f"({per*1000*a.predicts_per_forecast:.3f} ms/96h-forecast)", flush=True)


if __name__ == "__main__":
    main()
