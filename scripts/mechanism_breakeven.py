#!/usr/bin/env python3
"""Mechanism break-even: the minimum carbon prize that justifies the mechanism, per job/tier/leg.

Both legs reduce to ONE decision inequality. A shift (temporal suspend-in-place, or spatial migrate)
pays off only if the carbon PRIZE it unlocks exceeds the MEASURED mechanism carbon:

    prize  >=  N_cycles * E_mech * CI_mech                       (else the shift LOSES carbon)

Write the prize as a relative CI improvement f = dCI/CI on the job's compute energy E_compute = P*C:
    prize = f * CI * E_compute
Then, cancelling CI, the break-even is a DIMENSIONLESS, CI-independent, MEASURED constant:

    f >=  r*  =  N_cycles * E_mech / E_compute                   (the mechanism's "carbon tax rate")

  - Temporal: E_mech = dump + restore (local), N_cycles = K suspends (>=1 if you suspend at all).
  - Spatial : E_mech = dump + e_net*S + restore (adds the WAN leg), N_cycles = M migrations (~1).

r* is the fraction of clean-energy advantage the job must capture just to break even on ONE mechanism
cycle. A scheduler that counts the prize but ignores r* reports MISLEADING savings; the apparent-minus-
real gap is exactly r*. The rule: shift only if predicted f > N * r*.

  python scripts/mechanism_breakeven.py
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from carbon_spatial import WORKLOADS, E_NET_KJ_PER_GB


def main() -> None:
    print("r* = N_cycles * E_mech / E_compute  (break-even relative CI gain to justify ONE cycle).")
    print("Shift only if predicted prize f = dCI/CI exceeds r*.  E_compute = P*C.  CI-independent.\n")
    print(f"{'WL':3}{'C':>3}{'P_W':>6}{'S_GB':>6}{'E_comp_kJ':>10} | "
          f"{'TEMPORAL r* (1 suspend)':>24} | {'SPATIAL r* (1 migrate)':>24}")
    print(f"{'':3}{'':>3}{'':>6}{'':>6}{'':>10} | {'nvme':>11}{'sata':>13} | {'nvme':>11}{'sata':>13}")
    rows = []
    for wl, w in WORKLOADS.items():
        C, P, S = w["C"], w["P"], w["S"]
        Ecomp = P * C * 3600.0                                   # kJ
        net = E_NET_KJ_PER_GB * S
        r = {}
        for tier in ("nvme", "sata"):
            mech = w[f"dump_{tier}"] + w[f"rest_{tier}"]
            r[f"t_{tier}"] = mech / Ecomp                        # temporal, 1 suspend
            r[f"s_{tier}"] = (mech + net) / Ecomp                # spatial, 1 migrate
        rows.append((wl, w, Ecomp, r))
        print(f"{wl:3}{C:>3}{int(P*1000):>6}{S:>6}{Ecomp:>10.0f} | "
              f"{100*r['t_nvme']:>10.2f}%{100*r['t_sata']:>12.2f}% | "
              f"{100*r['s_nvme']:>10.2f}%{100*r['s_sata']:>12.2f}%")

    # Interpret r* as an absolute CI gap at a typical CI, and show the kill threshold.
    print("\nAbsolute break-even CI gap dCI* = r* * CI (gCO2/kWh) at CI=300; below this the mechanism LOSES:")
    print(f"{'WL':3} | {'temporal sata dCI*':>18} | {'spatial sata dCI*':>18}   (300 gCO2/kWh grid)")
    for wl, w, Ecomp, r in rows:
        print(f"{wl:3} | {300*r['t_sata']:>15.1f}    | {300*r['s_sata']:>15.1f}")

    print("\nReading: temporal r* is tiny (0.1-6.6%) -> a small diurnal swing pays for a suspend, EXCEPT")
    print("short/large jobs on SATA. spatial r* is LARGE (0.6-42%) because of e_net*S -> a job needs a big")
    print("in-zone CI spread; tight zones (EAST_ASIA ~8% spread) fall below r* for short/large-state jobs.")


if __name__ == "__main__":
    main()
