# Spatial migration (server A → B over a network) — latency & energy overhead

**Status: ANALYTICAL model** (no second node / BMC on the testbed). Endpoint host/GPU
constants are *measured on ford*; network constants are *assumed (datasheet)* or *from
literature*. Latency is a physical transfer time; energy depends on the **system
boundary** chosen. Companion to the measured temporal cost in `vllm_dump_cycle_tp1.md`.

## 1. Decomposition

Spatial migration = the temporal mechanism (suspend/resume + GPU term) **plus** moving
the dumped state across the network, with a **second machine now powered**:

```
L_spatial = H/BW_sus  +  S/BW_net  +  H/BW_res          (suspend + transfer + resume)
E_spatial = E_suspend  +  E_net(S, boundary)  +  E_resume
   H = min(S, C_HBM)   (HBM-resident portion; rest is host-DRAM data)
```

Two architectures: **direct-stream** (HBM→host→NIC→net→NIC→host→HBM; no disk) or
**stage-and-forward** (adds local store+load disk legs on both ends — on SAS this
re-creates the slow-tier penalty, itself a finding).

## 2. Coefficients

| symbol | value | source |
|---|---|---|
| `C_HBM` | 40 GB | A100 (measured cap) |
| `BW_sus` (HBM→host) | 4.6 GB/s | **measured (ford)** |
| `BW_res` (host→HBM) | 12 GB/s | **measured (ford)** |
| `E_suspend` (40 GB, host+GPU `∫P dt`) | 2.42 kJ | **measured** (1.21 host + 1.21 GPU) |
| `E_resume` (host+GPU) | 0.70 kJ | **measured** (0.44 host + 0.25 GPU) |
| `P_host` (per node, pkg floor) | 130 W | **measured (RAPL)** |
| `BW_net` | parameter | link spec / `iperf3` |
| `P_net` (NIC pair, 1 GbE) | ~2 W [1 W/port ×2] | **assumed (datasheet ballpark)** |
| `e_net` (full-path, marginal) | ~1.4 kJ/GB | **literature** (see §5) |

## 3. The two network-energy boundaries (THE choice)

The transfer **energy** differs by ~4 orders of magnitude depending on what you count.
**Latency is identical under both** (`t = S/BW_net`).

- **Boundary 1 — endpoints only** (our NICs): `E1 = P_net · S/BW_net`.
  **Bandwidth-DEPENDENT** (fixed NIC power × transfer time → slower link = more energy).
- **Boundary 2 — full internet path, marginal**: `E2 = e_net · S`, `e_net ≈ 1.4 kJ/GB`.
  **Bandwidth-INDEPENDENT** (per-byte property of the network).

## 4. Impact — 200 GB state, 1-hour run baseline (1.332 MJ @ 370 W / 3600 s)

### A. One migration, across link speeds

| link | BW_eff | latency | lat % | E1 (endpoints) | E1 % | E2 (full-path) | E2 % |
|---|---|---|---|---|---|---|---|
| 1000 Mbps | 0.125 GB/s | 27 min | 44% | 3.2 kJ | 0.24% | 280 kJ | **21%** |
| 400 Mbps | 0.050 GB/s | 67 min | 111% | 8.0 kJ | 0.60% | 280 kJ | **21%** |
| 200 Mbps | 0.025 GB/s | 133 min | 222% | 16.0 kJ | 1.20% | 280 kJ | **21%** |

E1 rises as the link slows (fixed NIC power held longer); E2 is flat (per-byte).

### B. N migrations at 400 Mbps

| N | latency | lat % | E1 | E1 % | E2 | E2 % |
|---|---|---|---|---|---|---|
| 1 | 67 min | 111% | 8 kJ | 0.6% | 280 kJ | 21% |
| 2 | 133 min | 222% | 16 kJ | 1.2% | 560 kJ | 42% |
| 3 | 200 min | 333% | 24 kJ | 1.8% | 840 kJ | 63% |
| 4 | 267 min | 444% | 32 kJ | 2.4% | 1120 kJ | 84% |
| 5 | 333 min | 556% | 40 kJ | 3.0% | 1400 kJ | **105%** |

## 5. Findings

1. **Latency is brutal and boundary-independent.** A *single* 200 GB migration over
   commodity internet (400 Mbps) stalls the workload **111% of the compute run** —
   longer than the work itself. 5 moves = 5.5× the run in wall-clock.
2. **Energy diverges by boundary.** Endpoints-only → **≤3%** (negligible; latency is the
   whole story). Full-path marginal → **21% per migration**, **>100% by 5 moves**
   (network energy alone exceeds the run).
3. **Both E2 and latency are LINEAR in `S`** → the app-aware KV-only dump (shrink
   200 GB → ~10 GB) cuts full-path energy 21% → ~1% **and** the stall 67 min → ~3 min.
   The transparency tax shows up identically on the energy and latency axes.
4. WAN migration of *large* state is viable only if (a) `S` is small (app-aware dump),
   (b) the fabric is fast (intra-DC ≥100 GbE), or (c) the run amortizes over a long
   remote residency. Commodity-internet + full-footprint = non-starter.

## 6. Literature for `e_net` (network energy intensity)

- **Aslan, Mayers, Koomey & France (2018)**, *J. Ind. Ecol.* — meta-analysis anchor:
  **0.06 kWh/GB** fixed-line transmission in **2015**, **halving ~every 2 years**.
  https://onlinelibrary.wiley.com/doi/10.1111/jiec.12630
- **Guennebaud & Bugeau (2024)**, *J. Ind. Ecol.* — kWh/GB is an **average**; network
  power is largely **fixed vs throughput**, so the **marginal** energy of an extra GB is
  much lower → don't use the average for attributional/migration decisions.
  https://hal.science/hal-04631084v1/document
- **Baliga et al. (2009/2011)** — bottom-up model, **~0.0064 kWh/GB** (≈ marginal-ish).
- Conversions: **1 kWh/GB = 3600 J/GB = 0.45 µJ/bit.** Projected 2025 avg ~0.002 kWh/GB
  (7.2 kJ/GB); marginal ~⅕ of avg → **~1.4 kJ/GB** (the `e_net` used above). Sensitivity
  band **0.002–0.02 kWh/GB**. Always label "marginal, fixed-network."
- **Latency**: WAN RTT ~30 ms–>1 s, typically tens–low-hundreds ms cross-region
  (physics bound ~130 ms antipodal); affects only TCP ramp-up — add as fixed setup term.

## 7. Recommended reporting

Report **both boundaries as bounds**: endpoints-only = "the energy we directly pay"
(latency-dominated story); full-path marginal = "the carbon-attributable energy"
(GHG Scope-3 honest view, energy + latency both bind). State the boundary explicitly —
a reviewer will ask which one and whether the endpoint nodes would be idle anyway
(same committed-vs-marginal question as the GPU-reservation choice).
