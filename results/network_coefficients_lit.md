# Network leg coefficients — AWS inter-region latency + transmission energy (literature)

Deep-research outputs (adversarially verified) for the **spatial/WAN migration** leg, which
is analytical (no second node on ford). Feeds `wan_migration_overhead.md`. Two parts:
inter-region **bandwidth/latency** (per-byte transfer time) and **energy per byte**.

## A. AWS inter-region throughput → per-byte transfer time (s/GB)
**Mechanism:** single-stream TCP is capped by the bandwidth-delay product AND a hard **5 Gbps
per-flow** ceiling (cross-region; same-AZ boosts like ENA Express 25G / cluster-PG 10G do NOT
apply). To exceed it you must parallelize (S3 has no connection limit; byte-range / multipart).

**RTTs (drive single-stream cost; high confidence, cloudping.co):**
us-east-1↔us-west-2 ~62 ms · us-east-1↔eu-west-1 ~69 ms · us-west-2↔ap-northeast-1 ~99 ms ·
us-east-1↔ap-southeast-2 (Sydney) ~200 ms. Fiber RTT floor = dist_km×2÷203,939×1000 ms; measured ~1.3–2.2× floor.

**Per-byte transfer time:**
| mode | throughput | per-byte |
|---|---|---|
| single tuned stream (RTT-scaled, ~8 MB window) | ~40–130 MB/s | **~8 s/GB (62 ms) → ~25 s/GB (Sydney)** |
| naive/un-tuned single stream | window-limited | ~20–1000 s/GB |
| parallelized (multipart/multi-stream) | multi-Gbps | **~0.2–1 s/GB** (≈distance-independent) |

→ Our **400/200 Mbps band (20–40 s/GB)** ≈ a single tuned stream (conservative). Engineered
parallel transfer ≈ ~0.5–1 s/GB (optimistic). Report both as a bracket; single-stream scales with RTT.
**Refuted (do NOT use):** all s5cmd speedup multipliers (27–80×); the $0.02/$0.09 per-GB pricing claims.

## B. Network transmission energy per byte
**Per-device energy intensity (W/Gbps = J/Gbit), 2024** [Porter 2502.16631 T4; Van Heddeghem; ETH green-routing — verified 3-0]:
core router **2** (was 10 in 2012) · transceiver **0.09** (was 1.5) · WDM switch 0.05 · amplifier 0.03 · regenerator 3.
Route energy = Σ_devices IE×bits. `EIDT_marginal = (P_max−P_idle)/C_max`; `1 kWh/bit = 3600 kW/bps`.

**End-to-end intensity by SCOPE (the answer spans ~5 orders of magnitude):**
| scope | coefficient | for 10 GB |
|---|---|---|
| endpoint NICs only | ~2 W ×t | ~50–100 J |
| **inter-DC hyperscaler** (Cloud Carbon Footprint) | **0.001 kWh/GB = 3.6 kJ/GB** | **~36 kJ** |
| inter-DC route marginal (Porter CIDT 0.017 gCO₂/Gb) | ~0.0003 kWh/GB | ~10 kJ |
| public internet, recent (Baliga 0.0064 → proj. 0.002) | | ~72–230 kJ |
| public internet 2015 avg (Aslan 0.06 kWh/GB, halving ~2 yr) | | ~2.2 MJ |

**Average vs marginal (Guennebaud 2024):** kWh/GB averages include fixed/idle/redundant network power;
the **marginal** energy of one extra GB is much lower (switches far from power-proportional). Use
**marginal, fixed-network** for migration accounting; cite the average only for total-footprint.

**Structural:** network-fabric energy is **per-byte (time-independent)** — same whether 5 s or 10 s;
**endpoint** energy is **per-time** (node floor × duration), so faster transmission is cheaper on endpoints only.

**Recommended for the model (AWS-to-AWS):** `e_net ≈ 0.001 kWh/GB (3.6 kJ/GB)` point estimate,
band 0.0003–0.006, labeled marginal/inter-DC. Endpoint host floor = measured 140 W (RAPL).

## Sources
Aslan/Koomey 2018 (jiec.12630); Guennebaud&Bugeau 2024 (hal-04631084); Baliga 2009/2011; Cloud Carbon
Footprint methodology; Porter et al. 2025 "geo-shifted workloads" (arXiv 2504.14022) + CIDT; cloudping.co;
AWS S3 perf docs / EC2 bandwidth docs; CRIUgpu 2502.16631 (device intensities).
