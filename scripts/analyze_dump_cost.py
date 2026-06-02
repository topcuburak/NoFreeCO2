#!/usr/bin/env python3
"""Full transparent-dump cost = the C1/C2 model instantiated with MEASURED coefficients.

  E_dump(S, tier) = E_extract(S)            # HBM -> host  (cuda-checkpoint)
                  + E_store(S, tier)        # host -> NVMe (per storage tier)

Each leg's energy is dominated by the TIME TERM (node power x latency), per the
measured finding that the absolute/time-term energy >> the marginal move energy
and is far more stable. Transparency tax = transparent(full footprint) vs
app-aware(live-KV only, no weights moved).

Pure analysis (no hardware) -- run anywhere:
    python scripts/analyze_dump_cost.py --footprint 42 --kv 16
    python scripts/analyze_dump_cost.py --footprint 70 --kv 30 --tiers nvme_raid0,sas_ssd

All coefficients below are MEASURED on ford (2026-06-02) unless marked MODELED.
"""
from __future__ import annotations

import argparse

# --- HBM -> host extract leg: cuda-checkpoint on real vLLM TP=1 ----------------
# 42.1 GB freed in 8.43 s -> ~5.0 GB/s; GPU DMA marginal ~75 J over the move.
BW_EXTRACT_GBPS = 5.0
E_EXTRACT_MARGINAL_J_PER_GB = 75.0 / 42.0

# --- host -> NVMe store leg: characterize_storage.py O_DIRECT, cpu_abs / 16 GB --
TIERS = {
    "nvme_raid0": {"write_gbps": 6.2, "read_gbps": 12.7, "cpu_abs_j_per_gb": 342.0 / 16.0},
    "sas_ssd":    {"write_gbps": 0.48, "read_gbps": 0.53, "cpu_abs_j_per_gb": 4186.0 / 16.0},
    # single PM1733 not separately mountable -> MODELED as raid0 / 2:
    "nvme_single": {"write_gbps": 3.1, "read_gbps": 6.35, "cpu_abs_j_per_gb": 342.0 / 16.0},
}

# --- node idle power (MEASURED CPU+GPU; rest MODELED) --------------------------
P_CPU_IDLE_W = 121.0     # RAPL package idle
P_GPU_IDLE_W = 222.0     # 4x A100 idle (NVML, summed)
P_REST_W = 150.0         # MODELED: NVMe/NIC/fans/PSU loss (no IPMI on ford)
P_NODE_IDLE_W = P_CPU_IDLE_W + P_GPU_IDLE_W + P_REST_W


def extract_cost(s_gb: float) -> tuple[float, float]:
    """HBM->host: node drawn for the move + GPU DMA marginal."""
    t = s_gb / BW_EXTRACT_GBPS
    e = P_NODE_IDLE_W * t + E_EXTRACT_MARGINAL_J_PER_GB * s_gb
    return t, e


def store_cost(s_gb: float, tier: str) -> tuple[float, float]:
    """host->NVMe: cpu_abs already includes CPU idle x t; add GPU+rest idle x t."""
    ti = TIERS[tier]
    t = s_gb / ti["write_gbps"]
    e = ti["cpu_abs_j_per_gb"] * s_gb + (P_GPU_IDLE_W + P_REST_W) * t
    return t, e


def dump_cost(s_gb: float, tier: str) -> dict:
    te, ee = extract_cost(s_gb)
    ts, es = store_cost(s_gb, tier)
    return {"extract_s": te, "store_s": ts, "total_s": te + ts,
            "extract_j": ee, "store_j": es, "total_j": ee + es}


def main() -> None:
    ap = argparse.ArgumentParser(description="transparent-dump cost from measured coefficients")
    ap.add_argument("--footprint", type=float, default=42.0,
                    help="transparent (CRIU/cuda-checkpoint) full GPU footprint, GB")
    ap.add_argument("--kv", type=float, default=16.0,
                    help="app-aware live KV moved, GB (no weights) -- for the tax")
    ap.add_argument("--tiers", default="nvme_raid0,nvme_single,sas_ssd")
    args = ap.parse_args()
    tiers = [t for t in args.tiers.split(",") if t in TIERS]

    print(f"=== TRANSPARENT DUMP COST  (footprint S = {args.footprint:.0f} GB) ===")
    print(f"{'tier':13}{'extract_s':>11}{'store_s':>9}{'total_s':>9}"
          f"{'extract_J':>11}{'store_J':>10}{'total_J':>10}")
    transp = {}
    for t in tiers:
        c = dump_cost(args.footprint, t)
        transp[t] = c
        print(f"{t:13}{c['extract_s']:11.1f}{c['store_s']:9.1f}{c['total_s']:9.1f}"
              f"{c['extract_j']:11.0f}{c['store_j']:10.0f}{c['total_j']:10.0f}")

    print(f"\n=== TRANSPARENCY TAX  (transparent {args.footprint:.0f} GB vs "
          f"app-aware KV-only {args.kv:.0f} GB) ===")
    print(f"{'tier':13}{'transp_J':>11}{'appaware_J':>12}{'tax_x':>8}"
          f"{'transp_s':>10}{'appaware_s':>12}{'tax_x':>8}")
    for t in tiers:
        a = dump_cost(args.kv, t)         # app-aware: only live KV, no weights
        tr = transp[t]
        tax_e = tr["total_j"] / a["total_j"] if a["total_j"] else 0
        tax_t = tr["total_s"] / a["total_s"] if a["total_s"] else 0
        print(f"{t:13}{tr['total_j']:11.0f}{a['total_j']:12.0f}{tax_e:8.2f}"
              f"{tr['total_s']:10.1f}{a['total_s']:12.1f}{tax_t:8.2f}")

    print("\nMEASURED: extract BW (cuda-checkpoint), store BW + cpu_abs (characterize_storage),"
          " CPU/GPU idle.  MODELED: P_rest (no IPMI), nvme_single (=raid0/2), DRAM energy.")


if __name__ == "__main__":
    main()
