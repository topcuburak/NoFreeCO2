# A2 TP=4 — lossless KV-cache suspend (real vLLM V1), measured energy, both tiers

**Testbed:** ford (4× A100-40GB, EPYC 75F3, PM1733 RAID-0 NVMe + HPE SATA SSD). **Date:** 2026-06-18.
**Tool:** `workloads/a2_vllm/tp4_kv_suspend.py` (vLLM V1 port of the TCC `bench_storage.py` mechanism +
our telemetry). Real vLLM TP=4 serving; the KV cache is swapped HBM↔host via `collective_rpc` (copy
each `model_runner.kv_caches` tensor per rank) and stored host↔disk. **LOSSLESS** — preserves the KV
(unlike `sleep`, which discards it). 4-leg cycle, each MEASURED (GPU NVML×4 + CPU RAPL ∫P dt, modeled
DRAM/drive), KV pool **131.9 GB** (gpu-mem-util 0.9), warm floor (drop cold cycle-0 alloc).

## Per-leg cost, both tiers (131.9 GB KV)
| leg | NVMe lat / FULL | SATA lat / FULL |
|---|---|---|
| swap_out HBM→host | 1.36 s / 0.90 kJ | 1.43 s / 0.95 kJ |
| store host→disk | 33.1 s (4.0 GB/s) / 22.6 kJ | 279.7 s (0.47 GB/s) / 174.5 kJ |
| load disk→host | 11.6 s (11.3 GB/s) / 8.1 kJ | 257.6 s (0.51 GB/s) / 160.7 kJ |
| swap_in host→HBM | 1.35 s / 0.88 kJ | 1.35 s / 0.89 kJ |
| **round-trip** | **47.5 s / 32.5 kJ** (247 J/GB) | **540 s / 337 kJ** (2556 J/GB) |

NVMe 5 cycles (4 warm), SATA 4 cycles (3 warm). Raw: `data/timed_dump.jsonl` tags `a2_tp4_kv_nvme`,
`a2_tp4_kv_sata`. Energy is MEASURED (not derived) — telemetry wrapped each `collective_rpc` leg.

## Findings
1. **The KV swap (HBM↔host) is nearly free and tier-independent** — ~1.4 s, ~0.9 kJ each on both tiers.
   It runs on **4 independent PCIe Gen4 x16 links in parallel** → ~96 GB/s aggregate (24 GB/s/GPU, ~92%
   of per-GPU peak; pinned host buffers; verified vs PCIe + EPYC DDR4 ~200 GB/s host BW). The 4 GPUs
   never touch disk, so swap is identical NVMe vs SATA.
2. **The lossless suspend is storage-WRITE-bound** — store+load are **94% of the NVMe round-trip and
   99.5% of SATA**. Even on fast NVMe the GPU transfer is a rounding error; the disk write (4.0 GB/s,
   4 concurrent writers contending on one array, below the 5.8 single-stream) dominates.
3. **NVMe→SATA flip: 11.4× latency, 10.4× energy** (47.5→540 s, 32.5→337 kJ) — *larger* than A1 (8.8×)
   and A2-TP1 (8.6×) precisely because the GPU legs are so cheap here, so storage is an even bigger share.
4. **SATA energy is GPU idle-hold, not the drive** — store's 174.5 kJ is GPU 124.5 kJ (4× A100 × 449 W ×
   284 s idling while the slow disk grinds) + CPU 40 kJ; the modeled SATA drive is 851 J. Same regime as
   A1 SATA: a slow tier costs the accelerator sitting reserved-but-idle.

## Mechanism-design point: parallel app-swap ≫ serial cuda-checkpoint
vLLM's KV swap copies across the 4 GPUs **concurrently** → ~96 GB/s. cuda-checkpoint's `lock-all →
checkpoint-all` (A1/S1) evicts the processes **sequentially** → ~5 GB/s aggregate. So the same HBM↔host
move is **~20× faster** as a parallel app-level copy than as a serial transparent checkpoint. *How* you
move HBM→host matters as much as how much.

## Where this sits in the A2 picture (three suspend mechanisms)
- **TP=1 transparent** (cuda-checkpoint, full footprint incl KV): 38 GB, 20 s / 5.25 kJ NVMe — lossless,
  single-process. [`vllm_dump_cycle_tp1.md`]
- **TP=4 lossless** (this, KV-swap + disk): 131.9 GB, 47.5 s / 32.5 kJ NVMe — preserves KV, storage-bound.
- **TP=4 lossy** (vLLM `sleep(1)`): weights-only (~16 GB offloaded, KV discarded), ~3 s sleep / 1.6 s wake
  — ~10× cheaper but drops in-flight requests. The realistic carbon-shift path *if you drain first*.

→ At TP>1 serving you choose: pay the full lossless storage cost (this), or take the cheap lossy `sleep`
that sacrifices in-flight KV. The transparent in-place path (cuda-checkpoint of live NCCL) is infeasible
without the destroy/reinit dance, which fights vLLM's engine (the `_PP` assertion) — A1 proved that path
on training where we own the loop; vLLM serving uses swap or sleep instead.

## Reproduce
```
sudo -E python workloads/a2_vllm/tp4_kv_suspend.py --tp 4 --gpu-mem-util 0.9 \
  --max-model-len 8192 --prefill-isl 8000 --cycles 5 --store-out /var/data --tag a2_tp4_kv_nvme
#   SATA: --store-out /home/test --tag a2_tp4_kv_sata
```
