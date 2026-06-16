# A1 — Llama-3-8B FSDP training: temporal checkpoint (real-world)

**Testbed:** ford (4× A100-40GB, NVLink, driver 590.48, NCCL 2.28.9). **Date:** 2026-06-16.
**Workload:** Llama-3.1-8B, FSDP FULL_SHARD, bf16 mixed precision, AdamW, batch 1 × seq 512.
**Scripts:** `workloads/a1_fsdp/fsdp_train.py` (in-place attempt), `fsdp_checkpoint.py` (DCP).

## Two mechanisms — and why A1 differs from serving (S1/A2)

### Mechanism 1 — in-place transparent (cuda-checkpoint): WORKS with a process-group rebind
Ran real FSDP training, then `pause → dist.destroy_process_group() → [cuda-checkpoint suspend/resume]
→ init_process_group(fresh store) → rebind FSDP → CONTINUE`.

**First attempt FAILED to resume** on the first FSDP forward after reinit:
```
_all_gather_flat_param → all_gather_into_tensor → DistBackendError: NCCL communicator was aborted on rank N
```
**Cause:** FSDP caches the process-group/communicator handle at wrap time (per wrapped layer AND per
flat-param handle); `destroy_process_group` aborts it, and re-init (new default PG) does NOT update
FSDP's cached reference → aborted comm. Note this is a *cached-handle* problem, independent of WHERE
we suspend — we already suspend at a clean step boundary (barrier + synchronize, no collective in flight),
so coarser granularity does not help.

**FIX (validated 2026-06-16):** after reinit, walk `FSDP.fsdp_modules(model)` and re-point every
cached `process_group` (on the module, its `_fsdp_state`, and each flat-param handle) to the fresh
`dist.group.WORLD` (`rebind_fsdp_pg` in `fsdp_train.py`). Re-pointed 33 modules + 33 handles; reinit
1.02 s. **Training then continues correctly** — loss kept descending across the suspend boundary
(12.56 → 12.25 → 12.21 → 12.18 → 12.08 over steps 6–11), proving weights AND AdamW momentum survived.
So **in-place transparent suspend/resume IS feasible for FSDP training**, same family as vLLM serving
(destroy/reinit communicators), once FSDP's cached PG refs are rebound.

(So far validated WITHOUT cuda-checkpoint in the loop — i.e. the NCCL teardown/rebuild + FSDP rebind
path. Next: slot the cuda-checkpoint suspend/store/load/resume into the held window to (a) prove it
still resumes after HBM is evicted and restored and (b) measure the transparent-dump cost end-to-end.)

### Mechanism 2 — app-native state-dict checkpoint (DCP): the REAL mechanism
Training jobs suspend via `torch.distributed.checkpoint` (DCP): gather sharded {model, optimizer}
state → disk, reload into a fresh process. Measured (NVMe `/var/data`):

| op | latency | size | rate |
|---|---|---|---|
| **SAVE** (gather + serialize + write) | **18.78 s** | **48.2 GB** | 2.57 GB/s |
| **LOAD** (read + deserialize + set_state_dict) | **17.54 s** | 48.2 GB | 2.75 GB/s |
| round-trip | **~36 s** | | |

- **Footprint 48.2 GB** = model (bf16 16 GB) + **AdamW m,v (bf16 2×16 GB)**. (fp32 master/optimizer
  would ~2×.) A *compute-bound* training job still carries a 48 GB checkpoint — the memory-per-compute point.
- **DCP is much slower than raw disk** (the framework checkpoint tax): save 2.57 GB/s vs raw NVMe
  write 5.8 (**2.3×**); load 2.75 vs raw read 12.7 (**4.6×**) — gather/serialize/deserialize overhead
  on top of the disk I/O. (SATA tier: TODO — expect save ~100 s+ at 0.48 GB/s.)

## FINAL — 10-cycle steady-state, transparent dump (2026-06-16, NVMe, tag `a1_nvme_10cyc`)
Real Alpaca fine-tune, 3000 steps, **10 in-place suspend/dump/resume cycles** at steps 300…3000,
driven by `scripts/multi_suspend_driver.py` (marker handshake). 10/10 cycles dumped+resumed, all
`rebind_fsdp_pg` succeeded, **loss continuous across every suspend** (0.9–1.2 band, no reset). This
supersedes the single-shot numbers below, which were contaminated (see note).

Variance **<0.3%** across 10 cycles. Energy = MEASURED GPU board (NVML, incl HBM) + CPU pkg (RAPL),
plus MODELED DRAM (0.3 W/GB resident × t) + drive (NVMe 50 W on store/load) — kept separate:

| leg | latency | rate | measured GPU+CPU | +DRAM | +drive | **FULL** |
|---|---|---|---|---|---|---|
| suspend HBM→host | 27.07 ± 0.08 s | 5.47 GB/s | 12.96 kJ | 1.22 | 0 | **14.18 kJ** |
| store host→NVMe | 23.92 ± 0.10 s | 6.19 GB/s | 9.21 kJ | 1.08 | 1.20 | **11.48 kJ** |
| load NVMe→host | 11.78 ± 0.02 s | 12.58 GB/s | 4.48 kJ | 0.53 | 0.59 | **5.60 kJ** |
| resume host→HBM | 8.96 ± 0.02 s | 16.54 GB/s | 4.35 kJ | 0.40 | 0 | **4.75 kJ** |
| **round-trip** | **71.74 ± 0.09 s** | | **30.99 kJ** | 3.23 | 1.79 | **36.02 ± 0.08 kJ** |

- **Footprint 148.2 ± 0.0 GB** freed (37 GB/GPU). Measured split: GPU board ≈ 2/3, CPU pkg ≈ 1/3 per leg.
  Modeled DRAM+drive add ~16% on top of measured (DRAM the larger; drive only on store/load).
- **No cost drift:** round-trip FULL 35.9–36.1 kJ across all 10 rounds (steps 300→3000) — the
  allocator high-water is set early and stable, so the transparent dump cost does not grow with training.
- **Energy/byte (FULL, round-trip):** 36.02 kJ / 148.2 GB ≈ **243 J/GB** (measured-only 209 J/GB).
- **Methodology fixes that made it clean:** `drop_caches()` before resume (killed the page-cache
  pressure that paged out the cuda staging buffer → the 84 s resume outlier); `--verbose-cc` gate
  (moved `get-state` subprocesses out of the timed op → un-inflated suspend). Both in commit `dd83199`.

> NOTE: the single-shot numbers in the next section (114 s / 49 kJ) are **superseded/contaminated** —
> 84 s resume was page-cache pressure, ~55 s suspend was in-loop get-state overhead. Use the 10-cycle
> steady-state above (71.7 s / 36.0 kJ). The 2.65–3.1× transparency-tax-in-bytes finding still holds
> (150 GB transparent vs 48 GB DCP logical).

## Transparent dump cost — MEASURED (2026-06-16, NVMe `/var/data`, 4× A100) — SUPERSEDED (see above)
Held the 4 FSDP ranks (PG destroyed), then ran `timed_dump_experiment.py --multiproc --skip-criu
--store --store-out /var/data --tag a1_fsdp_nvme`: suspend(HBM→host) → store(host→NVMe) → load → resume.

**Footprint dumped = 127.7 GB** (gpu_freed 125.5 GB) — the *entire* HBM image across 4 GPUs, i.e.
**2.65× the 48.2 GB DCP logical state**. Steady allocated was only 12.3 GB/GPU, but ~31 GB/GPU was
resident (caching-allocator high-water from the backward activation peak + CUDA context + NCCL
buffers) and cuda-checkpoint dumps all of it. **This is the transparency tax in bytes: transparent
checkpoint pays for the allocator's reserved pool, not the logical tensors.**

| leg | latency | rate | energy (abs ∫P dt) | J/GB |
|---|---|---|---|---|
| suspend HBM→host | 23.60 s | 5.41 GB/s | 10.88 kJ | 85.2 |
| store host→NVMe | 20.36 s | 6.27 GB/s | 7.58 kJ | 59.3 |
| load NVMe→host | 9.98 s | 12.80 GB/s | 3.68 kJ | 28.9 |
| resume host→HBM | 7.84 s | 16.29 GB/s | 3.73 kJ | 29.2 |
| **round-trip** | **61.8 s** | | **25.87 kJ** | **202.6** |

- Absolute energy includes all 4 GPUs idling at ~76 W each through the full 62 s (that idle-hold is
  real cost — why suspend's 85 J/GB exceeds the single-GPU HBM→host component fit of 41 J/GB).
- **Transparent vs DCP:** 2.65× the bytes (127.7 vs 48.2 GB), 1.7× the wall-clock (61.8 vs 36.3 s),
  but resumes the SAME process in place (no re-import/re-load/re-init), whereas DCP requires a full
  restart whose cost is NOT in the 36.3 s. The two mechanisms trade image size + in-place resume.
- Rows tagged `tag=a1_fsdp_nvme` in `data/timed_dump.jsonl`. SATA tier: TODO (`--store-out /home/test`).

### Random-data footprint generator vs REAL fine-tune (both transparent, NVMe)
The real Alpaca fine-tune (`fsdp_finetune.py`, 100 steps, fp32 reduce + grad-accum 4 + grad clip) dumps
**MORE** and costs **super-linearly** more than the random 6-step run, though steady alloc is identical (12.2 GB):

| | random (fsdp_train, 6 steps) | real fine-tune (Alpaca, 100 steps) |
|---|---|---|
| footprint | 127.7 GB | **150.3 GB** (37 GB/GPU, near 40 GB cap) |
| suspend HBM→host | 23.60 s / 5.41 GB/s / 10.88 kJ | **54.93 s / 2.74 GB/s / 24.93 kJ** |
| store host→NVMe | 20.36 s / 7.58 kJ | 25.22 s / 9.63 kJ |
| load NVMe→host | 9.98 s / 3.68 kJ | 11.75 s / 4.43 kJ |
| resume host→HBM | 7.84 s / 16.29 GB/s / 3.73 kJ | **22.66 s / 6.63 GB/s / 10.12 kJ** |
| round-trip | 61.8 s / 25.9 kJ | **114.6 s / 49.1 kJ** |
| J/GB (round-trip, abs) | 202.6 | **326.8** |

**Footprint +18% but round-trip +85% and energy +90%.** Disk legs scaled with bytes; the GPU-side
cuda-checkpoint legs (suspend/resume) got ~2× slower PER GB. Driver: real training fragments device
memory (fp32 grad-comm buffers, grad-accum, clip temporaries) into more/smaller allocations, and
cuda-checkpoint walks every allocation -> lower effective GB/s. This is the **allocation-structure
effect** (cf. the `--chunks` probe): transparent-checkpoint cost depends on allocation COUNT/layout,
not just total bytes. So the transparency tax for REAL training is 3.1× the DCP logical state in bytes
(150.3 vs 48.2 GB) AND a degraded per-GB rate. Real-finetune rows tagged `a1_finetune_nvme`.

## Real fine-tune, loss-continuity proof (2026-06-16, Alpaca instruction tuning)
`fsdp_finetune.py`: genuine instruction tuning (Alpaca, prompt masked, AdamW + cosine LR + grad-accum 4,
lr 2e-5, batch 1 × seq 512). Loss descends for real (1.35 @ step10 → ~1.0 @ step100), then suspended
in place at step 100 (destroy PG → full HBM dump to NVMe → hold → restore → reinit → rebind):

| step | 90 | 100 (suspend) | 110 (resumed) | 120 |
|---|---|---|---|---|
| loss | 1.0152 | 1.0646 | 1.0906 | 1.0204 |

**Loss is continuous across the dump** — post-resume values stay in the same noise band, no jump/reset.
Optimizer momentum, LR schedule, and dataloader iterator all survive because cuda-checkpoint evicts only
GPU state (we `--skip-criu`) so the Python process stays alive through the hold. This is the correctness
proof for a REAL fine-tune surviving a full 4-GPU HBM dump-to-disk-and-back. Footprint is set by shape
(batch×seq), not by the data: steady `gpu_alloc` 12.2 GB matches the random-data run's 12.3 GB.
Caveat: the logged `tok/s` is cumulative, so it drops after resume purely because the ~70 s held during
the dump averages in as zero-token time — NOT a throughput regression (per-step work unchanged).
Tagged `a1_finetune_nvme` in `data/timed_dump.jsonl`.

## Headline finding: the mechanism depends on the WORKLOAD TYPE
| workload type | temporal mechanism | resumes in place? |
|---|---|---|
| **serving** (TP=1, and TP>1 with destroy/reinit) | transparent cuda-checkpoint (HBM→host) | ✅ yes |
| **training** (FSDP), transparent | cuda-checkpoint (HBM→host) + destroy/reinit + **FSDP PG rebind** | ✅ yes (validated) |
| **training** (FSDP), app-native | state-dict (DCP) save + RESTART | ✅ yes (fresh process) |

→ A1 has **two** valid temporal mechanisms. (1) **Transparent in-place** — same as serving: evict the
full HBM image, hold, restore, continue the same process; works once FSDP's cached PG is rebound. Cost
is the full per-GPU HBM footprint (suspend/store/load/resume), comparable per-leg to A2. (2) **App-native
DCP** — a single gather+write of logical state (48.2 GB: weights + AdamW m,v), storage-bound, 2–5×
framework tax over raw disk, requires a process restart. Transparent dumps the whole image; DCP dumps
only the numbers needed to keep training.

## Caveats / TODO
- DCP run captured **latency + size only** (no telemetry wrapper around torchrun) → energy not
  measured; derive from storage coeffs (CPU floor + GPU idle×4 + drive over the save/load time) or
  add a RAPL/NVML sampler.
- SATA tier run pending (`--ckpt-dir /home/test/a1_ckpt`).

## Reproduce
```
sudo install -d -o test -g test /var/data/a1_ckpt
torchrun --nproc_per_node=4 workloads/a1_fsdp/fsdp_checkpoint.py \
  --model meta-llama/Llama-3.1-8B --batch 1 --seq-len 512 --warmup 4 --ckpt-dir /var/data/a1_ckpt
# in-place attempt (shows the FSDP-resume failure):
torchrun --nproc_per_node=4 workloads/a1_fsdp/fsdp_train.py --suspend-step 6 ...
```
