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
