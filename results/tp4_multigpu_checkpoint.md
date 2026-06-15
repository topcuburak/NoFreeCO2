# Multi-GPU (TP>1 / FSDP) checkpoint — infeasible transparently, FEASIBLE app-aware (measured)

**Testbed:** ford (4× A100-40GB, NVLink, driver 590.48, CUDA 13, NCCL 2.28.9). **Date:** 2026-06-16.
**Tools:** `scripts/nccl_holdtest.py` (minimal 4-GPU NCCL target), `timed_dump_experiment.py --multiproc`.
Supersedes the earlier "infeasibility" framing: transparent fails, but a **pause → destroy-NCCL →
checkpoint → reinit** path works, and we measured every term.

## 1. Transparent suspend of a LIVE NCCL job — INFEASIBLE (confirmed + corroborated)
cuda-checkpoint **lock** succeeds but **checkpoint** fails on a process holding live NCCL
communicators (cross-process CUDA-IPC / NVLink-P2P state it can't serialize):
```
cuda-checkpoint --action lock       --pid <worker>  -> lock OK (state: locked)
cuda-checkpoint --action checkpoint --pid <worker>  -> "OS call failed or operation not
                                                        supported on this OS" (CUDA_ERROR_OPERATING_SYSTEM)
```
Tried 3 ways, all fail: busy (NCCL all-reduce in flight → lock-all deadlock); idle/`--hold-idle`
(lock OK, checkpoint segfaults); `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1` (same error).
**Pausing/quiescing is NOT sufficient** — it fixes the in-flight deadlock but the *communicators
still exist*, and that is what cuda-checkpoint can't serialize.

Corroboration (lit search): NVIDIA cuda-checkpoint README ("does not support UVM or IPC memory");
CRIUgpu paper arXiv:2502.16631 ("the cuda-checkpoint tool does not support checkpoint/restore
operations with NCCL"); cuda-checkpoint issues #5 (2024) and #37 (Sep 2025) report our exact error;
NCCL roadmap (nccl#2090) lists preserving communicators across checkpoint as "Under Consideration."

## 2. The WORKING path: pause → destroy NCCL → checkpoint → resume → reinit
Destroying the process group (`dist.destroy_process_group()` → `ncclCommAbort`) releases the
IPC/communicator state, after which cuda-checkpoint **succeeds**. Measured on the 4-GPU target
(8 GiB/GPU, 37.1 GB total freed across the 4 GPUs):

| leg | latency | energy (GPU+CPU, ∫P dt) | notes |
|---|---|---|---|
| suspend (lock-all → checkpoint-all, 4 GPUs) | **7.66 s** (warm) | 3.64 kJ | first-of-session cold = 15.0 s (warmup outlier) |
| resume (restore-all → unlock-all) | **3.36 s** | 1.59 kJ | |
| **NCCL re-init** (fresh TCPStore) | **0.98 s** (all 4 ranks) | ~0 | fixed term, footprint-independent |
| **round-trip (HBM↔host + reinit)** | **~12 s** | **~5.2 kJ** | for 37 GB / 4 GPUs; + store/load if persisting |

Warm suspend ≈ 4.9 GB/s aggregate ≈ single-GPU rate × 4 (lock-all/checkpoint-all runs the ranks
**sequentially**, not contended).

## 3. Two non-obvious requirements (both findings)
1. **Destroy, don't just pause.** `--hold-idle` (quiesce) alone still fails; you must
   `destroy_process_group` to release the communicators before the checkpoint succeeds.
2. **cuda-checkpoint restores CUDA state, NOT sockets.** Naive re-init on torchrun's original
   TCPStore crashed: `socketPollConnect ... connection refused` (stale NCCL bootstrap addrs). The
   re-init MUST use a **fresh rendezvous** (new TCPStore/port). So re-init isn't free plumbing.

## 4. Implication
- **Multi-GPU temporal/spatial suspend is feasible but inherently APP-AWARE**: the app must
  quiesce at a collective-free point, `destroy_process_group`, then `init_process_group` (fresh
  rendezvous) on resume (vLLM RFC #34303 does this via `collective_rpc`). This is the transparency
  tax at TP>1 — now quantified: **+~1 s NCCL rebuild + a forced framework hook**, on top of the
  HBM↔host + storage legs.
- **A1 (FSDP) uses the identical path** → A1 temporal cost = multi-GPU HBM↔host + storage(70 GB) +
  ~1 s reinit.

## Reproduce
```
# T1: torchrun --nproc_per_node=4 scripts/nccl_holdtest.py --gb 8   (proves NCCL, destroys it, holds)
# T2 (root, after 4x 'NCCL DESTROYED'): timed_dump_experiment.py --multiproc --skip-criu --marks-min 0 --hold-seconds 8
# T2: touch /tmp/nccl_resume   -> ranks reinit NCCL (fresh store) and print the timing
```
Raw per-op records: `data/timed_dump.jsonl` (multiproc=true). Refs: CRIUgpu 2502.16631, vLLM RFC
vllm#34303, cuda-checkpoint #5/#37, NVIDIA cuda-checkpoint README, nccl#2090.
