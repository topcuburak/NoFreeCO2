# TP>1 transparent checkpoint — INFEASIBILITY finding (cuda-checkpoint + NCCL)

**Testbed:** ford (4× A100-40GB, NVLink, driver 590.48, CUDA 13). **Date:** 2026-06-08.
**Workload:** vLLM Llama-3.1-8B, `--tensor-parallel-size 4`, `--enforce-eager`, V1
multiproc executor. **Tool:** `scripts/timed_dump_experiment.py --multiproc` +
`scripts/transparent_dump.py` (lock-all → checkpoint-all → restore-all → unlock-all).

## Result: transparent cuda-checkpoint of a live TP>1 NCCL job does NOT work on this stack.

The cuda-checkpoint **lock** phase succeeds; the **checkpoint** phase fails:

```
$ sudo cuda-checkpoint --action lock       --pid <worker>   ->  lock OK  (state: locked)
$ sudo cuda-checkpoint --action checkpoint --pid <worker>
  Could not checkpoint on process ID <worker>:
    "OS call failed or operation not supported on this OS"      # CUDA_ERROR_OPERATING_SYSTEM (304)
```

TP=1 (single process, no NCCL) checkpoints fine (see `vllm_dump_cycle_tp1.md`); any
TP>1 fails. The differentiator is **cross-process CUDA-IPC shared GPU memory**, held by
(a) NCCL for intra-node TP all-reduce and (b) vLLM's V1 multiproc executor for
tensor/IPC passing. cuda-checkpoint cannot serialize a context with live IPC handles.

## Three distinct failure modes, all pointing the same way

| # | condition | symptom | cause |
|---|---|---|---|
| 1 | **busy** (generation running) | controller hangs at lock-all; `0%/100%/100%/100%` GPU util; rank0 `locked`, others `running` | in-flight NCCL all-reduce: locking rank 0 strands the peers mid-collective (barrier deadlock) |
| 2 | **idle** (`--hold-idle`, no collective in flight) | lock-all OK, then `checkpoint` errors + rank0 **segfaults**; HeartbeatMonitor cascades; `Executor failed` | checkpoint can't serialize the live NCCL/IPC GPU state — *existence* of communicators, not in-flight comm |
| 3 | **idle + `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 TORCH_NCCL_ENABLE_MONITORING=0`** | identical to #2: `CUDA_ERROR_OPERATING_SYSTEM` | vLLM's multiproc executor still uses CUDA-IPC; transport flags don't remove IPC memory |

Mitigations tried that did NOT fix it: idle-quiesce (fixes #1 only), disabling NCCL
P2P + SHM, disabling the PyTorch NCCL heartbeat monitor.

## What WOULD be required (and why it's the transparency tax)

To checkpoint, the process must hold **no IPC-shared device memory** — i.e. **tear down
the process group** (`dist.destroy_process_group()` / `ncclCommDestroy`) before the
dump and **rebuild it on resume**. vLLM exposes no "pause → destroy comms → hold →
rebuild" hook, so this is **app-level surgery**, not transparent.

→ **Finding for the paper:** at TP>1 the transparency tax is not merely *higher cost* —
the transparent mechanism is **infeasible**. Suspending a multi-GPU served model
*requires* application cooperation (quiesce to a collective-free point AND tear down /
rebuild the NCCL communicators). This is the sharpest form of the thesis: app-awareness
isn't an optimization for multi-GPU suspend, it's a precondition.

## TP=4 cost — MODELED from the measured TP=1 coefficients (since we can't measure it)

Each rank holds an independent per-GPU footprint `S_gpu` (model shard + KV-pool shard);
total dump = 4 × `S_gpu`. Using the TP=1 affine fit (`vllm_dump_cycle_tp1.md`):

- **dump  ≈ 4 × (240 J + 44 J/GB·S_gpu)** ; **restore ≈ 4 × (155 J + 17 J/GB·S_gpu)**
- store/load = 4 × `S_gpu` bytes at the tier rate (NVMe 5.8/12 GB/s, SAS 0.48/0.53).
- **Caveat — suspend-bandwidth contention:** the 4 ranks stage HBM→host concurrently
  over the shared PCIe/host-DRAM path, so aggregate suspend bandwidth likely < 4× the
  TP=1 ~4.6 GB/s (sub-linear); the per-rank cost is a lower bound. Host-DRAM staging
  also needs ≥ 4 × `S_gpu` free RAM (≈144 GB at util 0.9).

This is a projection; the *measured* contribution of this experiment is the
**infeasibility result above**, not a cost number.

## Reproduce (the failure)
```
# terminal 1: TP=4 idle hold
python workloads/a2_vllm/serve.py --model meta-llama/Llama-3.1-8B \
  --tensor-parallel-size 4 --enforce-eager --gpu-memory-utilization 0.5 \
  --max-model-len 16384 --max-tokens 64 --input-len 1024 --num-prompts 8 --hold-idle 900
# terminal 2 (root): manual capture of the exact error
W=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -1 | tr -d ' ')
sudo cuda-checkpoint --action lock       --pid $W   # -> lock OK
sudo cuda-checkpoint --action checkpoint --pid $W   # -> "OS call failed or operation not supported"
```
