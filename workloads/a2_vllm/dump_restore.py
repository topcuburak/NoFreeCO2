"""A2 dump / restore / migrate operations -- the callables handed to
measure_operation(). Each takes a live vLLM `llm` handle and returns bytes moved
so the harness can cross-check S.

Data path for the accelerator domain: HBM -> PCIe -> host DRAM -> NVMe (dump),
reversed for restore. For migrate, NVMe is replaced by NIC egress (host DRAM ->
NIC), with WAN modeled from coefficients.

serve.py is now a standalone batch runner; for the dump/restore slice you hold KV
mid-generation by driving llm.llm_engine.step() until occupancy hits target, then
stop stepping (see run_a2_slice.py). These functions reach the held KV via the
cache engine on the (TP=1) driver worker.
"""
from __future__ import annotations

import os


def dump_kv_to_nvme(llm, out_dir: str) -> int:
    """Suspend: copy held KV from HBM to NVMe checkpoint. Returns bytes written.

    TODO(ford): pull paged KV blocks from the vLLM cache engine on the driver
    worker, stage through a pinned host buffer (cudaMemcpy D2H), write to
    {out_dir}/a2_kv.bin on /mnt/md0. Pin GPU clock (nvidia-smi -lgc) and the dump
    thread (numactl --membind) to the GPU's NUMA node BEFORE calling (pitfalls 1 & 2).
    """
    os.makedirs(out_dir, exist_ok=True)
    raise NotImplementedError("dump_kv_to_nvme: implement on ford")


def restore_kv_from_nvme(llm, in_dir: str) -> int:
    """Resume: load KV from NVMe back into HBM. Returns bytes read.

    Asymmetry vs dump: NVMe read ~2x faster than write; expect a warmup tail
    (first inferences slower while caches refill) -- captured by the op window.
    """
    raise NotImplementedError("restore_kv_from_nvme: implement on ford")


def migrate_kv_egress(llm, sink) -> int:
    """Source-side migrate: HBM -> host DRAM -> NIC egress. Returns bytes sent.
    WAN/destination cost is modeled (coefficients.yaml), not measured single-node.
    """
    raise NotImplementedError("migrate_kv_egress: implement on ford")
