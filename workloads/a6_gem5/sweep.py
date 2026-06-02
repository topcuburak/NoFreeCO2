#!/usr/bin/env python3
"""A6 -- gem5 multi-process parameter sweep (SPEC CPU2017). State ~10-100 MB x N.

Standalone launcher: spawns N gem5 simulation processes (a SPEC CPU2017 sweep) and
drives them to a steady state where each process is CRIU-checkpointed. This is the
HOST-domain anchor (CPU cores + DRAM + NVMe; no accelerator) and the
"shifting-friendly" baseline where prior carbon-aware work actually holds.

    python sweep.py --gem5-bin /opt/gem5/build/X86/gem5.opt \
        --config /opt/gem5/configs/example/se.py \
        --spec-dir /mnt/md0/spec2017 --benchmarks 600.perlbench,602.gcc \
        --procs 64 --checkpoint-dir /mnt/md0/a6ckpt

Dump = CRIU process checkpoint (NVMe-write + CRIU bookkeeping). subprocess-based;
runs on ford (CRIU needs root/CAP_SYS_ADMIN).
"""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="A6 gem5 + SPEC CPU2017 sweep launcher (host-domain anchor)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gem5-bin", default="/opt/gem5/build/X86/gem5.opt")
    p.add_argument("--config", default="/opt/gem5/configs/example/se.py",
                   help="gem5 config script")
    p.add_argument("--spec-dir", default="/mnt/md0/spec2017")
    p.add_argument("--benchmarks", default="600.perlbench,602.gcc,605.mcf",
                   help="comma-separated SPEC CPU2017 workloads to sweep")
    p.add_argument("--procs", type=int, default=64,
                   help="concurrent gem5 processes (CPU has 64 threads)")
    p.add_argument("--cpu-type", default="O3CPU", help="gem5 CPU model swept")
    p.add_argument("--warmup-insts", type=int, default=1_000_000_000,
                   help="fast-forward this many insts, then hold for checkpointing")
    p.add_argument("--checkpoint-engine", choices=["criu", "gem5"], default="criu",
                   help="criu = process checkpoint (host-domain dump); gem5 = native ckpt")
    p.add_argument("--checkpoint-dir", default="/mnt/md0/a6ckpt")
    return p


def main() -> None:
    args = build_parser().parse_args()
    # import subprocess, os here
    # 1. for each (benchmark, sweep-point), launch a gem5 process:
    #      args.gem5_bin args.config --cpu-type args.cpu_type --benchmark <b> ...
    #    pin across the 64 threads / 4 NUMA nodes (numactl) to control DRAM energy
    # 2. fast-forward args.warmup_insts so each process holds ~10-100 MB resident
    # 3. CRIU-dump each process tree to args.checkpoint_dir (criu dump -t <pid>)
    #    -- aggregate dump = the host-domain mechanism cost the harness measures
    raise NotImplementedError(
        "A6 gem5 sweep + CRIU dump: wire up on ford with the args above")


if __name__ == "__main__":
    main()
