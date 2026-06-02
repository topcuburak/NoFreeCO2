#!/usr/bin/env python3
"""A5 -- HACC cosmology simulation (GPU-capable). State 10-100 GB.

Standalone launcher: runs the HACC N-body simulation to a step where its particle
state (positions+velocities) can be snapshotted to NVMe -- the "large memory-bound
state, NVMe-write-bound dump" point. HACC itself is a C++/MPI binary; this wraps it
with the config knobs that set state size and drives it to a checkpoint step.

    python simulate.py --hacc-bin /opt/hacc/bin/hacc_tpm \
        --particles 256 --steps 50 --checkpoint-step 25 \
        --checkpoint-dir /mnt/md0/a5ckpt

State (bytes) ~= particles^3 * bytes_per_particle. subprocess used to launch the
binary; runs on ford.
"""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="A5 HACC cosmology simulation launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--hacc-bin", default="/opt/hacc/bin/hacc_tpm",
                   help="path to the HACC simulation binary")
    p.add_argument("--params-file", default=None, help="HACC indat/params file")
    p.add_argument("--particles", type=int, default=256,
                   help="np per dim (total = particles^3); sets state size")
    p.add_argument("--box-size", type=float, default=256.0, help="Mpc/h")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--checkpoint-step", type=int, default=25,
                   help="step at which to snapshot particle state for dump measurement")
    p.add_argument("--ranks", type=int, default=4, help="MPI ranks (1 per GPU)")
    p.add_argument("--checkpoint-dir", default="/mnt/md0/a5ckpt")
    return p


def main() -> None:
    args = build_parser().parse_args()
    # import subprocess, os here
    # 1. compose the HACC command (mpirun -n args.ranks args.hacc_bin args.params_file ...)
    #    with --particles / --box-size mapped into the indat file or CLI
    # 2. launch and run to args.checkpoint_step
    # 3. trigger HACC's restart/analysis dump (particle state -> args.checkpoint_dir)
    #    -- this NVMe write is the dump the harness measures (NVMe-write-bound)
    raise NotImplementedError(
        "A5 HACC launcher: wire up the binary + restart dump on ford with the args above")


if __name__ == "__main__":
    main()
