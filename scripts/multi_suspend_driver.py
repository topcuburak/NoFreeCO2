#!/usr/bin/env python3
"""Event-driven multi-suspend driver (root): watch held FSDP ranks, dump+resume, repeat.

Pairs with workloads/a1_fsdp/fsdp_finetune.py --suspend-steps. At each suspend step the
training script destroys its process group, writes /tmp/a1_destroyed.{rank}, and waits
for /tmp/a1_resume. This driver loops, once per suspend:

    wait until all <world> ranks are held  (all /tmp/a1_destroyed.{rank} present)
      -> cuda-checkpoint suspend -> store(host->disk) -> load -> resume   [measured, tagged]
      -> touch /tmp/a1_resume   (training reinit+rebind+continues)
      -> wait until resumed     (markers cleared by the ranks)

So you launch the training run (with N suspend steps) and this driver (with the same N
labels) and it cycles through all of them with no manual touch. Records land in
data/timed_dump.jsonl tagged --tag, with mark_min set to the suspend step.

    sudo -E $(which python) scripts/multi_suspend_driver.py \
        --suspend-steps 300,600,900,1200,1500,1800,2100,2400,2700,3000 \
        --store-out /var/data --tag a1_ft10_nvme

Needs cuda-checkpoint + root (criu not used: skip_criu). store-out picks the tier.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

import transparent_dump as td                          # noqa: E402
import timed_dump_experiment as tde                     # noqa: E402  (reuse dump_and_resume + pid detect)
from _common import build_telemetry                     # noqa: E402


def all_held(world, prefix):
    return all(os.path.exists(f"{prefix}{r}") for r in range(world))


def none_held(world, prefix):
    return not any(os.path.exists(f"{prefix}{r}") for r in range(world))


def wait_until(pred, timeout, poll=2.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(poll)
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="drive repeated suspend/dump/resume of held FSDP ranks")
    ap.add_argument("--suspend-steps", default=None,
                    help="comma labels, one per round (e.g. 300,600,...); used as mark_min tags")
    ap.add_argument("--rounds", type=int, default=None,
                    help="number of rounds if --suspend-steps not given (labels 1..rounds)")
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--store-out", default="/var/data", help="dir for the store proxy (picks the tier)")
    ap.add_argument("--tag", default="a1_ft_multi")
    ap.add_argument("--hold-seconds", type=float, default=0.0)
    ap.add_argument("--baseline", type=float, default=5.0)
    ap.add_argument("--resume-flag", default="/tmp/a1_resume")
    ap.add_argument("--destroyed-prefix", default="/tmp/a1_destroyed.")
    ap.add_argument("--wait-timeout", type=float, default=3600.0,
                    help="max wait for the next held state (covers training between suspends)")
    ap.add_argument("--cc-bin", default=None)
    ap.add_argument("--criu-bin", default=None)
    args = ap.parse_args()

    pf = td.preflight(args)
    print(f"[driver] cuda-checkpoint={pf['cuda_checkpoint']} euid={pf['euid']}")
    if not pf["cuda_checkpoint"]:
        raise SystemExit("[driver] BLOCKER: cuda-checkpoint not found")
    if pf["euid"] != 0:
        raise SystemExit("[driver] BLOCKER: not root -- run with sudo -E (cuda-checkpoint needs it)")

    labels = ([s.strip() for s in args.suspend_steps.split(",") if s.strip()]
              if args.suspend_steps else [str(i + 1) for i in range(args.rounds or 1)])
    rounds = len(labels)
    print(f"[driver] {rounds} rounds, labels={labels}, store_out={args.store_out}, tag={args.tag}")

    # clear any stale coordination files from a previous run
    for r in range(args.world):
        try: os.remove(f"{args.destroyed_prefix}{r}")
        except OSError: pass
    try: os.remove(args.resume_flag)
    except OSError: pass

    tele = build_telemetry()                             # all GPUs (the whole 4-GPU job is the workload)
    tele.start()
    try:
        for i, label in enumerate(labels):
            print(f"\n[driver] === round {i+1}/{rounds} (step {label}): waiting for all "
                  f"{args.world} ranks held ===")
            if not wait_until(lambda: all_held(args.world, args.destroyed_prefix), args.wait_timeout):
                print("[driver] timeout waiting for held state -- training ended/stuck; stopping.")
                break
            pids = tde.gpu_compute_pids()
            print(f"[driver] held. pids={pids} -> suspend/store/load/resume")
            try:
                mark = int(label)
            except ValueError:
                mark = i + 1
            try:
                tde.dump_and_resume(tele, pf["cuda_checkpoint"], pf["criu"], pids,
                                    os.path.join(_REPO, "dumps"), mark, args.baseline, False,
                                    multiproc=True, criu_root=None, hold_seconds=args.hold_seconds,
                                    skip_criu=True, store=True, store_out=args.store_out, tag=args.tag)
            except Exception:
                print(f"[driver] round {i+1} (step {label}) dump FAILED -- full traceback:")
                traceback.print_exc()
                print(f"[driver] cc states now: {td._states(pf['cuda_checkpoint'], pids)}")
            open(args.resume_flag, "w").close()             # release training (GPU is restored by now)
            print(f"[driver] touched {args.resume_flag}; waiting for training to resume ...")
            wait_until(lambda: none_held(args.world, args.destroyed_prefix), args.wait_timeout)
        print(f"\n[driver] done. {rounds} rounds -> data/timed_dump.jsonl (tag {args.tag})")
    finally:
        tele.stop()


if __name__ == "__main__":
    main()
