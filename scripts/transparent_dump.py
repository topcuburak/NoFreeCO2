#!/usr/bin/env python3
"""Transparent (app-blind) dump: cuda-checkpoint (HBM -> host) + criu dump (host -> NVMe).

Measures the FULL-footprint transparent checkpoint of a live GPU process, decomposed
across the two domains of the cost model:

  phase 1  cuda-checkpoint : all device memory  HBM -> PCIe -> host DRAM   (ACCELERATOR)
  phase 2  criu dump       : process image      host DRAM -> NVMe          (HOST)

Each phase is wrapped in the measurement harness (energy + latency); bytes are taken
from the GPU memory freed (phase 1) and the CRIU image size (phase 2). Compare the
S here (full footprint) against S from the app-aware kv_dump to get the transparency
tax. See measurement plan: two-domain decomposition.

REQUIREMENTS (on ford):
  - cuda-checkpoint in PATH (NVIDIA, needs driver R550+)
  - criu in PATH; run as root (criu needs CAP_SYS_ADMIN):  sudo -E python ...
  - host RAM >= the process's GPU footprint (cuda-checkpoint stages HBM in DRAM)

VALIDATE THE LOOP FIRST on a trivial target, then move to vLLM:
    python scripts/hold_gpu.py --gb 20 &                       # prints PID
    sudo -E python scripts/transparent_dump.py --pids <PID> --out /mnt/md0/tdump
    # real vLLM (start with TP=1, single worker process):
    sudo -E python scripts/transparent_dump.py --proc-name VllmWorker --out /mnt/md0/tdump

NOTE: multi-process TP=4 + NCCL is the hard case for criu; prove TP=1 first, then
model the TP=4 cost from the per-byte coefficients (measurement plan).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from harness import measure_operation                              # noqa: E402
from _common import build_telemetry, write_record, print_record    # noqa: E402

try:
    import pynvml  # noqa: E402
    _HAVE_NVML = True
except Exception:
    _HAVE_NVML = False


# --------------------------------------------------------------------------- #
# preflight + helpers
# --------------------------------------------------------------------------- #
def _resolve(name: str, override: str | None, extra_dirs: list[str]) -> str | None:
    """which(), then explicit override, then common sbin/cuda dirs (criu lives in
    /usr/sbin on Debian, which sudo -E keeps out of PATH)."""
    if override:
        return override if os.path.exists(override) else None
    p = shutil.which(name)
    if p:
        return p
    for d in extra_dirs:
        cand = os.path.join(d, name)
        if os.path.exists(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def preflight(args) -> dict:
    info = {}
    info["cuda_checkpoint"] = _resolve(
        "cuda-checkpoint", args.cc_bin, ["/usr/local/bin", "/usr/local/cuda/bin"])
    info["criu"] = _resolve(
        "criu", args.criu_bin, ["/usr/sbin", "/sbin", "/usr/local/sbin"])
    info["euid"] = os.geteuid()
    problems = []
    if not info["cuda_checkpoint"]:
        problems.append("cuda-checkpoint not in PATH (NVIDIA tool, driver R550+)")
    if not info["criu"]:
        problems.append("criu not in PATH (install criu)")
    if info["euid"] != 0:
        problems.append("not root: criu needs CAP_SYS_ADMIN -> run with `sudo -E`")
    info["problems"] = problems
    return info


def find_pids(args) -> list[int]:
    if args.pids:
        return [int(p) for p in args.pids.split(",") if p.strip()]
    if args.proc_name:
        out = subprocess.run(["pgrep", "-f", args.proc_name],
                             capture_output=True, text=True).stdout
        return [int(p) for p in out.split()]
    raise SystemExit("specify --pids or --proc-name")


def gpu_used_bytes() -> int:
    """Total used HBM across all GPUs (proxy for the process footprint to dump)."""
    if not _HAVE_NVML:
        return 0
    try:
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        return sum(pynvml.nvmlDeviceGetMemoryInfo(
            pynvml.nvmlDeviceGetHandleByIndex(i)).used for i in range(n))
    except Exception:
        return 0


def rss_bytes(pids: list[int]) -> int:
    """Sum of resident set size across the target PIDs (KB in /proc -> bytes)."""
    total = 0
    for pid in pids:
        try:
            for line in open(f"/proc/{pid}/status"):
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
        except OSError:
            pass
    return total


def dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def host_free_bytes() -> int:
    try:
        import os as _os
        return _os.sysconf("SC_AVPHYS_PAGES") * _os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return 0


# --------------------------------------------------------------------------- #
# the two measured operations
# --------------------------------------------------------------------------- #
def _cc(cc_bin: str, pid: int, *cc_args: str) -> str:
    """Run one cuda-checkpoint action; on failure raise RuntimeError that SURFACES the
    tool's stderr (the real reason, e.g. CUDA_ERROR_OPERATING_SYSTEM 304 on a live IPC
    handle) instead of a bare 'non-zero exit status'."""
    r = subprocess.run([cc_bin, *cc_args, "--pid", str(pid)], capture_output=True, text=True)
    if r.returncode != 0:
        msg = (r.stderr or r.stdout or "").strip()
        raise RuntimeError(f"cuda-checkpoint {' '.join(cc_args)} --pid {pid} "
                           f"failed (rc={r.returncode}): {msg}")
    return (r.stdout or "").strip()


def cc_get_state(cc_bin: str, pid: int) -> str:
    """Best-effort process checkpoint state (running/locked/checkpointed). For diagnosis."""
    try:
        r = subprocess.run([cc_bin, "--get-state", "--pid", str(pid)],
                           capture_output=True, text=True)
        return (r.stdout or r.stderr or "").strip() or "?"
    except Exception as e:
        return f"<get-state err: {e}>"


def _states(cc_bin, pids):
    return " ".join(f"{p}:{cc_get_state(cc_bin, p)}" for p in pids)


def cuda_suspend(cc_bin: str, pids: list[int], multiproc: bool = False) -> int:
    """running -> checkpointed (HBM -> host, GPU freed).

    TP=1 (single proc): per-pid --toggle.
    TP>1 (multiproc):  lock ALL first (drains in-flight NCCL collectives to a
    consistent, collective-free point), THEN checkpoint ALL. Locking one proc
    while others run mid-collective would deadlock -- hence lock-all-then-ckpt-all.
    """
    if multiproc:
        print(f"[cc] suspend: pre-state {_states(cc_bin, pids)}", flush=True)
        for p in pids:
            _cc(cc_bin, p, "--action", "lock")
        print(f"[cc] suspend: locked   {_states(cc_bin, pids)}", flush=True)
        for p in pids:
            _cc(cc_bin, p, "--action", "checkpoint")
        print(f"[cc] suspend: ckpted   {_states(cc_bin, pids)}", flush=True)
    else:
        for p in pids:
            _cc(cc_bin, p, "--toggle")
    return len(pids)


def cuda_resume(cc_bin: str, pids: list[int], multiproc: bool = False) -> int:
    """checkpointed -> running (host -> HBM). multiproc: restore ALL then unlock ALL.
    (--device-map can be added to restore when GPUs differ -- for migration.)"""
    if multiproc:
        for p in pids:
            _cc(cc_bin, p, "--action", "restore")
        for p in pids:
            _cc(cc_bin, p, "--action", "unlock")
        print(f"[cc] resume:  post-state {_states(cc_bin, pids)}", flush=True)
    else:
        for p in reversed(pids):
            _cc(cc_bin, p, "--toggle")
    return len(pids)


def cuda_recover(cc_bin: str, pids: list[int]) -> None:
    """Best-effort: bring processes back to running after a partial/failed suspend
    (some may be locked, some checkpointed). restore then unlock each, ignoring errors
    for the ones not in that state. Leaves everything as close to 'running' as possible."""
    for p in pids:
        for act in ("restore", "unlock"):
            try:
                _cc(cc_bin, p, "--action", act)
            except Exception:
                pass
    print(f"[cc] recover: post-state {_states(cc_bin, pids)}", flush=True)


# kept for back-compat (TP=1 single-phase callers)
def op_cuda_checkpoint(cc_bin: str, pids: list[int]) -> int:
    return cuda_suspend(cc_bin, pids, multiproc=False)


def op_criu_dump(criu_bin: str, pids: list[int], out_dir: str, leave_running: bool,
                 criu_root: int | None = None) -> int:
    """criu dump the (now GPU-free) process image to NVMe. Returns image bytes.

    criu_root set (TP>1): dump the tree rooted at that PID (-t root captures all
    descendant workers as one consistent set). Else dump each pid separately (TP=1).
    """
    os.makedirs(out_dir, exist_ok=True)
    targets = [criu_root] if criu_root else pids
    for pid in targets:
        d = os.path.join(out_dir, f"pid_{pid}")
        os.makedirs(d, exist_ok=True)
        cmd = [criu_bin, "dump", "-t", str(pid), "-D", d, "--shell-job"]
        if leave_running:
            cmd.append("--leave-running")
        # NOTE(ford): real procs often need extra flags -- add as criu complains:
        #   --tcp-established --ext-unix-sk --file-locks --link-remap
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            # surface criu's actual complaint (also in {d}/dump.log)
            tail = (r.stderr or "")[-2500:]
            raise RuntimeError(f"criu dump failed for pid {pid} (see {d}/dump.log):\n{tail}")
    return dir_size_bytes(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Transparent dump (cuda-checkpoint + criu), measured + decomposed")
    ap.add_argument("--pids", default=None, help="comma-separated target PIDs")
    ap.add_argument("--proc-name", default=None, help="pgrep -f pattern (e.g. VllmWorker)")
    ap.add_argument("--out", default=os.path.join(_REPO, "dumps"),
                    help="CRIU image dir (default: <repo>/dumps; gitignored)")
    ap.add_argument("--baseline", type=float, default=10.0, help="telemetry baseline seconds")
    ap.add_argument("--leave-running", action="store_true",
                    help="criu --leave-running (don't kill the process after dump)")
    ap.add_argument("--label", default="transparent_dump")
    ap.add_argument("--criu-bin", default=None, help="explicit path to criu (else auto: /usr/sbin)")
    ap.add_argument("--cc-bin", default=None, help="explicit path to cuda-checkpoint")
    ap.add_argument("--multiproc", action="store_true",
                    help="TP>1: lock ALL pids then checkpoint ALL (instead of per-pid --toggle)")
    ap.add_argument("--criu-root", type=int, default=None,
                    help="TP>1: root PID for a criu tree dump (parent of the workers)")
    args = ap.parse_args()

    pf = preflight(args)
    print(f"[tdump] cuda-checkpoint={pf['cuda_checkpoint']} criu={pf['criu']} euid={pf['euid']}")
    if pf["problems"]:
        for p in pf["problems"]:
            print(f"[tdump] BLOCKER: {p}")
        raise SystemExit(1)

    pids = find_pids(args)
    if not pids:
        raise SystemExit("[tdump] no target PIDs found")
    print(f"[tdump] target PIDs: {pids}")

    gpu_before = gpu_used_bytes()
    free_before = host_free_bytes()
    print(f"[tdump] GPU used (footprint to dump): {gpu_before/1e9:.2f} GB")
    print(f"[tdump] host free RAM: {free_before/1e9:.2f} GB")
    if free_before and gpu_before and free_before < gpu_before:
        print(f"[tdump] WARNING: host free RAM < GPU footprint -- cuda-checkpoint may fail "
              f"(it stages HBM in DRAM).")

    tele = build_telemetry()
    tele.start()
    try:
        # ---- phase 1: ACCELERATOR -- cuda-checkpoint (HBM -> host) ----
        rec1 = measure_operation(
            tele, workload=args.label, operation="cuda_checkpoint",
            state_bytes=gpu_before, baseline_seconds=args.baseline,
            op=lambda: cuda_suspend(pf["cuda_checkpoint"], pids, args.multiproc),
            config={"pids": pids, "domain": "accelerator", "multiproc": args.multiproc},
        )
        gpu_after = gpu_used_bytes()
        rec1.extra["gpu_freed_bytes"] = gpu_before - gpu_after
        print_record(rec1)
        print(f"  GPU freed: {(gpu_before-gpu_after)/1e9:.2f} GB  (now {gpu_after/1e9:.2f} GB used)")
        write_record(rec1, "transparent_dump")

        # ---- phase 2: HOST -- criu dump (host DRAM -> NVMe) ----
        host_footprint = rss_bytes(pids)
        rec2 = measure_operation(
            tele, workload=args.label, operation="criu_dump",
            state_bytes=host_footprint, baseline_seconds=args.baseline,
            op=lambda: op_criu_dump(pf["criu"], pids, args.out, args.leave_running,
                                    criu_root=args.criu_root),
            config={"pids": pids, "domain": "host", "out": args.out,
                    "criu_root": args.criu_root},
        )
        img_bytes = int(rec2.extra.get("op_result") or 0)
        rec2.extra["image_bytes"] = img_bytes
        rec2.extra["rss_bytes"] = host_footprint
        print_record(rec2)
        print(f"  CRIU image size: {img_bytes/1e9:.2f} GB  (process RSS {host_footprint/1e9:.2f} GB)")
        write_record(rec2, "transparent_dump")

        # ---- summary: the transparent-dump cost (both domains) ----
        total_e = rec1.total_energy_j + rec2.total_energy_j
        total_t = rec1.latency_s + rec2.latency_s
        print(f"\n[tdump] TRANSPARENT DUMP total: {total_e:.1f} J, {total_t:.2f} s, "
              f"S(full footprint)~{gpu_before/1e9:.1f} GB")
        print(f"[tdump] -> compare against kv_dump (S=live KV) for the transparency tax")
    finally:
        tele.stop()


if __name__ == "__main__":
    main()
