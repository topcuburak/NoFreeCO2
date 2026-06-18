#!/usr/bin/env python3
"""S2 -- criu dump/restore cost vs DRAM footprint, MEASURED energy, per tier (CPU domain).

hold_dram holds N GiB resident DRAM; per cycle: criu dump (DRAM->disk, --leave-running) ->
[kill] -> criu restore (disk->DRAM). The host-domain analogue of the GPU suspend/restore, and
the only place real criu runs (a plain anon-memory process; no io_uring/GPU/sockets).

Per leg, measured: CPU pkg (RAPL ∫P dt) + MODELED DRAM (0.3 W/GB) + drive (NVMe 50 / SATA 3 W).
No GPU (CPU workload). criu dump = DRAM read + disk write in ONE op; restore = disk read + DRAM
write. Image dir on the TIER under test (NOT /tmp = tmpfs).

    sudo -E python scripts/sweep_criu_dump.py --sizes 4,8,16,32 --cycles 4 \
        --store-out /var/data --tag s2_criu_nvme        # SATA: --store-out /home/test --tag s2_criu_sata
Run as root (criu needs CAP_SYS_ADMIN + RAPL). Records -> data/timed_dump.jsonl.
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO)

from harness import measure_operation                       # noqa: E402
from _common import build_telemetry, write_record, print_record  # noqa: E402

DRAM_W_PER_GB = 0.3


def _resolve_criu(override=None):
    return override or shutil.which("criu") or "/usr/sbin/criu"


def dir_size(p):
    t = 0
    for root, _, fs in os.walk(p):
        for f in fs:
            try: t += os.path.getsize(os.path.join(root, f))
            except OSError: pass
    return t


def criu_dump(criu, pid, img):
    r = subprocess.run([criu, "dump", "-t", str(pid), "-D", img,
                        "--leave-running", "--file-locks", "--shell-job"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("criu dump rc=%d: %s" % (r.returncode, (r.stderr or r.stdout or "")[-600:]))
    os.sync()                                               # FLUSH the image to disk (else dump
    return dir_size(img)                                    # returns at page-cache speed, not the tier)


def drop_caches():
    os.system("sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null")   # untimed: force cold restore


def criu_restore(criu, img):
    r = subprocess.run([criu, "restore", "-D", img,
                        "--restore-detached", "--file-locks", "--shell-job"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("criu restore rc=%d: %s" % (r.returncode, (r.stderr or r.stdout or "")[-600:]))
    return 1


def _comm(pid):
    try:
        return open(f"/proc/{pid}/comm").read().strip()
    except OSError:
        return None


def all_pids(pat, want_comm=None):
    out = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True).stdout.split()
    me = os.getpid()
    pids = sorted(int(p) for p in out if int(p) != me)   # never match the driver itself
    if want_comm:                                        # require comm (e.g. "pr"/"a7mp"/"a8mt"):
        pids = [p for p in pids if _comm(p) == want_comm]   # excludes python/sudo/criu ancestors
    return pids


def hold_pid(pat, want_comm=None):
    pids = all_pids(pat, want_comm)          # tree ROOT = lowest pid (launched before its forks)
    return pids[0] if pids else None


def rss_of(pid):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def _children_map():
    m = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            data = open(f"/proc/{d}/stat").read()
            ppid = int(data[data.rindex(")") + 2:].split()[1])   # field after comm: state, PPID
        except (OSError, ValueError, IndexError):
            continue
        m.setdefault(ppid, []).append(int(d))
    return m


def subtree(root):
    if not root:
        return []
    m = _children_map()
    seen, stack = [], [root]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.append(p)
        stack.extend(m.get(p, []))
    return seen                                       # root + all descendants (mixed comm ok)


def tree_rss(root):
    return sum(rss_of(p) or 0 for p in subtree(root))


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except OSError:
        return False


def wait_resident(min_bytes, pat, timeout=120, want_comm=None):
    dl = time.monotonic() + timeout
    while time.monotonic() < dl:
        root = hold_pid(pat, want_comm)              # tree root, found by its (unique) comm
        if root and tree_rss(root) >= min_bytes:     # footprint may span the whole descendant tree
            return root
        time.sleep(1)
    return hold_pid(pat, want_comm)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="4,8,16,32", help="comma GiB DRAM footprints")
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--store-out", default="/var/data", help="criu image dir TIER (not /tmp)")
    ap.add_argument("--baseline", type=float, default=5.0)
    ap.add_argument("--tag", default="s2_criu_nvme")
    ap.add_argument("--criu-bin", default=None)
    ap.add_argument("--target", default="work_dram.py",
                    help="target script in scripts/: work_dram.py (ACTIVE compute, default) or "
                         "hold_dram.py (idle). Must take --gb --seconds.")
    ap.add_argument("--launch", default=None,
                    help="full command to launch the workload instead of the python --target "
                         "(e.g. a GAPBS binary). The process it spawns IS the criu target.")
    ap.add_argument("--pat", default=None,
                    help="pgrep pattern for the RSS-holding PID (defaults to --target). Required "
                         "with --launch so we find/kill the right process.")
    ap.add_argument("--wait-timeout", type=int, default=120,
                    help="seconds to wait for the workload to reach footprint (bump for big graphs)")
    ap.add_argument("--comm", default=None,
                    help="override target comm (/proc/PID/comm) for pid matching. Use when the target "
                         "sets its own name via prctl (A7 'a7mp', A8 'a8mt') so we never match the "
                         "python driver / sudo (which share the script name in argv).")
    ap.add_argument("--target-extra", default="",
                    help="extra args appended to the python --target launch (e.g. '--procs 16')")
    args = ap.parse_args()
    if os.geteuid() != 0:
        raise SystemExit("[s2] run as root (sudo -E) -- criu needs CAP_SYS_ADMIN + RAPL")
    criu = _resolve_criu(args.criu_bin)
    drive_w = 3.0 if "home" in args.store_out else 50.0
    target_py = os.path.join(_REPO, "scripts", args.target)
    pat = args.pat or args.target
    # require the target's comm so we never match the python driver / sudo / criu (which all carry
    # the launch/target string in their argv). Explicit --comm wins (A7/A8 set it via prctl);
    # else derive from a --launch binary's basename (15-char kernel limit).
    want_comm = args.comm or (os.path.basename(shlex.split(args.launch)[0])[:15] if args.launch else None)
    sizes = [float(s) for s in args.sizes.split(",") if s.strip()]
    print(f"[s2] criu={criu} target={args.target} drive_w={drive_w} store_out={args.store_out} sizes={sizes}")

    tele = build_telemetry(nvml_gpus=[])                     # CPU domain: exclude GPU
    tele.start()

    def emit(rec, phase, gb, c, img_bytes=None, rss_bytes=None):
        dram = DRAM_W_PER_GB * gb * rec.latency_s
        drive = drive_w * rec.latency_s
        meas = sum(s.energy_abs_j or 0.0 for s in rec.sources if s.name == "cpu_pkg_energy_rapl")
        rec.extra.update(mark_min=c, measured_abs_j=round(meas, 1), dram_model_j=round(dram, 1),
                         drive_model_j=round(drive, 1), full_total_j=round(meas + dram + drive, 1),
                         footprint_gb=gb, image_bytes=img_bytes, rss_bytes=rss_bytes)
        rec.config.update(tag=args.tag, phase=phase, workload="s2_criu")
        print_record(rec)
        print(f"  modeled DRAM {dram:.0f} + drive {drive:.0f} | FULL (CPU+DRAM+drive): {meas+dram+drive:.0f} J",
              flush=True)
        write_record(rec, "timed_dump")

    try:
        for gb in sizes:
            img = os.path.join(args.store_out, "criu_s2_img")
            # stdout -> /dev/null: a growing log fd breaks criu restore (fd-size check); the
            # workload stays fully active, we just don't capture progress.
            if args.launch:                                  # arbitrary binary (e.g. GAPBS) IS the target
                cmd = args.launch
            else:
                cmd = f"{sys.executable} {target_py} --gb {gb} --seconds 100000 {args.target_extra}"
            proc = subprocess.Popen(shlex.split(cmd), start_new_session=True,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            pid = wait_resident(gb * (1024 ** 3) * 0.9, pat, timeout=args.wait_timeout,
                                want_comm=want_comm)
            if not pid:
                print(f"[s2] {gb}GB: {args.target} not resident -- skip"); continue
            print(f"\n[s2] ===== {gb} GiB, PID={pid} =====", flush=True)
            for c in range(args.cycles):
                if pid is None:
                    print(f"[s2] {gb}GB: lost the process before cycle {c} -- stopping this size"); break
                shutil.rmtree(img, ignore_errors=True); os.makedirs(img, exist_ok=True)
                rss = tree_rss(pid)                          # whole-tree resident bytes at dump time
                try:
                    rec_d = measure_operation(tele, workload="timed_dump", operation="criu_dump",
                        state_bytes=int(rss or gb * 1e9), baseline_seconds=args.baseline,
                        op=lambda: criu_dump(criu, pid, img), config={"phase": "dump"})
                    emit(rec_d, "dump", gb, c, dir_size(img), rss)
                except Exception as e:
                    print(f"[s2] {gb}GB cyc{c} DUMP failed: {e}"); break
                try: pgid = os.getpgid(pid)                   # kill the whole GROUP (multi-process tree):
                except OSError: pgid = pid                    # leaving children alive orphans them and
                try: os.killpg(pgid, 9)                       # collides pids on restore
                except OSError:
                    try: os.kill(pid, 9)
                    except OSError: pass
                try: proc.wait(timeout=2)                    # reap our child (cycle 0) so the PID frees
                except Exception: pass
                gone = False                                  # criu restore reclaims the ORIGINAL pids +
                for _ in range(240):                          # every TID; a lingering zombie (slow 64-
                    if not group_alive(pgid) and not os.path.exists(f"/proc/{pid}"):   # thread SIGKILL
                        gone = True; break                    # under load) holds a pid -> 'fork: File
                    try: proc.wait(timeout=0.5)               # exists'. Block until the WHOLE group
                    except Exception: pass                    # (parent + every child) is gone.
                    time.sleep(0.5)
                if not gone:
                    print(f"[s2] {gb}GB cyc{c}: pid {pid} would not exit -- skip restore"); break
                drop_caches()                                # untimed: evict the image -> cold read
                try:
                    rec_r = measure_operation(tele, workload="timed_dump", operation="criu_restore",
                        state_bytes=int(gb * 1e9), baseline_seconds=args.baseline,
                        op=lambda: criu_restore(criu, img), config={"phase": "restore"})
                    emit(rec_r, "restore", gb, c)
                except Exception as e:
                    print(f"[s2] {gb}GB cyc{c} RESTORE failed: {e}"); break
                pid = (wait_resident(gb * (1024 ** 3) * 0.9, pat, timeout=30, want_comm=want_comm)
                       or hold_pid(pat, want_comm))
                print(f"[s2] {gb}GB cycle {c} done (pid now {pid})", flush=True)
            if want_comm:
                subprocess.run(["pkill", "-9", "-x", want_comm])   # kill by comm, not the driver
            else:
                subprocess.run(["pkill", "-9", "-f", pat])
            time.sleep(2); shutil.rmtree(img, ignore_errors=True)
    finally:
        tele.stop()
    print("[s2] done -> data/timed_dump.jsonl tag", args.tag)


if __name__ == "__main__":
    main()
