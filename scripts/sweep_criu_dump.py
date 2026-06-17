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
    return dir_size(img)


def criu_restore(criu, img):
    r = subprocess.run([criu, "restore", "-D", img,
                        "--restore-detached", "--file-locks", "--shell-job"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("criu restore rc=%d: %s" % (r.returncode, (r.stderr or r.stdout or "")[-600:]))
    return 1


def hold_pid():
    out = subprocess.run(["pgrep", "-f", "hold_dram.py"], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def wait_resident(min_bytes, timeout=120):
    dl = time.monotonic() + timeout
    while time.monotonic() < dl:
        pid = hold_pid()
        if pid:
            try:
                for line in open(f"/proc/{pid}/status"):
                    if line.startswith("VmRSS:") and int(line.split()[1]) * 1024 >= min_bytes:
                        return pid
            except OSError:
                pass
        time.sleep(1)
    return hold_pid()


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="4,8,16,32", help="comma GiB DRAM footprints")
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--store-out", default="/var/data", help="criu image dir TIER (not /tmp)")
    ap.add_argument("--baseline", type=float, default=5.0)
    ap.add_argument("--tag", default="s2_criu_nvme")
    ap.add_argument("--criu-bin", default=None)
    args = ap.parse_args()
    if os.geteuid() != 0:
        raise SystemExit("[s2] run as root (sudo -E) -- criu needs CAP_SYS_ADMIN + RAPL")
    criu = _resolve_criu(args.criu_bin)
    drive_w = 3.0 if "home" in args.store_out else 50.0
    hold_py = os.path.join(_REPO, "scripts", "hold_dram.py")
    sizes = [float(s) for s in args.sizes.split(",") if s.strip()]
    print(f"[s2] criu={criu} drive_w={drive_w} store_out={args.store_out} sizes={sizes}")

    tele = build_telemetry(nvml_gpus=[])                     # CPU domain: exclude GPU
    tele.start()

    def emit(rec, phase, gb, c, img_bytes=None):
        dram = DRAM_W_PER_GB * gb * rec.latency_s
        drive = drive_w * rec.latency_s
        meas = sum(s.energy_abs_j or 0.0 for s in rec.sources if s.name == "cpu_pkg_energy_rapl")
        rec.extra.update(mark_min=c, measured_abs_j=round(meas, 1), dram_model_j=round(dram, 1),
                         drive_model_j=round(drive, 1), full_total_j=round(meas + dram + drive, 1),
                         footprint_gb=gb, image_bytes=img_bytes)
        rec.config.update(tag=args.tag, phase=phase, workload="s2_criu")
        print_record(rec)
        print(f"  modeled DRAM {dram:.0f} + drive {drive:.0f} | FULL (CPU+DRAM+drive): {meas+dram+drive:.0f} J",
              flush=True)
        write_record(rec, "timed_dump")

    try:
        for gb in sizes:
            img = os.path.join(args.store_out, "criu_s2_img")
            proc = subprocess.Popen([sys.executable, hold_py, "--gb", str(gb), "--seconds", "100000"],
                                    start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            pid = wait_resident(gb * (1024 ** 3) * 0.9)
            if not pid:
                print(f"[s2] {gb}GB: hold_dram not resident -- skip"); continue
            print(f"\n[s2] ===== {gb} GiB, PID={pid} =====", flush=True)
            for c in range(args.cycles):
                shutil.rmtree(img, ignore_errors=True); os.makedirs(img, exist_ok=True)
                try:
                    rec_d = measure_operation(tele, workload="timed_dump", operation="criu_dump",
                        state_bytes=int(gb * 1e9), baseline_seconds=args.baseline,
                        op=lambda: criu_dump(criu, pid, img), config={"phase": "dump"})
                    emit(rec_d, "dump", gb, c, dir_size(img))
                except Exception as e:
                    print(f"[s2] {gb}GB cyc{c} DUMP failed: {e}"); break
                try: os.kill(pid, 9)
                except OSError: pass
                try: proc.wait(timeout=5)                    # reap our child so criu can reuse the PID
                except Exception: pass
                for _ in range(20):
                    if hold_pid() is None: break
                    time.sleep(0.5)
                try:
                    rec_r = measure_operation(tele, workload="timed_dump", operation="criu_restore",
                        state_bytes=int(gb * 1e9), baseline_seconds=args.baseline,
                        op=lambda: criu_restore(criu, img), config={"phase": "restore"})
                    emit(rec_r, "restore", gb, c)
                except Exception as e:
                    print(f"[s2] {gb}GB cyc{c} RESTORE failed: {e}"); break
                pid = wait_resident(gb * (1024 ** 3) * 0.9, timeout=30) or hold_pid()
                print(f"[s2] {gb}GB cycle {c} done (pid now {pid})", flush=True)
            subprocess.run(["pkill", "-9", "-f", "hold_dram.py"])
            time.sleep(2); shutil.rmtree(img, ignore_errors=True)
    finally:
        tele.stop()
    print("[s2] done -> data/timed_dump.jsonl tag", args.tag)


if __name__ == "__main__":
    main()
