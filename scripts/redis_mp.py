#!/usr/bin/env python3
"""A7 -- multi-PROCESS real workload: N redis-server instances (a real, on-theme stateful KV
service), each loaded to ~gb/N via server-side DEBUG POPULATE. The python launcher sets comm
'a7redis' and is the criu TREE ROOT; the driver finds it by comm, sums RSS over the whole
subtree (launcher + every redis child), and criu dumps the tree (`criu -t <root>` includes
descendants). redis uses an epoll event loop + jemalloc anon memory (no io_uring/GPU), so it is
criu-friendly; we disable RDB/AOF saves so no background fork races the dump, and there are no
established client connections at dump time (only listening sockets, which criu handles).

    python scripts/redis_mp.py --gb 64 --procs 8 --seconds 100000
"""
from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import time


def _setcomm(name: str) -> None:
    try:
        ctypes.CDLL("libc.so.6").prctl(15, name.encode(), 0, 0, 0)   # PR_SET_NAME
    except Exception:
        pass


def _cli(port: int, *args) -> str:
    return subprocess.run(["redis-cli", "-p", str(port), *[str(a) for a in args]],
                          capture_output=True, text=True).stdout.strip()


def _wait_ready(port: int, timeout: float = 30.0) -> bool:
    dl = time.time() + timeout
    while time.time() < dl:
        if _cli(port, "PING") == "PONG":
            return True
        time.sleep(0.5)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=64.0, help="TOTAL dataset across all redis (GB)")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--seconds", type=int, default=100000)
    ap.add_argument("--base-port", type=int, default=7000)
    ap.add_argument("--valsize", type=int, default=1024, help="value bytes per key (DEBUG POPULATE)")
    ap.add_argument("--redis-bin", default="redis-server")
    a = ap.parse_args()
    _setcomm("a7redis")

    P = max(1, a.procs)
    per_bytes = a.gb * 1e9 / P
    count = max(1, int(per_bytes / (a.valsize + 100)))     # ~valsize + key/overhead per entry
    ports = [a.base_port + i for i in range(P)]
    for port in ports:                                     # our own instances, custom ports
        subprocess.Popen([a.redis_bin, "--port", str(port), "--save", "", "--appendonly", "no",
                          "--enable-debug-command", "yes", "--maxmemory", "0",
                          "--daemonize", "no", "--protected-mode", "no"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for port in ports:
        if not _wait_ready(port):
            print(f"[a7-redis] redis on {port} not ready", flush=True)
    print(f"[a7-redis] PID={os.getpid()} {a.gb} GB over {P} redis, "
          f"populating {count} keys x {a.valsize}B each", flush=True)
    for port in ports:
        _cli(port, "DEBUG", "POPULATE", count, "key", a.valsize)
    total = sum(int(_cli(port, "DBSIZE") or 0) for port in ports)
    print(f"[a7-redis] populated, total keys={total}", flush=True)

    stop = time.time() + a.seconds
    while time.time() < stop:                              # idle-hold the dataset for the criu cycles
        time.sleep(5)


if __name__ == "__main__":
    main()
