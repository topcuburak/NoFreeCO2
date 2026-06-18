#!/usr/bin/env python3
"""A7 -- multi-PROCESS real workload: P independent DuckDB instances (separate address spaces),
each holding gb/P of an uncompressed in-memory table and looping a batch GROUP BY. The criu
process-tree counterpart to A8's single-process / many-threads DuckDB (work_duck.py): same real
analytics engine, latency-insensitive batch, but parallelised across processes instead of threads.
criu-safe (pure anon memory, no runtime sockets), unlike NPB-MPI whose OOB TCP listener cannot be
re-bound on restore. Every process sets comm 'a7duck'; the driver sums the subtree RSS and dumps
the tree root.

    python scripts/duck_mp.py --gb 64 --procs 16 --seconds 100000
"""
from __future__ import annotations

import argparse
import ctypes
import os
import time


def _setcomm(name: str) -> None:
    try:
        ctypes.CDLL("libc.so.6").prctl(15, name.encode(), 0, 0, 0)   # PR_SET_NAME
    except Exception:
        pass


def hold(gb: float, threads: int, seconds: int) -> None:
    import duckdb
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={max(1, threads)}")
    con.execute("PRAGMA memory_limit='700GB'")
    con.execute("PRAGMA force_compression='Uncompressed'")       # predictable per-proc RSS
    rows = int(gb * 1e9 / 40)                                    # 5 cols x 8 bytes ~ 40 B/row
    con.execute(f"""CREATE TABLE t AS
        SELECT i AS id,
               (random() * 1e18)::BIGINT AS k,
               (random() * 1e18)::BIGINT AS a,
               (random() * 1e18)::BIGINT AS b,
               random() AS v
        FROM range({rows}) tbl(i)""")
    stop = time.time() + seconds
    while time.time() < stop:
        con.execute("SELECT k % 1000 AS g, sum(a), avg(v), count(*) FROM t GROUP BY g").fetchall()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=64.0, help="TOTAL table footprint across procs (GB)")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--seconds", type=int, default=100000)
    a = ap.parse_args()
    _setcomm("a7duck")

    P = max(1, a.procs)
    per = a.gb / P
    threads = max(1, (os.cpu_count() or P) // P)                 # spread cores; process-dominated
    print(f"[a7-duck] PID={os.getpid()} {a.gb} GB over {P} DuckDB procs "
          f"({per:.2f} GB, {threads} thr each)", flush=True)
    for _ in range(P - 1):
        if os.fork() == 0:                                       # child: its own DuckDB instance
            _setcomm("a7duck")
            hold(per, threads, a.seconds)
            os._exit(0)
    hold(per, threads, a.seconds)                               # parent holds a share + is tree root


if __name__ == "__main__":
    main()
