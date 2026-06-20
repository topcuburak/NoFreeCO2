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


QUERY = "SELECT k % 1000 AS g, sum(a), avg(v), count(*) FROM t GROUP BY g"


def build_con(gb: float, threads: int):
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
    return con


def hold(gb: float, threads: int, seconds: int) -> None:
    con = build_con(gb, threads)
    stop = time.time() + seconds
    while time.time() < stop:
        con.execute(QUERY).fetchall()


def job_worker(idx: int, gb: float, threads: int, queries: int, readydir: str, trig: str) -> None:
    _setcomm("a7duck")
    con = build_con(gb, threads)                                # setup
    open(os.path.join(readydir, str(idx)), "w").close()        # signal this child is built
    while trig and not os.path.exists(trig):                   # barrier: wait for the job trigger
        time.sleep(0.05)
    for _ in range(queries):                                   # the fixed job
        con.execute(QUERY).fetchall()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=64.0, help="TOTAL table footprint across procs (GB)")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--seconds", type=int, default=100000)
    ap.add_argument("--job-queries", type=int, default=0,
                    help="dump-free baseline: each proc builds, barrier, runs N queries, exit")
    a = ap.parse_args()
    _setcomm("a7duck")

    P = max(1, a.procs)
    per = a.gb / P
    threads = max(1, (os.cpu_count() or P) // P)                 # spread cores; process-dominated
    print(f"[a7-duck] PID={os.getpid()} {a.gb} GB over {P} DuckDB procs "
          f"({per:.2f} GB, {threads} thr each)", flush=True)

    if a.job_queries > 0:                                        # dump-free baseline (parent coordinates)
        import shutil
        readydir = "/tmp/a7_ready"
        shutil.rmtree(readydir, ignore_errors=True); os.makedirs(readydir)
        trig = os.environ.get("RUNJOB_TRIGGER", "")
        kids = []
        for i in range(P):
            pid = os.fork()
            if pid == 0:
                job_worker(i, per, threads, a.job_queries, readydir, trig)
                os._exit(0)
            kids.append(pid)
        while len(os.listdir(readydir)) < P:                     # wait until all procs built (setup)
            time.sleep(0.1)
        print("RUNJOB_READY", flush=True)                       # job_energy now creates the trigger
        for pid in kids:
            os.waitpid(pid, 0)                                  # job runs until every proc exits
        print(f"RUNJOB_DONE queries={a.job_queries} procs={P}", flush=True)
        return

    for _ in range(P - 1):
        if os.fork() == 0:                                       # child: its own DuckDB instance
            _setcomm("a7duck")
            hold(per, threads, a.seconds)
            os._exit(0)
    hold(per, threads, a.seconds)                               # parent holds a share + is tree root


if __name__ == "__main__":
    main()
