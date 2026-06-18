#!/usr/bin/env python3
"""A8 -- multi-THREAD real workload: DuckDB in-memory analytics. Builds a large in-memory table
(footprint dialed by row count; compression forced OFF so RSS is predictable ~gb) and loops a
multi-threaded GROUP BY aggregation. One process, DuckDB's thread pool spans the cores -- the
many-TID / single-address-space criu case. comm 'a8duck' for the driver.

    python scripts/work_duck.py --gb 64 --threads 64 --seconds 100000
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=64.0, help="approx in-memory table footprint (GB)")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--seconds", type=int, default=100000)
    a = ap.parse_args()
    _setcomm("a8duck")

    import duckdb
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={a.threads}")
    con.execute("PRAGMA memory_limit='700GB'")
    con.execute("PRAGMA force_compression='Uncompressed'")   # predictable RSS (no int-seq packing)
    rows = int(a.gb * 1e9 / 40)                              # 5 cols x 8 bytes ~ 40 B/row
    print(f"[a8-duck] PID={os.getpid()} building {rows} rows (~{a.gb} GB), {a.threads} threads",
          flush=True)
    con.execute(f"""CREATE TABLE t AS
        SELECT i AS id,
               (random() * 1e18)::BIGINT AS k,
               (random() * 1e18)::BIGINT AS a,
               (random() * 1e18)::BIGINT AS b,
               random() AS v
        FROM range({rows}) tbl(i)""")                       # random -> incompressible -> real bytes
    print("[a8-duck] table built, looping aggregation query", flush=True)

    stop = time.time() + a.seconds
    while time.time() < stop:                               # multi-threaded scan+aggregate
        con.execute("SELECT k % 1000 AS g, sum(a), avg(v), count(*) FROM t GROUP BY g").fetchall()


if __name__ == "__main__":
    main()
