"""Off-testbed smoke test: proves the harness wires up and produces a RunRecord
even with NO telemetry sources present. Measures a trivial sleep op.

    python scripts/smoke.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import measure_operation               # noqa: E402
from _common import build_telemetry, print_record   # noqa: E402


def main() -> None:
    tele = build_telemetry()
    tele.start()
    try:
        rec = measure_operation(
            tele, workload="smoke", operation="sleep", state_bytes=0,
            op=lambda: time.sleep(0.5),
            config={"note": "off-testbed wiring check"},
            baseline_seconds=1.0, settle_seconds=0.2,
        )
        print_record(rec)
        assert rec.outcome == "ok" and rec.latency_s >= 0.5
        print("\n[smoke] OK -- harness wiring is sound.")
    finally:
        tele.stop()


if __name__ == "__main__":
    main()
