#!/usr/bin/env python3
"""Regenerate every statistic and every figure in the manuscript, in order.

    python3 make_all.py            # everything
    python3 make_all.py --figures  # skip run_analysis.py

Each step runs in its own process, so a script that crashes cannot leave a
half-written PNG behind from an earlier run and pass it off as current; the
driver stops on the first non-zero exit.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = [
    ("run_analysis.py", "statistics -> ../results/"),
    ("fig1_processing.py", "Fig 1  the processing chain"),
    ("fig2_record.py", "Fig 2  the CONUS record and its trend"),
    ("fig3_decomposition.py", "Fig 3  frequency against intensity"),
    ("fig4_maps.py", "Fig 4  spatial distribution and trend"),
    ("fig5_season.py", "Fig 5  seasonal timing"),
    ("fig6_validation.py", "Fig 6  against NOAA HMS heavy smoke"),
]


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    steps = [s for s in STEPS
             if not (argv and "--figures" in argv and s[0] == "run_analysis.py")]
    for script, what in steps:
        print(f"\n{script}  —  {what}")
        t0 = time.time()
        r = subprocess.run([sys.executable, "-W", "ignore", str(HERE / script)],
                           cwd=HERE)
        if r.returncode != 0:
            print(f"\nFAILED: {script} exited {r.returncode}", file=sys.stderr)
            return r.returncode
        print(f"  ({time.time() - t0:.0f} s)")
    print("\nAll steps completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
