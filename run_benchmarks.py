"""Run the six-case benchmark register and write benchmark_results.json.

Usage:  python run_benchmarks.py
"""
import datetime
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))

import numpy as np

import beam_dsm
from benchmarks import register, register_example2, summarise


def main():
    records = register(beam_dsm) + register_example2(beam_dsm)
    meta = {
        "executed": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "module_sha256": hashlib.sha256(open(os.path.join(HERE, "beam_dsm.py"), "rb").read()).hexdigest()[:16],
        "numpy": np.__version__,
    }
    with open(os.path.join(HERE, "benchmark_results.json"), "w") as f:
        json.dump({"meta": meta, "records": records}, f, indent=1, default=float)
    for test, case, rep, ok, n, worst in summarise(records):
        print(f"Test {test} | {case[:44]:44s} | {rep[:28]:28s} | {'PASS' if ok else 'FAIL'} | {n:2d} checks | max abs err {worst:.1e}")
    failed = [r for r in records if not r["passed"]]
    print(meta)
    print(f"{len(records)} checks, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
