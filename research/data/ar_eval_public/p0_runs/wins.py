"""Summarize hw01_compare.csv files: medians and how many pairs AR beats the bare lip on each metric.

    python wins.py base_compare.csv [p01_compare.csv ...]
"""
import csv
import sys
from collections import defaultdict
from statistics import median

for path in sys.argv[1:]:
    by = defaultdict(dict)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            by[r["name"]][r["method"]] = {k: float(r[k]) for k in ("dE", "dL_spread", "dL_hi", "ssim")}
    pairs = [v for v in by.values() if "ar" in v and "bare" in v]
    n = len(pairs)
    win = lambda k, better: sum(better(p["ar"][k], p["bare"][k]) for p in pairs)
    print(f"{path}  pairs {n}")
    print(f"  ar dE median {median(p['ar']['dE'] for p in pairs):.2f}  (bare {median(p['bare']['dE'] for p in pairs):.2f})")
    print(f"  wins vs bare: spread {win('dL_spread', lambda a, b: a < b)}/{n}  "
          f"highlight {win('dL_hi', lambda a, b: a < b)}/{n}  ssim {win('ssim', lambda a, b: a > b)}/{n}")
    print(f"  ar medians: spread {median(p['ar']['dL_spread'] for p in pairs):.2f}  "
          f"hi {median(p['ar']['dL_hi'] for p in pairs):.2f}  ssim {median(p['ar']['ssim'] for p in pairs):.3f}")
