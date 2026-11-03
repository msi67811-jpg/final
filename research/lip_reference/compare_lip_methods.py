"""MediaPipe landmark lips vs BiSeNet lips on the same LaPa 200-image subset, with the BiSeNet adoption verdict.

Reads baseline_landmarks.csv (eval_lapa.py) and bisenet_subset200.csv (eval_bisenet.py), keeps the images both
methods have, and prints one table: IoU, spill, miss, boundary error, boundary F1, closed / open mouth (open =
inner-mouth label > 10% of the lip, as region_metrics.py) and five equal-count skin-lightness groups.

Verdict rule (docs: lapa_metrics/RESULT.md):
  accuracy   BiSeNet IoU is at least 0.03 higher AND its boundary error is at least 20% lower than the landmarks'
  speed      per-frame time on the device at most 33 ms (the phone WASM figure is 862.6 ms; the desktop CPU time
             measured here is shown only as a reference, it is not the device)
Adopt only when both hold; accuracy alone is a "lighter-model research candidate". A server GPU path sends the face
off the device and is not a candidate whatever its accuracy.

    python research/lip_reference/compare_lip_methods.py [--phone-ms 862.6]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "research/data/reference/lapa/eval"
METRICS = ["iou", "spill", "miss", "bnd_mean_rel", "bnd_f1"]
BINS = ["1 dark", "2", "3", "4", "5 light"]


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load():
    lm = {r["name"]: r for r in read(EVAL / "baseline_landmarks.csv") if r["detected"] == "1"}
    bn = {r["name"]: r for r in read(EVAL / "bisenet_subset200.csv")}
    subset = json.loads((EVAL / "subset200.json").read_text())
    rows = []
    for n in subset:
        if n in lm and n in bn:
            r = {"name": n, "open": float(lm[n]["open"]), "lightness": float(lm[n]["lightness"]), "face_ms": float(bn[n]["face_ms"]),
                 "mouth_ms": float(bn[n]["mouth_ms"])}
            for k in METRICS:
                r[f"landmarks_{k}"] = float(lm[n][k])
                r[f"face_{k}"] = float(bn[n][f"face_{k}"])
                r[f"mouth_{k}"] = float(bn[n][f"mouth_{k}"])
            rows.append(r)
    light = np.array([r["lightness"] for r in rows])
    edges = np.nanquantile(light, [0.2, 0.4, 0.6, 0.8])
    for r in rows:
        r["mouth_state"] = "open" if r["open"] > 0.1 else "closed"
        r["skin_bin"] = BINS[int(np.searchsorted(edges, r["lightness"], side="right"))]
    return rows


def mean(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return float(np.nanmean(v))


def median(rows, key):
    return float(np.nanmedian(np.array([r[key] for r in rows], dtype=float)))


def print_table(rows, group=None):
    print(f"{'group':8s} {'method':14s} {'n':>4s} " + " ".join(f"{k:>13s}" for k in METRICS))
    groups = [("all", rows)] if group is None else [(g, [r for r in rows if r[group] == g]) for g in (BINS if group == "skin_bin" else ["closed", "open"])]
    for g, rs in groups:
        for m in ("landmarks", "face", "mouth"):
            name = "landmarks" if m == "landmarks" else f"bisenet-{m}"
            print(f"{g:8s} {name:14s} {len(rs):4d} " + " ".join(f"{mean(rs, f'{m}_{k}'):13.3f}" for k in METRICS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phone-ms", type=float, default=862.6, help="BiSeNet per frame on the phone (WASM), measured earlier")
    a = ap.parse_args()
    rows = load()
    print(f"images with both methods: {len(rows)}\n")
    print_table(rows)
    for g in ("mouth_state", "skin_bin"):
        print(f"\nby {g}")
        print_table(rows, g)
    lm_iou, lm_bnd = mean(rows, "landmarks_iou"), mean(rows, "landmarks_bnd_mean_rel")
    print("\nverdict")
    for tag in ("face", "mouth"):
        d_iou = mean(rows, f"{tag}_iou") - lm_iou
        d_bnd = 1 - mean(rows, f"{tag}_bnd_mean_rel") / lm_bnd
        acc = d_iou >= 0.03 and d_bnd >= 0.20
        ms = float(np.median([r[f"{tag}_ms"] for r in rows]))
        # The mean is pulled by a few failed masks; the median shows the typical image.
        md = median(rows, f"{tag}_bnd_mean_rel") / median(rows, "landmarks_bnd_mean_rel") - 1
        worse = sum(r[f"{tag}_bnd_mean_rel"] > 2 * r["landmarks_bnd_mean_rel"] for r in rows)
        print(f"  bisenet-{tag:5s}  IoU {d_iou:+.3f} (need +0.030)  boundary error {-d_bnd * 100:+.1f}% (need -20%)  "
              f"(median {md * 100:+.1f}%, outline > 2x worse on {worse} of {len(rows)}) "
              f"-> accuracy {'ADVANTAGE' if acc else 'no advantage'};  speed: phone {a.phone_ms:.0f} ms "
              f"(desktop CPU {ms:.0f} ms) vs 33 ms -> {'ok' if a.phone_ms <= 33 else 'NOT met'}")
    print("  adopt only when accuracy AND speed hold; server GPU is off the table (face leaves the device)")


if __name__ == "__main__":
    main()
