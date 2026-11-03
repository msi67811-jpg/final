"""자연스러움 블라인드 테스트 채점.

    python score.py            # 같은 폴더의 key.csv 와 results/*.csv 를 읽는다

맞히기 정답률(50%에 가까울수록 AR이 진짜와 구분되지 않음)과 1~5점 평균을 계산한다.
목표(평가 기준 v1.1 제안, 검증 전): AR 평균 ≥3.5점, 정답률 ≤65%.
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    key = {(r["set"], r["image_id"]): r for r in csv.DictReader(open(HERE / "key.csv", encoding="utf-8-sig"))}
    rows = []
    for f in sorted((HERE / "results").glob("*.csv")):
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            k = key.get((r["set"], r["image_id"]))
            if k:
                rows.append({**r, "truth": k["truth"], "pair": k["pair"], "score": int(r["score"])})
    if not rows:
        raise SystemExit("results/ 폴더에 평가 CSV가 없습니다")
    raters = sorted({r["rater"] for r in rows})
    ok = sum(r["answer"] == r["truth"] for r in rows)
    lo, hi = wilson(ok, len(rows))
    print(f"평가자 {len(raters)}명, 판단 {len(rows)}번")
    print(f"맞히기 정답률 {ok / len(rows):.1%} (95% 범위 {lo:.1%}~{hi:.1%}) → 목표 ≤65%: {'통과' if ok / len(rows) <= 0.65 else '미달'}")
    ar_hit = [r["answer"] == "AR" for r in rows if r["truth"] == "AR"]
    print(f"AR 사진을 AR로 맞힌 비율 {sum(ar_hit) / len(ar_hit):.1%}")
    for t in ("REAL", "AR"):
        s = [r["score"] for r in rows if r["truth"] == t]
        print(f"{t} 자연스러움 평균 {sum(s) / len(s):.2f}점 (n={len(s)})")
    ar = [r["score"] for r in rows if r["truth"] == "AR"]
    print(f"→ AR 목표 ≥3.5점: {'통과' if sum(ar) / len(ar) >= 3.5 else '미달'}")
    print("\n평가자별 정답률:")
    for rt in raters:
        rr = [r for r in rows if r["rater"] == rt]
        print(f"  {rt} ({rr[0]['set']} 세트): {sum(r['answer'] == r['truth'] for r in rr) / len(rr):.0%}")
    per = defaultdict(list)
    for r in rows:
        if r["truth"] == "AR":
            per[r["pair"]].append(r)
    out = HERE / "naturalness_by_pair.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["pair", "n", "ar_score_mean", "detected_as_ar"])
        for p, rr in sorted(per.items()):
            w.writerow([p, len(rr), round(sum(r["score"] for r in rr) / len(rr), 2),
                        round(sum(r["answer"] == "AR" for r in rr) / len(rr), 2)])
    print(f"\n쌍별 AR 결과 → {out.name}")


if __name__ == "__main__":
    main()
