"""자연스러움 블라인드 테스트 채점.

    python score.py

같은 폴더의 key.csv(정답표)와 results/ 폴더의 결과를 읽는다.
results/ 에는 평가자가 보낸 CSV 파일을 넣거나, 디스코드에 붙여넣은 글을 메모장에 붙여 .csv 또는 .txt 로 저장해 넣는다
(디스코드가 긴 글을 message.txt 파일로 바꿔 보낸 경우도 그대로 넣으면 된다).

출력: 실제/AR 맞히기 정답률(50%에 가까울수록 AR이 진짜와 구분 안 됨), 자연스러움 점수 평균,
AR 사진에서 어색하다고 고른 부위·이유 비율. 목표(평가 기준 v1.1 제안, 검증 전): 정답률 ≤65%, AR 평균 ≥3.5점.
"""
import csv
import io
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def read_results(folder):
    seen = {}
    for f in sorted(list(folder.glob("*.csv")) + list(folder.glob("*.txt"))):
        text = f.read_text(encoding="utf-8-sig", errors="replace")
        start = text.find('"rater"')
        if start < 0:
            start = text.find("rater,")
        if start < 0:
            print(f"  건너뜀(형식 아님): {f.name}")
            continue
        body = text[start:].split("```")[0].strip()
        for r in csv.DictReader(io.StringIO(body)):
            if r.get("image_id"):
                seen[(r["rater"], r["image_id"])] = r  # 같은 사람이 두 번 보내도 한 번만 센다
    return list(seen.values())


def pct(a, b):
    return f"{a / b:.0%}" if b else "-"


def main():
    key = {r["image_id"]: r for r in csv.DictReader(open(HERE / "key.csv", encoding="utf-8-sig"))}
    rows = []
    for r in read_results(HERE / "results"):
        k = key.get(r["image_id"])
        if k:
            rows.append({**r, "truth": k["truth"], "person": k["person"], "score": int(r["score"]),
                         "parts": [p for p in r.get("parts", "").split("|") if p],
                         "reasons": [p for p in r.get("reasons", "").split("|") if p]})
    if not rows:
        raise SystemExit("results/ 폴더에 평가 결과가 없습니다")
    raters = sorted({r["rater"] for r in rows})
    n = len(rows)
    ok = sum(r["answer"] == r["truth"] for r in rows)
    lo, hi = wilson(ok, n)
    ar = [r for r in rows if r["truth"] == "AR"]
    real = [r for r in rows if r["truth"] == "REAL"]
    ar_mean = sum(r["score"] for r in ar) / len(ar) if ar else float("nan")
    print(f"평가자 {len(raters)}명, 판단 {n}번 (AR {len(ar)} · 실제 {len(real)})\n")
    print("[자연스러움]")
    print(f"  맞히기 정답률 {ok / n:.1%} (95% 범위 {lo:.0%}~{hi:.0%})  목표 ≤65% → {'통과' if ok / n <= 0.65 else '미달'}")
    print(f"  AR 평균 {ar_mean:.2f}점 / 실제 평균 {sum(r['score'] for r in real) / max(len(real), 1):.2f}점"
          f"  목표 AR ≥3.5 → {'통과' if ar_mean >= 3.5 else '미달'}")
    print(f"  AR을 실제로 착각한 비율 {pct(sum(r['answer'] == 'REAL' for r in ar), len(ar))}")
    print("\n[어색한 부위·이유 — AR 사진 기준, 괄호는 실제 사진에서 같은 답이 나온 비율(기준선)]")
    for label, field in (("부위", "parts"), ("이유", "reasons")):
        ca, cr = Counter(p for r in ar for p in r[field]), Counter(p for r in real for p in r[field])
        for p in sorted(set(ca) | set(cr), key=lambda x: -ca[x]):
            print(f"  {label} {p:12s} AR {pct(ca[p], len(ar)):>4s}  (실제 {pct(cr[p], len(real))})")
    print("\n[평가자별 정답률]")
    for rt in raters:
        rr = [r for r in rows if r["rater"] == rt]
        print(f"  {rt}: {pct(sum(r['answer'] == r['truth'] for r in rr), len(rr))} ({len(rr)}장)")
    per = defaultdict(list)
    for r in rows:
        per[(r["person"], r["truth"])].append(r)
    out = HERE / "naturalness_by_image.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["person", "truth", "n", "score_mean", "judged_ar", "parts", "reasons"])
        for (p, t), rr in sorted(per.items()):
            w.writerow([p, t, len(rr), round(sum(r["score"] for r in rr) / len(rr), 2),
                        round(sum(r["answer"] == "AR" for r in rr) / len(rr), 2),
                        "|".join(f"{k}:{v}" for k, v in Counter(x for r in rr for x in r["parts"]).most_common()),
                        "|".join(f"{k}:{v}" for k, v in Counter(x for r in rr for x in r["reasons"]).most_common())])
    print(f"\n사진별 결과 → {out.name}")


if __name__ == "__main__":
    main()
