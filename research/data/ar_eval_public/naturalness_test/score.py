"""자연스러움 블라인드 테스트 채점.

    python score.py

같은 폴더의 key.csv(정답표)와 results/ 폴더의 결과를 읽는다.
results/ 에는 평가자가 보낸 CSV 파일을 넣거나, 디스코드에 붙여넣은 글을 메모장에 붙여 .csv 또는 .txt 로 저장해 넣는다
(디스코드가 긴 글을 message.txt 파일로 바꿔 보낸 경우도 그대로 넣으면 된다).

출력
1) 실제/AR 맞히기 (같은 부위만 화장한 '쌍' 사진으로만 계산): 정답률이 50%에 가까울수록 AR이 진짜와 구분 안 됨
2) 자연스러움 점수, 어색한 부위·이유 비율
3) 4부위 AR 단계(약·중·강)별 점수와 어색한 부위 비율
4) 컴퓨터 수치와 연결 (같은 폴더에 있으면 자동 사용)
   - zone_change.csv  : 4부위 AR이 맨얼굴에서 부위별로 바뀐 양(ΔE) → 변화량 구간별 "그 부위가 어색" 비율
   - lip_4metrics_by_pair.csv : 립 쌍별 색·질감·위치 수치 → 수치 구간별 "색/경계/질감 어색" 비율
   어색 비율이 급히 오르는 구간이 합격선 후보다.
목표(평가 기준 v1.1 제안, 검증 전): 정답률 ≤65%, AR 평균 ≥3.5점.
"""
import csv
import io
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEVEL = {"1": "약", "2": "중", "3": "강"}


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


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def binned(rows, value, flagged, edges, label):
    """value(r) 구간별로 flagged(r) 비율을 표로 출력."""
    print(f"  {label}")
    bounds = [float("-inf")] + edges + [float("inf")]
    for lo, hi in zip(bounds, bounds[1:]):
        rr = [r for r in rows if value(r) is not None and lo <= value(r) < hi]
        if rr:
            name = f"{'' if lo == float('-inf') else f'{lo:g}'}~{'' if hi == float('inf') else f'{hi:g}'}"
            print(f"    {name:>8s}: 어색 {pct(sum(flagged(r) for r in rr), len(rr)):>4s}  점수 {mean([r['score'] for r in rr]):.2f}  (n={len(rr)})")


def main():
    key = {r["image_id"]: r for r in csv.DictReader(open(HERE / "key.csv", encoding="utf-8-sig"))}
    rows = []
    for r in read_results(HERE / "results"):
        k = key.get(r["image_id"])
        if k:
            rows.append({**r, "truth": k["truth"], "person": k["person"], "kind": k.get("kind") or "pair",
                         "level": k.get("level") or "", "score": int(r["score"]),
                         "parts": [p for p in r.get("parts", "").split("|") if p],
                         "reasons": [p for p in r.get("reasons", "").split("|") if p]})
    if not rows:
        raise SystemExit("results/ 폴더에 평가 결과가 없습니다")
    raters = sorted({r["rater"] for r in rows})
    pair = [r for r in rows if r["kind"] == "pair"]
    full = [r for r in rows if r["kind"] == "full"]
    ar = [r for r in pair if r["truth"] == "AR"]
    real = [r for r in pair if r["truth"] == "REAL"]
    print(f"평가자 {len(raters)}명, 판단 {len(rows)}번 (쌍 사진 {len(pair)}: 실제 {len(real)}·AR {len(ar)} / 4부위 AR {len(full)})\n")

    print("[1. 실제/AR 맞히기 — 쌍 사진만]")
    if pair:
        ok = sum(r["answer"] == r["truth"] for r in pair)
        lo, hi = wilson(ok, len(pair))
        print(f"  정답률 {ok / len(pair):.1%} (95% 범위 {lo:.0%}~{hi:.0%})  목표 ≤65% → {'통과' if ok / len(pair) <= 0.65 else '미달'}")
        print(f"  AR을 실제로 착각한 비율 {pct(sum(r['answer'] == 'REAL' for r in ar), len(ar))}")
        if full:
            print(f"  (참고) 4부위 AR을 실제로 착각한 비율 {pct(sum(r['answer'] == 'REAL' for r in full), len(full))}")

    print("\n[2. 자연스러움 점수]")
    print(f"  실제 {mean([r['score'] for r in real]):.2f}점 / 쌍 AR {mean([r['score'] for r in ar]):.2f}점"
          f"  목표 AR ≥3.5 → {'통과' if mean([r['score'] for r in ar]) >= 3.5 else '미달'}")
    print("  어색한 부위·이유 (쌍 AR, 괄호는 실제 사진 기준선)")
    for label, field in (("부위", "parts"), ("이유", "reasons")):
        ca, cr = Counter(p for r in ar for p in r[field]), Counter(p for r in real for p in r[field])
        for p in sorted(set(ca) | set(cr), key=lambda x: -ca[x]):
            print(f"    {label} {p:12s} AR {pct(ca[p], len(ar)):>4s}  (실제 {pct(cr[p], len(real))})")

    if full:
        print("\n[3. 4부위 AR 단계별]")
        parts = sorted({p for r in full for p in r["parts"]} - {"없음"})
        print("    단계   n  점수   없음  " + "  ".join(f"{p:>3s}" for p in parts))
        for lv in ("1", "2", "3"):
            rr = [r for r in full if r["level"] == lv]
            if rr:
                print(f"    {LEVEL[lv]:2s} {len(rr):4d}  {mean([r['score'] for r in rr]):.2f}  {pct(sum('없음' in r['parts'] for r in rr), len(rr)):>4s}  "
                      + "  ".join(f"{pct(sum(p in r['parts'] for r in rr), len(rr)):>4s}" for p in parts))
        groups = {"일반 사진 얼굴(p)": [r for r in full if r["person"].startswith("p")],
                  "스튜디오 얼굴(n)": [r for r in full if r["person"].startswith("n")]}
        if all(groups.values()):
            print("    사진 출처별 (스튜디오 얼굴이 AI 생성이면 점수가 얼굴 자체 때문에 달라질 수 있어 따로 본다)")
            for name, rr in groups.items():
                print(f"      {name}: n={len(rr)}  점수 {mean([r['score'] for r in rr]):.2f}  어색 없음 {pct(sum('없음' in r['parts'] for r in rr), len(rr))}")

    zc = HERE / "zone_change.csv"
    lm = HERE / "lip_4metrics_by_pair.csv"
    if (full and zc.exists()) or (ar and lm.exists()):
        print("\n[4. 컴퓨터 수치와 연결 — 어색 비율이 급히 오르는 구간이 합격선 후보]")
    if full and zc.exists():
        z = {(r["person"], r["level"]): r for r in csv.DictReader(open(zc, encoding="utf-8-sig"))}
        for r in full:
            r["z"] = z.get((r["person"], r["level"]))
        fz = [r for r in full if r["z"]]
        val = lambda col: (lambda r: float(r["z"][col]))
        binned(fz, val("lip"), lambda r: "입술" in r["parts"], [4, 6, 8, 10, 12], "립 변화 ΔE → '입술' 어색")
        binned(fz, val("blush_add"), lambda r: "볼" in r["parts"], [1, 2, 3, 4, 6], "블러셔가 더한 변화 → '볼' 어색")
        binned(fz, val("eye_add"), lambda r: "눈" in r["parts"], [1, 2, 3, 4, 6], "아이섀도가 더한 변화 → '눈' 어색")
        binned(fz, val("forehead"), lambda r: "피부" in r["parts"], [2, 3, 4, 6], "파운데이션 변화(이마) → '피부' 어색")
    if ar and lm.exists():
        m = {r["pair"]: r for r in csv.DictReader(open(lm, encoding="utf-8-sig"))}
        la = [r for r in ar if r["person"] in m]
        binned(la, lambda r: float(m[r["person"]]["dE2000"]), lambda r: "색" in r["reasons"], [2, 4, 6, 10], "립 색 차이 ΔE2000 → '색' 어색")
        binned(la, lambda r: float(m[r["person"]]["ssim"]), lambda r: "질감·광택" in r["reasons"], [0.55, 0.65, 0.75], "립 결 유사도 SSIM → '질감' 어색")
        binned(la, lambda r: float(m[r["person"]]["dL_hi"]), lambda r: "질감·광택" in r["reasons"], [4, 8, 12], "하이라이트 차이 dL_hi → '질감' 어색")

    print("\n[평가자별 정답률 (쌍 사진)]")
    for rt in raters:
        rr = [r for r in pair if r["rater"] == rt]
        print(f"  {rt}: {pct(sum(r['answer'] == r['truth'] for r in rr), len(rr))} ({len(rr)}장)")

    per = defaultdict(list)
    for r in rows:
        per[(r["person"], r["kind"], r["truth"], r["level"])].append(r)
    out = HERE / "naturalness_by_image.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["person", "kind", "truth", "level", "n", "score_mean", "judged_ar", "parts", "reasons"])
        for (p, kd, t, lv), rr in sorted(per.items()):
            w.writerow([p, kd, t, lv, len(rr), round(mean([r["score"] for r in rr]), 2),
                        round(sum(r["answer"] == "AR" for r in rr) / len(rr), 2),
                        "|".join(f"{k}:{v}" for k, v in Counter(x for r in rr for x in r["parts"]).most_common()),
                        "|".join(f"{k}:{v}" for k, v in Counter(x for r in rr for x in r["reasons"]).most_common())])
    out2 = HERE / "responses_all.csv"
    with open(out2, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rater", "image_id", "person", "kind", "truth", "level", "answer", "score", "parts", "reasons", "ms"])
        for r in rows:
            w.writerow([r["rater"], r["image_id"], r["person"], r["kind"], r["truth"], r["level"], r["answer"], r["score"],
                        "|".join(r["parts"]), "|".join(r["reasons"]), r.get("ms", "")])
    print(f"\n사진별 결과 → {out.name}, 전체 응답(정답 포함) → {out2.name}")


if __name__ == "__main__":
    main()
