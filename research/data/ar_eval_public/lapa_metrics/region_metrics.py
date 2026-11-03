"""LaPa 200장: 앱 얼굴 점(MediaPipe 478점) 기준 부위 영역 vs LaPa 정답 마스크.

부위: 립(7+9), 아이브로우(2+3), 눈(4+5, 아이라인 기준선), 파운데이션(피부 1 + 코 6).
얼굴 점은 eval_lapa.py dump_subset 이 저장한 subset200_landmarks.json 을 그대로 쓴다(앱과 같은 모델).
앱 렌더 결과가 아니라 '앱이 화장을 놓는 기준 영역'의 정확도다. 파운데이션은 앱이 머리카락 경계를
분할 모델로 한 번 더 자르므로 여기 값보다 넘침이 적을 수 있다. 블러셔·섀도는 정답 마스크가 없어 제외.
LaPa 는 비상업 연구용: research/data/ 안에서만 계산하고 이미지는 밖으로 보내지 않는다.

    python3 research/data/ar_eval_public/lapa_metrics/region_metrics.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
LAPA = ROOT / "research/data/reference/lapa"
OUT = Path(__file__).resolve().parent
LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
LIPS_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
BROW_R = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]          # faceTrackerHtml.ts BROWS 앞 10개
BROW_L = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]  # 뒤 10개
EYE_R = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
EYE_L = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
             148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def poly(pts, idx, shape):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(pts[idx]).astype(np.int32)], 1)
    return m.astype(bool)


def score(pred, gt):
    inter, union = (pred & gt).sum(), (pred | gt).sum()
    return {"iou": inter / max(union, 1), "spill": (pred & ~gt).sum() / max(pred.sum(), 1),
            "miss": (gt & ~pred).sum() / max(gt.sum(), 1)}


def frontal(p):
    t = (p[1][0] - p[33][0]) / max(p[263][0] - p[33][0], 1e-6)
    return 1.0 - min(abs(t - 0.5) * 4.0, 1.0)


def main():
    lms = json.loads((LAPA / "eval/subset200_landmarks.json").read_text())
    rows = []
    for name, P in lms.items():
        P = np.array(P)
        img = cv2.imread(str(LAPA / f"LaPa/test/images/{name}.jpg"))
        lab = cv2.imread(str(LAPA / f"LaPa/test/labels/{name}.png"), cv2.IMREAD_UNCHANGED)
        lab = lab[..., 0] if lab.ndim == 3 else lab
        sh = lab.shape
        lip = poly(P, LIPS_OUTER, sh) & ~poly(P, LIPS_INNER, sh)
        brow = poly(P, BROW_R, sh) | poly(P, BROW_L, sh)
        eye = poly(P, EYE_R, sh) | poly(P, EYE_L, sh)
        mouth = poly(P, LIPS_OUTER, sh)
        face = poly(P, FACE_OVAL, sh) & ~brow & ~eye & ~mouth
        gt_lip = (lab == 7) | (lab == 9)
        skin = lab == 1
        Lc = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
        r = {"name": name, "lightness": float(Lc[skin].mean() * 100 / 255) if skin.sum() > 200 else np.nan,
             "open": float((lab == 8).sum() / max(gt_lip.sum(), 1)), "frontal": frontal(P)}
        for k, pred, gt in [("lip", lip, gt_lip), ("brow", brow, (lab == 2) | (lab == 3)),
                            ("eye", eye, (lab == 4) | (lab == 5)), ("found", face, (lab == 1) | (lab == 6))]:
            if gt.sum() < 50:
                continue
            for m, v in score(pred, gt).items():
                r[f"{k}_{m}"] = float(v)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lapa200_region_metrics.csv", index=False)
    df["skin_bin"] = pd.qcut(df["lightness"], 5, labels=["1어두움", "2", "3", "4", "5밝음"])
    df["mouth"] = np.where(df["open"] > 0.1, "벌림", "다묾")
    summary = {"n_images": len(df)}
    for k in ["lip", "brow", "eye", "found"]:
        s = df.dropna(subset=[f"{k}_iou"])
        g = s.groupby("skin_bin", observed=True)[f"{k}_iou"].mean()
        summary[k] = {"n": len(s), "iou_mean": s[f"{k}_iou"].mean(), "iou_median": s[f"{k}_iou"].median(),
                      "iou_p10": s[f"{k}_iou"].quantile(0.1), "spill_mean": s[f"{k}_spill"].mean(),
                      "miss_mean": s[f"{k}_miss"].mean(), "iou_by_skin": g.round(3).to_dict(),
                      "worst_skin_gap": float(s[f"{k}_iou"].mean() - g.min())}
        if k == "lip":
            summary[k]["iou_by_mouth"] = s.groupby("mouth")["lip_iou"].mean().round(3).to_dict()
    b = pd.read_csv(LAPA / "eval/baseline_landmarks.csv")
    d = b[b.detected == 1]
    summary["lip_2000"] = {"rows": len(b), "detect_rate": float(b.detected.mean()), "iou_mean": d.iou.mean(),
                           "iou_median": d.iou.median(), "spill_mean": d.spill.mean(), "miss_mean": d.miss.mean(),
                           "iou_closed": d[d.open <= 0.1].iou.mean(), "iou_open": d[d.open > 0.1].iou.mean()}
    (OUT / "lapa200_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    main()
