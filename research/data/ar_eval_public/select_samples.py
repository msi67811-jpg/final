"""AR 평가용 공개 데이터셋 샘플 선별 (립 기준, 연구·내부 평가 전용).

공개 데이터셋에서 내려받은 얼굴 이미지 중 AR 평가 조건에 맞는 20개를 고르고,
출처·라이선스 열이 있는 manifest.csv 와 render_ar_offline.py 가 읽는 hw01 이름으로 복사본을 만든다.
원본 파일은 수정·이동하지 않는다. 결과물은 research/data/ (git 제외) 안에만 둔다.

지원하는 입력
  --source ffhq-makeup  <root>/<id>/bare.jpg + makeup_01..05.jpg   (같은 사람 BEFORE/AFTER 쌍, AFTER는 합성)
  --source flat-bare    <root>/*.jpg|png  맨얼굴 후보 (예: LaPa test/images)
  --source flat-real    <root>/*.jpg|png  실제 화장 사진 후보 (예: 허가 확인한 Commons 사진)

    python select_samples.py --source ffhq-makeup --root raw/ffhq_makeup --n 20 --out selected
    python select_samples.py --source flat-bare --root ../reference/lapa/LaPa/test/images --n 20 --out selected_bare

자동 판정(정면·입 다묾·입술 크기·피부 밝기·입술 색 계열·광택)은 후보를 줄이는 용도다.
맨입술도 채도가 높아(시험 사진에서 입술-볼 채도 차 약 18) 색만으로 '화장 안 함'을 판정할 수 없다.
화장 여부, 미성년자로 보이는지, 워터마크 등은 반드시 사람이 최종 확인한다(manifest 의 reviewed 열).
"""
import argparse
import csv
import math
import random
import shutil
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE.parent / "models" / "face_landmarker.task"
OUTER_LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
INNER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
CHEEKS = [50, 280]  # 왼·오른 광대 아래 볼
EXTS = {".jpg", ".jpeg", ".png"}

LICENSES = {
    "ffhq-makeup": ("FFHQ-Makeup (cyberagent)", "CC BY-NC-SA 4.0 (FFHQ 상속)",
                    "https://huggingface.co/datasets/cyberagent/FFHQ-Makeup", "AFTER 이미지는 확산 모델 합성"),
    "flat-bare": ("", "", "", ""),
    "flat-real": ("", "", "", ""),
}


def landmarker(model: Path):
    opts = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model)),
                                 running_mode=RunningMode.IMAGE, num_faces=1)
    return FaceLandmarker.create_from_options(opts)


def lab(img_bgr):
    f = cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    return f  # L 0..100, a/b ±127


def poly_mask(shape, pts):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 255)
    return m


def measure(lm_model, path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return {"ok": False, "why": "읽기 실패"}
    h, w = img.shape[:2]
    res = lm_model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    if not res.face_landmarks:
        return {"ok": False, "why": "얼굴 없음"}
    P = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
    d = lambda i, j: float(np.linalg.norm(P[i] - P[j]))
    mouth_w = d(61, 291)
    yaw_ratio = d(1, 234) / max(d(1, 454), 1e-6)          # 1 근처 = 정면
    mouth_open = d(13, 14) / max(mouth_w, 1e-6)          # 0 근처 = 다문 입
    lip_frac = mouth_w / min(h, w)                         # 화면 짧은 변 대비 입 너비
    L = lab(img)
    lips = (poly_mask(img.shape, P[OUTER_LIPS]) > 0) & ~(poly_mask(img.shape, P[INNER_LIPS]) > 0)
    px = L[lips]
    if len(px) < 50:
        return {"ok": False, "why": "입술 영역 너무 작음"}
    hl_cut = np.percentile(px[:, 0], 90)
    body = px[px[:, 0] <= hl_cut]                            # 반사광 제외한 입술 색
    lip_L, lip_a, lip_b = body.mean(0)
    lip_C = math.hypot(lip_a, lip_b)
    lip_h = math.degrees(math.atan2(lip_b, lip_a)) % 360
    gloss = float((px[:, 0] > np.median(px[:, 0]) + 15).mean())  # 참고값(검증 전)
    r = max(3, int(mouth_w * 0.12))
    cheek = np.concatenate([L[max(0, int(y) - r):int(y) + r, max(0, int(x) - r):int(x) + r].reshape(-1, 3)
                            for x, y in P[CHEEKS]])
    sk_L, sk_a, sk_b = cheek.mean(0)
    ita = math.degrees(math.atan2(sk_L - 50, sk_b))
    rgb = cv2.cvtColor(np.array([[[lip_L, lip_a, lip_b]]], np.float32), cv2.COLOR_LAB2RGB)[0, 0]
    return {"ok": True, "w": w, "h": h, "yaw_ratio": yaw_ratio, "mouth_open": mouth_open, "lip_frac": lip_frac,
            "lip_L": lip_L, "lip_C": lip_C, "lip_h": lip_h, "lip_hex": "#%02X%02X%02X" % tuple(int(round(c * 255)) for c in np.clip(rgb, 0, 1)),
            "lip_minus_skin_C": lip_C - math.hypot(sk_a, sk_b), "gloss_frac": gloss, "ita": ita, "lab": (lip_L, lip_a, lip_b)}


def de2000(c1, c2):
    from skimage.color import deltaE_ciede2000
    return float(deltaE_ciede2000(np.array([c1]), np.array([c2]))[0])


def color_family(m):
    if m["lip_C"] < 22:
        return "nude"
    if m["lip_h"] < 18 or m["lip_h"] > 330:
        return "pink"
    if m["lip_h"] < 32:
        return "red" if m["lip_L"] < 52 else "rose"
    return "coral"


def skin_group(ita):
    return "light" if ita > 41 else ("mid" if ita > 10 else "deep")


def passes(m, a):
    if not m["ok"]:
        return m["why"]
    if not (1 / a.max_yaw <= m["yaw_ratio"] <= a.max_yaw):
        return "정면 아님"
    if m["mouth_open"] > a.max_open:
        return "입 벌림"
    if m["lip_frac"] < a.min_lip:
        return "입술 작음"
    return ""


def stratified(cands, n, seed):
    rnd = random.Random(seed)
    groups = {}
    for c in cands:
        groups.setdefault((c["skin_group"], c["family"]), []).append(c)
    for g in groups.values():
        rnd.shuffle(g)
    keys = sorted(groups)
    out = []
    while len(out) < n and any(groups[k] for k in keys):
        for k in keys:
            if groups[k] and len(out) < n:
                out.append(groups[k].pop())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=list(LICENSES))
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--max-yaw", type=float, default=1.35, help="좌우 얼굴 폭 비율 허용 (1=정면)")
    ap.add_argument("--max-open", type=float, default=0.08, help="입 벌림 비율 허용")
    ap.add_argument("--min-lip", type=float, default=0.15, help="입 너비 / 짧은 변 최소")
    ap.add_argument("--min-lip-change", type=float, default=4.0, help="쌍에서 입술 색 변화 최소 ΔE00")
    ap.add_argument("--dataset", default="", help="flat 입력의 데이터셋 이름")
    ap.add_argument("--license", default="", help="flat 입력의 라이선스")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    ds, lic, url, note = LICENSES[a.source]
    ds, lic = a.dataset or ds, a.license or lic
    lmk = landmarker(a.model)
    cands, rejected = [], {}

    if a.source == "ffhq-makeup":
        for sub in sorted(p for p in a.root.iterdir() if p.is_dir()):
            bare = sub / "bare.jpg"
            if not bare.exists():
                continue
            mb = measure(lmk, bare)
            why = passes(mb, a)
            if why:
                rejected[why] = rejected.get(why, 0) + 1
                continue
            best = None
            for mk in sorted(sub.glob("makeup_*.jpg")):
                mm = measure(lmk, mk)
                if not mm["ok"]:
                    continue
                change = de2000(mb["lab"], mm["lab"])
                if change >= a.min_lip_change and (best is None or change > best[2]):
                    best = (mk, mm, change)
            if best is None:
                rejected["입술 화장 변화 작음"] = rejected.get("입술 화장 변화 작음", 0) + 1
                continue
            mk, mm, change = best
            cands.append({"id": sub.name, "bare": bare, "real": mk, "m": mb, "mr": mm, "lip_change": change,
                          "skin_group": skin_group(mb["ita"]), "family": color_family(mm)})
    else:
        for f in sorted(p for p in a.root.rglob("*") if p.suffix.lower() in EXTS):
            m = measure(lmk, f)
            why = passes(m, a)
            if why:
                rejected[why] = rejected.get(why, 0) + 1
                continue
            role = "bare" if a.source == "flat-bare" else "real"
            cands.append({"id": f.stem, role: f, "m": m, "skin_group": skin_group(m["ita"]),
                          "family": color_family(m) if role == "real" else "bare"})

    picked = stratified(cands, a.n, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, c in enumerate(picked, 1):
        sid = f"e{i:02d}"
        row = {"sample_id": sid, "dataset": ds, "license": lic, "source_url": url, "source_id": c["id"],
               "attribution": "", "pair": "yes" if "bare" in c and "real" in c else "no",
               "skin_group": c["skin_group"], "ita": round(c["m"]["ita"], 1), "family": c["family"],
               "yaw_ratio": round(c["m"]["yaw_ratio"], 2), "mouth_open": round(c["m"]["mouth_open"], 3),
               "lip_frac": round(c["m"]["lip_frac"], 3), "lip_skin_dC": round(c["m"]["lip_minus_skin_C"], 1), "note": note, "reviewed": "", "use": ""}
        for role in ("bare", "real"):
            if role in c:
                dst = a.out / f"{sid}_{role}{c[role].suffix.lower()}"
                shutil.copy2(c[role], dst)
                row[f"{role}_file"] = dst.name
                row[f"{role}_source_file"] = str(c[role])
        if "mr" in c:
            row.update(real_lip_hex=c["mr"]["lip_hex"], real_gloss=round(c["mr"]["gloss_frac"], 3),
                       lip_change_de00=round(c["lip_change"], 1))
        elif "real" in c:
            row.update(real_lip_hex=c["m"]["lip_hex"], real_gloss=round(c["m"]["gloss_frac"], 3))
        rows.append(row)
    cols = ["sample_id", "pair", "bare_file", "real_file", "dataset", "license", "source_url", "source_id",
            "bare_source_file", "real_source_file", "attribution", "skin_group", "ita", "family", "real_lip_hex",
            "real_gloss", "lip_change_de00", "yaw_ratio", "mouth_open", "lip_frac", "lip_skin_dC", "note", "reviewed", "use"]
    with open(a.out / "manifest.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in cols})
    print(f"후보 {len(cands)}개 중 {len(picked)}개 선택 → {a.out / 'manifest.csv'}")
    print("제외 사유:", rejected or "없음")
    print("그룹 분포:", {k: sum(1 for c in picked if (c['skin_group'], c['family']) == k)
                       for k in sorted({(c['skin_group'], c['family']) for c in picked})})


if __name__ == "__main__":
    main()
