"""Foundation with a skin mask injected: how the paint looks when the mask is a few frames old (offline, app shader).

The usual offline renders have no skin mask (the page falls back to a brightness gate). Here each job brings its own mask
(a 256x256 selfie-multiclass skin mask, like the page's `foundSkinRgba`) and the lower-face cover value `foundOcc`, so the
mask path of the shader is exercised: a FRESH mask of the frame (the reference) and STALE masks of k frames before it.

    python render_found_mask.py --clip CLIP.mp4 --frames 16,100,104 --ks 4,12 --page PAGE.ts --out DIR [--tag new]

Writes DIR/<tag>_f<frame>_k<k>.png (k=0 is the fresh mask). The page works on a copy of its source text, so any file whose
content is the page (.ts or .ts.txt) can be given. Local use only; nothing leaves the PC.
"""
import argparse, base64, csv, functools, json, re, shutil, subprocess, sys, threading
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "research/lip_reference"))
sys.path.insert(0, str(HERE))
import render_ar_offline as r  # noqa: E402
import eval_foundation_lag as E  # noqa: E402
import mediapipe as mp  # noqa: E402
from mediapipe.tasks import python as mpp  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

OLD = "makeup = { lip: { hex: job.hex, intensity: job.intensity, texture: job.texture }, blush: null, eyeshadow: null };"
NEW = ("makeup = { lip: null, blush: null, eyeshadow: null, eyeliner: null, mascara: null, highlighter: null, contour: null, brow: null, "
       "foundation: { hex: job.hex, intensity: job.intensity, texture: 'matte' } };")
UPLOAD = "      renderer.upload();\n      for (let k = 0; k < 4; k++) renderer.render(W, H);"
WITH_MASK = """      renderer.upload();
      {
        const mimg = await load(job.mask);
        const mc = document.createElement('canvas'); mc.width = job.mw; mc.height = job.mh;
        const mctx = mc.getContext('2d'); mctx.drawImage(mimg, 0, 0, job.mw, job.mh);
        renderer.uploadFoundSkin(new Uint8Array(mctx.getImageData(0, 0, job.mw, job.mh).data.buffer), job.mw, job.mh);
        foundSkinReady = true;
        if (typeof foundOcc !== 'undefined') foundOcc = job.occ;   // the page before the lower-face floor has no such variable
      }
      for (let k = 0; k < 4; k++) renderer.render(W, H);"""
# Cheek / chin landmarks used for the cover check (same list as the page's FOUND_COVER_PTS).
COVER = [1, 2, 98, 327, 129, 358, 203, 423, 205, 425, 50, 280, 101, 330, 118, 347, 117, 346, 187, 411, 207, 427, 147, 376, 213, 433, 200, 199, 175, 396]


def smooth(lo, hi, x):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--frames", default="16,100,104")
    ap.add_argument("--ks", default="4,12")
    ap.add_argument("--page", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default="new")
    ap.add_argument("--hex", default="#2E9E4F")
    a = ap.parse_args()
    frames = [int(x) for x in a.frames.split(",")]
    ks = [0] + [int(x) for x in a.ks.split(",")]
    a.out.mkdir(parents=True, exist_ok=True)
    work = a.out / f"_work_{a.tag}"
    shutil.rmtree(work, ignore_errors=True); work.mkdir()

    need = sorted({f - k for f in frames for k in ks})
    cap = cv2.VideoCapture(str(a.clip)); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(E.FACE_MODEL)), running_mode=vision.RunningMode.VIDEO, num_faces=1))
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=mpp.BaseOptions(model_asset_path=str(E.SEG_MODEL)), running_mode=vision.RunningMode.VIDEO,
        output_category_mask=True, output_confidence_masks=True))
    img_of, lm_of, mask_of = {}, {}, {}
    t = 0
    while True:
        ok, bgr = cap.read()
        if not ok or t > max(need):
            break
        H, W = bgr.shape[:2]
        m = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); ts = int(t * 1000 / fps)
        res = face.detect_for_video(m, ts); s = seg.segment_for_video(m, ts)
        if t in need and res.face_landmarks:
            lm_of[t] = [[q.x, q.y, q.z] for q in res.face_landmarks[0]]
            prob = s.confidence_masks[3].numpy_view().astype(np.float32); cat = s.category_mask.numpy_view()
            mask_of[t] = np.squeeze(E.smoothstep(*E.SKIN_PROB, prob) * (cat != 1)).astype(np.float32)      # as applyFoundSkinMask, 256 x 256
            img_of[t] = bgr
        t += 1

    jobs = []
    for f in frames:
        cv2.imwrite(str(work / f"frame_{f}.png"), img_of[f])
        xs = [p[0] for p in lm_of[f]]; ys = [p[1] for p in lm_of[f]]; zs = [p[2] for p in lm_of[f]]
        for k in ks:
            mk = mask_of[f - k]; mh, mw = mk.shape
            pts = np.array(lm_of[f - k])[COVER][:, :2]
            ix = np.clip((pts[:, 0] * mw).astype(int), 0, mw - 1); iy = np.clip((pts[:, 1] * mh).astype(int), 0, mh - 1)
            occ = float(smooth(0.6, 0.85, (mk[iy, ix] * 255 > 127).mean()))
            name = f"{a.tag}_f{f}_k{k}"
            cv2.imwrite(str(work / f"{name}_mask.png"), (mk * 255).astype(np.uint8))
            jobs.append({"name": name, "frame": f"frame_{f}.png", "w": int(W), "h": int(H), "x": xs, "y": ys, "z": zs,
                         "hex": a.hex, "intensity": 0.9, "texture": "matte", "mask": f"{name}_mask.png", "mw": int(mw), "mh": int(mh), "occ": occ})
    html = r.page_html(a.page)
    assert OLD in html and UPLOAD in html, "the offline harness text changed"
    html = html.replace(OLD, NEW).replace(UPLOAD, WITH_MASK)
    (work / "jobs.json").write_text(json.dumps(jobs)); (work / "index.html").write_text(html, encoding="utf-8")
    httpd = r.serve(work); url = f"http://127.0.0.1:{httpd.server_address[1]}/index.html"
    dom = subprocess.run([str(r.CHROME), "--headless=new", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                          f"--virtual-time-budget={60000 + 15000 * len(jobs)}", "--dump-dom", url],
                         capture_output=True, text=True, encoding="utf-8", timeout=1500).stdout
    httpd.shutdown()
    title = re.search(r"<title>([^<]*)", dom); print(a.tag, title.group(1)[:300] if title else "no title")
    n = 0
    for name, data in re.findall(r'<img class="result" data-name="([^"]+)" src="data:image/png;base64,([^"]+)"', dom):
        (a.out / f"{name}.png").write_bytes(base64.b64decode(data)); n += 1
    shutil.rmtree(work, ignore_errors=True)
    print("saved", n, "| occ per job:", {j["name"].split("_", 1)[1]: round(j["occ"], 2) for j in jobs})


if __name__ == "__main__":
    main()
