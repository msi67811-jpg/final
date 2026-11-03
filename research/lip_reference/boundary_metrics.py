"""Boundary error between a predicted and a ground-truth binary mask (AR_SPEC 8-B, P1-1 "lip edge error").

IoU says how much area overlaps; it hides a lip outline that is a few pixels off all the way round (the paint
edge then reads as a drawn line next to the real lip edge). These add the distance between the two outlines.

  bnd_mean_px   mean distance between the outlines, both directions averaged (px)
  bnd_mean_rel  the same, divided by the ground-truth mouth width (removes the photo size)
  bnd_f1        boundary F1 within a tolerance of max(2 px, 1% of the mouth width): the share of each outline
                that lies within the tolerance of the other (precision = prediction, recall = ground truth)

Only numpy and OpenCV, so every eval script (project Python or another env) can import it. Local evaluation only.
"""
import cv2
import numpy as np

TOL_PX = 2.0
TOL_REL = 0.01


def _outline(mask):
    """Boundary pixels of a boolean mask (the mask minus its 1 px erosion)."""
    m = mask.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))).astype(bool)


def boundary_stats(pred, gt):
    """Boundary metrics of two boolean masks of the same shape; NaNs when either mask is empty."""
    nan = {"bnd_mean_px": float("nan"), "bnd_mean_rel": float("nan"), "bnd_f1": float("nan")}
    if not pred.any() or not gt.any():
        return nan
    ys, xs = np.nonzero(pred | gt)
    ex = np.nonzero(gt)[1]
    mouth_w = float(ex.max() - ex.min() + 1)
    pad = 12
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, pred.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, pred.shape[1])
    bp, bg = _outline(pred[y0:y1, x0:x1]), _outline(gt[y0:y1, x0:x1])
    # Distance of every pixel to the nearest outline pixel of the other mask.
    to_gt = cv2.distanceTransform((~bg).astype(np.uint8), cv2.DIST_L2, 5)
    to_pred = cv2.distanceTransform((~bp).astype(np.uint8), cv2.DIST_L2, 5)
    d_pred, d_gt = to_gt[bp], to_pred[bg]
    mean_px = 0.5 * (float(d_pred.mean()) + float(d_gt.mean()))
    tol = max(TOL_PX, TOL_REL * mouth_w)
    precision, recall = float((d_pred <= tol).mean()), float((d_gt <= tol).mean())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"bnd_mean_px": mean_px, "bnd_mean_rel": mean_px / mouth_w, "bnd_f1": f1}


if __name__ == "__main__":
    # Self-check: a disc against itself, shifted 3 px, and a 20% larger disc.
    yy, xx = np.mgrid[0:200, 0:200]
    disc = lambda cx, r: (xx - cx) ** 2 + (yy - 100) ** 2 <= r ** 2
    base = disc(100, 40)
    for label, other in (("same", base), ("shift 3px", disc(103, 40)), ("radius +8", disc(100, 48))):
        s = boundary_stats(other, base)
        print(f"{label:10s} mean {s['bnd_mean_px']:.2f}px  rel {s['bnd_mean_rel']:.3f}  f1 {s['bnd_f1']:.2f}")
