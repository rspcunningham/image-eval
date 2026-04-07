from __future__ import annotations

import numpy as np
from pydantic import BaseModel
from scipy import ndimage


class Point(BaseModel):
    x: float
    y: float


class AnchorResult(BaseModel):
    center_x: float
    center_y: float
    angle_deg: float
    edge_length_px: float
    corners: list[Point]


def _find_large_square(data: np.ndarray) -> tuple[int, int, int, int]:
    """Find the bounding box of the largest square-ish dark blob."""
    mask = data < 0.2
    labels, _ = ndimage.label(mask)
    objs = ndimage.find_objects(labels)

    best = None
    best_area = 0
    for i, slc in enumerate(objs, 1):
        if slc is None:
            continue
        h = slc[0].stop - slc[0].start
        w = slc[1].stop - slc[1].start
        area = int(np.sum(labels[slc] == i))
        aspect = min(h, w) / max(h, w)
        if aspect > 0.8 and area > best_area and area / (h * w) > 0.8:
            best_area = area
            best = (slc[1].start, slc[0].start, slc[1].stop, slc[0].stop)

    if best is None:
        raise RuntimeError("Could not find a reference square in the image")
    return best


def _refine_anchor(data: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> AnchorResult:
    margin = 25
    ry1 = max(0, y1 - margin)
    ry2 = min(data.shape[0] - 1, y2 + margin)
    rx1 = max(0, x1 - margin)
    rx2 = min(data.shape[1] - 1, x2 + margin)
    roi = data[ry1 : ry2 + 1, rx1 : rx2 + 1]

    smoothed = ndimage.gaussian_filter(roi, 1.0)
    gx = ndimage.sobel(smoothed, axis=1)
    gy = ndimage.sobel(smoothed, axis=0)
    mag = np.sqrt(gx**2 + gy**2)

    edge_mask = mag > 0.15
    ey, ex = np.where(edge_mask)
    if len(ex) < 8:
        raise RuntimeError("Anchor edge detection produced too few edge points")

    gx_vals = gx[ey, ex]
    gy_vals = gy[ey, ex]

    roi_cy = roi.shape[0] / 2
    roi_cx = roi.shape[1] / 2

    h_mask = np.abs(gy_vals) > np.abs(gx_vals)
    h_x, h_y = ex[h_mask], ey[h_mask]
    top_mask = h_y < roi_cy
    bot_mask = h_y >= roi_cy

    v_mask = ~h_mask
    v_x, v_y = ex[v_mask], ey[v_mask]
    left_mask = v_x < roi_cx
    right_mask = v_x >= roi_cx

    if not top_mask.any() or not bot_mask.any() or not left_mask.any() or not right_mask.any():
        raise RuntimeError("Could not classify anchor edges")

    top_p = np.polyfit(h_x[top_mask], h_y[top_mask], 1)
    bot_p = np.polyfit(h_x[bot_mask], h_y[bot_mask], 1)
    left_p = np.polyfit(v_y[left_mask], v_x[left_mask], 1)
    right_p = np.polyfit(v_y[right_mask], v_x[right_mask], 1)

    def intersect_hv(h_coeffs: np.ndarray, v_coeffs: np.ndarray) -> tuple[float, float]:
        a, b = h_coeffs
        c, d = v_coeffs
        denominator = 1 - c * a
        if abs(denominator) < 1e-6:
            raise RuntimeError("Anchor line fit became degenerate")
        x = (c * b + d) / denominator
        y = a * x + b
        return x, y

    tl = intersect_hv(top_p, left_p)
    tr = intersect_hv(top_p, right_p)
    br = intersect_hv(bot_p, right_p)
    bl = intersect_hv(bot_p, left_p)

    corners = [
        Point(x=tl[0] + rx1, y=tl[1] + ry1),
        Point(x=tr[0] + rx1, y=tr[1] + ry1),
        Point(x=br[0] + rx1, y=br[1] + ry1),
        Point(x=bl[0] + rx1, y=bl[1] + ry1),
    ]

    center_x = np.mean([c.x for c in corners])
    center_y = np.mean([c.y for c in corners])

    def dist(p1: Point, p2: Point) -> np.float64:
        return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    edge_length = np.mean([
        dist(corners[0], corners[1]),
        dist(corners[1], corners[2]),
        dist(corners[2], corners[3]),
        dist(corners[3], corners[0]),
    ])

    angle_top = np.degrees(np.arctan(top_p[0]))
    angle_bot = np.degrees(np.arctan(bot_p[0]))

    return AnchorResult(
        center_x=float(center_x),
        center_y=float(center_y),
        angle_deg=float((angle_top + angle_bot) / 2),
        edge_length_px=float(edge_length),
        corners=corners,
    )


def find_anchor(data: np.ndarray) -> AnchorResult:
    x1, y1, x2, y2 = _find_large_square(data)
    return _refine_anchor(data, x1, y1, x2, y2)


def main(data: np.ndarray) -> AnchorResult:
    return find_anchor(data)
