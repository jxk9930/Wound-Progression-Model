"""Shared utility functions for worsening model (standalone)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, median_filter


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    shp = arr.shape
    flat = arr.reshape(-1, 3)
    r, g, b = flat[:, 0] / 255.0, flat[:, 1] / 255.0, flat[:, 2] / 255.0
    cmax = np.maximum(r, np.maximum(g, b))
    cmin = np.minimum(r, np.minimum(g, b))
    d = cmax - cmin
    h = np.zeros_like(cmax)
    nz = d > 1e-8
    rm = (cmax == r) & nz
    gm = (cmax == g) & nz
    bm = (cmax == b) & nz
    h[rm] = (60.0 * ((g[rm] - b[rm]) / d[rm]) + 360.0) % 360.0
    h[gm] = 60.0 * ((b[gm] - r[gm]) / d[gm] + 2.0)
    h[bm] = 60.0 * ((r[bm] - g[bm]) / d[bm] + 4.0)
    s = np.zeros_like(cmax)
    pos = cmax > 1e-8
    s[pos] = d[pos] / cmax[pos]
    hsv = np.stack([h, s, cmax], axis=1)
    return hsv.reshape(*shp[:-1], 3).astype(np.float32)


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    arr = np.asarray(hsv, dtype=np.float32)
    shp = arr.shape
    flat = arr.reshape(-1, 3)
    h = flat[:, 0] % 360.0
    s = np.clip(flat[:, 1], 0.0, 1.0)
    v = np.clip(flat[:, 2], 0.0, 1.0)
    c = v * s
    hp = h / 60.0
    x = c * (1.0 - np.abs((hp % 2.0) - 1.0))
    m = v - c
    rgb = np.zeros_like(flat)
    conds = [
        (0 <= hp) & (hp < 1), (1 <= hp) & (hp < 2), (2 <= hp) & (hp < 3),
        (3 <= hp) & (hp < 4), (4 <= hp) & (hp < 5), (5 <= hp) & (hp < 6),
    ]
    vals = [(c, x, 0), (x, c, 0), (0, c, x), (0, x, c), (x, 0, c), (c, 0, x)]
    for mask, (rv, gv, bv) in zip(conds, vals):
        if np.any(mask):
            rgb[mask, 0] = rv[mask] if hasattr(rv, '__len__') else rv
            rgb[mask, 1] = gv[mask] if hasattr(gv, '__len__') else gv
            rgb[mask, 2] = bv[mask] if hasattr(bv, '__len__') else bv
    rgb = (rgb + m[:, None]) * 255.0
    return rgb.reshape(*shp[:-1], 3).astype(np.float32)


def circular_h_distance(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    d = np.abs(h1 - h2)
    return np.minimum(d, 360.0 - d)


def circular_h_lerp(h_from: np.ndarray, h_to: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    delta = ((h_to - h_from + 180.0) % 360.0) - 180.0
    return (h_from + alpha * delta) % 360.0


def smooth_field(mask: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter(mask.astype(np.float32), sigma=sigma)


def signed_distance(mask: np.ndarray) -> np.ndarray:
    inside = mask.astype(bool)
    d_in = distance_transform_edt(inside)
    d_out = distance_transform_edt(~inside)
    phi = d_out.astype(np.float32)
    phi[inside] = -d_in[inside]
    return phi.astype(np.float32)


def wound_radius(mask: np.ndarray) -> float:
    area = float(np.count_nonzero(mask))
    return max(np.sqrt(area / np.pi), 1.0)


def _nearest_feature_map(values: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    support = support_mask.astype(bool)
    if not np.any(support):
        mean_val = values.reshape(-1, values.shape[-1]).mean(axis=0)
        out = np.broadcast_to(mean_val, values.shape).copy()
        return out.astype(np.float32)
    _, inds = distance_transform_edt(~support, return_indices=True)
    out = values[inds[0], inds[1]]
    return out.astype(np.float32)


def compute_skin_rgb_map(rgb_image: np.ndarray, wound_mask: np.ndarray,
                         foot_mask: np.ndarray | None = None, ring_px: int = 6) -> np.ndarray:
    rgb = np.asarray(rgb_image, dtype=np.float32)
    wound = wound_mask.astype(bool)
    h, w = wound.shape
    foot = np.ones((h, w), dtype=bool) if foot_mask is None else foot_mask.astype(bool)
    if not foot.any():
        foot = np.ones((h, w), dtype=bool)
    ring = binary_dilation(wound, iterations=ring_px) & ~binary_dilation(wound, iterations=max(1, ring_px - 2))
    support = foot & ~wound
    ring_support = ring & support
    if np.count_nonzero(ring_support) >= 25:
        support = ring_support
    elif np.count_nonzero(support) < 25:
        support = ~wound
    return _nearest_feature_map(rgb, support)


def compute_skin_hsv_map(rgb_image: np.ndarray, wound_mask: np.ndarray,
                         foot_mask: np.ndarray | None = None, ring_px: int = 6) -> np.ndarray:
    return rgb_to_hsv(compute_skin_rgb_map(rgb_image, wound_mask, foot_mask=foot_mask, ring_px=ring_px))


def conditional_boundary_smooth(rgb_image: np.ndarray, process_mask: np.ndarray,
                                hue_threshold_deg: float = 24.0, iterations: int = 1) -> np.ndarray:
    rgb = np.asarray(rgb_image, dtype=np.float32).copy()
    if not np.any(process_mask):
        return rgb
    for _ in range(max(1, int(iterations))):
        hsv = rgb_to_hsv(rgb)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        H_med = median_filter(H, size=3, mode='nearest')
        S_med = median_filter(S, size=3, mode='nearest')
        V_med = median_filter(V, size=3, mode='nearest')
        gap = circular_h_distance(H, H_med)
        take = process_mask & (gap > hue_threshold_deg)
        if not np.any(take):
            continue
        H[take] = circular_h_lerp(H[take], H_med[take], 0.55)
        S[take] = 0.55 * S[take] + 0.45 * S_med[take]
        V[take] = 0.70 * V[take] + 0.30 * V_med[take]
        rgb = hsv_to_rgb(np.stack([H, S, V], axis=-1))
        np.clip(rgb, 0, 255, out=rgb)
    return rgb.astype(np.float32)
