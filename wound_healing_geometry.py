"""Geometry helpers for healing-only wound progression."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


def smooth_field(mask: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter(mask.astype(np.float32), sigma=sigma)


def binary_from_field(w: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return np.asarray(w > threshold, dtype=bool)


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
