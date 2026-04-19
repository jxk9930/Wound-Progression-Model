"""
Wound Progression Model (v5.1) — Edge Smoothing + Outward Extension
====================================================================
INPUTS: wound_mask, rgb_image, rykw_initial, G/S/N, foot_mask (optional)
MACRO:  Vermolen boundary v=(A+B·κ)·H(c-Q), size-dependent speed
        + outward w extension: periwound band participates in blending
MICRO:  Schugart 4-ODE per pixel (M,T,E,f), masked to w>0.5
BLEND:  HSV method with 3-zone model:
        Zone 1 (healed):     H,S from real skin (foot seg), V from original
        Zone 2 (epithelial): H,S blend wound→white/pink→skin, V from original
        Zone 3 (deep wound): H,S from ODE-driven target, V from original
POST:   Conditional median H,S filter — smooths boundary where neighbor
        hue gap is large (circular H distance), V untouched

Changes from v5:
  - Outward w extension: w field extends ~8% wound_radius beyond mask0
    so periwound pixels participate in blending (no hard cutoff)
  - Conditional median H filter: per-pixel 8-neighbor check, if circular
    H gap > 30°, pull H,S toward median. Runs 2 iterations per snapshot.
  - Processing region expanded from mask0 to mask0 + outward band
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

# ── Literature Parameters (Krishna 2015 Table 3) ───────────────────────
K_GOOD = np.array([32.715, 3.613, 0.042, 0.652, 4.724, 0.027,
                    2.001, 1.582, 0.365, 0.013, 3.499, 0.067])
K_POOR = np.array([1.597, 0.834, 0.206, 0.009, 39.861, 18.973,
                    0.003, 0.689, 0.002, 0.002, 3.025, 1.837])

DEFAULT_RYKW_COLORS = {
    1: np.array([180, 60, 60],   dtype=np.float32),   # Red (granulation)
    2: np.array([200, 180, 100], dtype=np.float32),    # Yellow (slough)
    3: np.array([50, 35, 30],    dtype=np.float32),    # Black (necrosis)
    4: np.array([220, 215, 200], dtype=np.float32),    # White (epithelial)
}

# Epithelial white/pink target in HSV [H:0-360, S:0-1, V:0-1]
# Light pinkish-white: low saturation, warm hue, high brightness
EPITHELIAL_HSV = np.array([15.0, 0.12, 0.88], dtype=np.float32)

# ODE initial states per RYKW class: (M, T, E, f)
INIT_STATE = {
    0: (0.0, 0.0, 1.0, 1.0),
    1: (0.3, 0.7, 0.4, 0.4),
    2: (1.5, 0.3, 0.2, 0.2),
    3: (2.0, 0.1, 0.05, 0.05),
    4: (0.5, 0.5, 0.3, 0.15),
}

# Keratinocyte migration speed: ~0.5-1 mm/day → in pixel-ratio units
# This is the fraction of wound_radius covered per day
EPI_MIGRATION_RATE = 0.04  # ~4% of wound radius per day

# Outward band: fraction of wound_radius to extend w field beyond mask0
# This creates a periwound transition zone for smooth blending
OUTWARD_BAND_RATIO = 0.08  # ~8% of wound radius

# Conditional median filter: circular H gap threshold (degrees)
# If a pixel's H differs from any neighbor by more than this, smooth it
MEDIAN_H_THRESHOLD = 30.0
MEDIAN_ITERATIONS = 2  # passes per snapshot


# ── RGB ↔ HSV (vectorized, no OpenCV) ──────────────────────────────────

def rgb_to_hsv(rgb):
    """(N,3) float32 RGB [0-255] → (N,3) float32 HSV [H:0-360, S:0-1, V:0-1]"""
    r, g, b = rgb[:, 0] / 255, rgb[:, 1] / 255, rgb[:, 2] / 255
    cmax = np.maximum(r, np.maximum(g, b))
    cmin = np.minimum(r, np.minimum(g, b))
    d = cmax - cmin + 1e-8
    h = np.zeros_like(r)
    rm = cmax == r
    gm = (cmax == g) & ~rm
    bm = ~rm & ~gm
    h[rm] = 60 * (((g[rm] - b[rm]) / d[rm]) % 6)
    h[gm] = 60 * ((b[gm] - r[gm]) / d[gm] + 2)
    h[bm] = 60 * ((r[bm] - g[bm]) / d[bm] + 4)
    s = d / (cmax + 1e-8)
    return np.stack([h, s, cmax], axis=1).astype(np.float32)


def hsv_to_rgb(hsv):
    """(N,3) float32 HSV [H:0-360, S:0-1, V:0-1] → (N,3) float32 RGB [0-255]"""
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    c = v * s
    x = c * (1 - np.abs((h / 60) % 2 - 1))
    m = v - c
    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)
    for lo, hi, rv, gv, bv in [(0, 60, c, x, 0), (60, 120, x, c, 0),
                                (120, 180, 0, c, x), (180, 240, 0, x, c),
                                (240, 300, x, 0, c), (300, 360, c, 0, x)]:
        mask = (h >= lo) & (h < hi)
        r[mask] = rv[mask] if hasattr(rv, '__len__') else rv
        g[mask] = gv[mask] if hasattr(gv, '__len__') else gv
        b[mask] = bv[mask] if hasattr(bv, '__len__') else bv
    return (np.stack([r + m, g + m, b + m], axis=1) * 255).astype(np.float32)


def _circular_h_distance(h1, h2):
    """Circular distance between two hue values (0-360 degrees)."""
    d = np.abs(h1 - h2)
    return np.minimum(d, 360.0 - d)


def _conditional_median_hs(rgb_image, process_mask, threshold=MEDIAN_H_THRESHOLD):
    """Conditional median filter on H,S channels where H gap is large.

    For each pixel in process_mask, checks 8-neighbor H values.
    If max circular H gap > threshold, replaces H,S with median of 3x3.
    V channel is never touched (preserves original brightness/texture).

    Args:
        rgb_image: (H,W,3) float32 RGB [0-255]
        process_mask: (H,W) bool — which pixels to consider
        threshold: circular H gap in degrees to trigger smoothing

    Returns:
        rgb_image: (H,W,3) float32 RGB — smoothed in-place
    """
    H, W = process_mask.shape
    if not process_mask.any():
        return rgb_image

    # Convert full image to HSV
    flat_rgb = rgb_image.reshape(-1, 3).astype(np.float32)
    flat_hsv = rgb_to_hsv(np.clip(flat_rgb, 0, 255))
    hsv_img = flat_hsv.reshape(H, W, 3)

    h_ch = hsv_img[:, :, 0].copy()
    s_ch = hsv_img[:, :, 1].copy()
    v_ch = hsv_img[:, :, 2].copy()  # preserved, never modified

    # 8-neighbor offsets
    offsets = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),           (0, 1),
              (1, -1),  (1, 0),  (1, 1)]

    # Pad H and S for neighbor access
    h_pad = np.pad(h_ch, 1, mode='edge')
    s_pad = np.pad(s_ch, 1, mode='edge')

    # Find pixels where max neighbor H gap exceeds threshold
    coords = np.argwhere(process_mask)
    if len(coords) == 0:
        return rgb_image

    # Vectorized: gather 3x3 neighborhood for all process pixels
    cy, cx = coords[:, 0], coords[:, 1]
    center_h = h_ch[cy, cx]

    # Collect all 9 H and S values for each pixel (center + 8 neighbors)
    h_neighbors = np.zeros((len(coords), 9), np.float32)
    s_neighbors = np.zeros((len(coords), 9), np.float32)
    h_neighbors[:, 0] = center_h
    s_neighbors[:, 0] = s_ch[cy, cx]

    for k, (dy, dx) in enumerate(offsets):
        ny, nx = cy + dy + 1, cx + dx + 1  # +1 for padding offset
        h_neighbors[:, k + 1] = h_pad[ny, nx]
        s_neighbors[:, k + 1] = s_pad[ny, nx]

    # Compute max circular H gap between center and each neighbor
    h_gaps = _circular_h_distance(center_h[:, None], h_neighbors[:, 1:])
    max_gap = h_gaps.max(axis=1)

    # Only smooth pixels where gap exceeds threshold
    needs_smooth = max_gap > threshold
    if not needs_smooth.any():
        return rgb_image

    # Compute median H (circular-aware) and median S — vectorized
    smooth_mask = needs_smooth
    if not smooth_mask.any():
        return rgb_image

    h_vals = h_neighbors[smooth_mask]  # (K, 9)
    s_vals = s_neighbors[smooth_mask]  # (K, 9)

    # Circular median for H: shift all values so center H is at 180°
    ref_h = h_vals[:, 0:1]  # (K, 1) — center pixel H
    shifted = (h_vals - ref_h + 180) % 360  # (K, 9)
    med_shifted = np.median(shifted, axis=1)  # (K,)
    new_h = (med_shifted + ref_h[:, 0] - 180) % 360  # (K,)

    # Standard median for S
    new_s = np.median(s_vals, axis=1)  # (K,)

    # Write back
    smooth_ys = cy[smooth_mask]
    smooth_xs = cx[smooth_mask]
    h_ch[smooth_ys, smooth_xs] = new_h
    s_ch[smooth_ys, smooth_xs] = new_s

    # Reconstruct RGB from smoothed H,S + original V
    new_hsv = np.stack([h_ch, s_ch, v_ch], axis=2).reshape(-1, 3)
    new_rgb = hsv_to_rgb(new_hsv).reshape(H, W, 3)

    # Only update pixels that were smoothed
    smooth_ys = cy[needs_smooth]
    smooth_xs = cx[needs_smooth]
    rgb_image[smooth_ys, smooth_xs] = new_rgb[smooth_ys, smooth_xs]

    return rgb_image


# ── Foot Segmentation → Skin HSV Map ──────────────────────────────────

def foot_mask_from_roboflow(result, image_shape):
    """Convert Roboflow foot-segmentation API result to binary mask.

    Args:
        result: dict from Roboflow inference (Option A: inference_sdk)
        image_shape: (H, W) of the original image

    Returns:
        foot_mask: (H, W) uint8, 1 inside foot, 0 outside
    """
    import cv2
    H, W = image_shape[:2]
    foot_mask = np.zeros((H, W), dtype=np.uint8)

    if not result.get("predictions"):
        return foot_mask

    # Use highest-confidence foot prediction
    preds = sorted(result["predictions"],
                   key=lambda p: p.get("confidence", 0), reverse=True)

    for pred in preds:
        if pred.get("class", "").lower() != "foot":
            continue
        points = pred.get("points", [])
        if not points:
            continue
        poly = np.array([[p["x"], p["y"]] for p in points], dtype=np.int32)
        cv2.fillPoly(foot_mask, [poly], 1)

    return foot_mask


def compute_skin_hsv_map(rgb_image, wound_mask, foot_mask=None):
    """Build per-pixel skin H,S target map from foot segmentation.

    For each wound pixel, finds the nearest valid skin pixel and stores
    its H and S values. V is always taken from the original photo.

    Args:
        rgb_image: (H,W,3) uint8
        wound_mask: (H,W) uint8, 1=wound
        foot_mask: (H,W) uint8, 1=foot (from Roboflow). If None, falls
                   back to using all non-wound pixels.

    Returns:
        skin_hs_map: (H,W,2) float32 — H and S channels for skin target
    """
    H, W = wound_mask.shape

    # Determine valid skin region
    if foot_mask is not None and foot_mask.sum() > 10:
        skin_region = (foot_mask > 0) & (wound_mask == 0)
    else:
        skin_region = wound_mask == 0

    if skin_region.sum() < 10:
        # Fallback: use nearest non-wound pixel
        skin_region = wound_mask == 0
    if skin_region.sum() < 10:
        # Absolute fallback: return neutral skin HSV
        skin_hs = np.zeros((H, W, 2), np.float32)
        skin_hs[:, :, 0] = 20.0   # warm hue
        skin_hs[:, :, 1] = 0.25   # moderate saturation
        return skin_hs

    # Distance transform: for each wound pixel, find nearest skin pixel
    search_mask = ~skin_region  # True where we need to find nearest skin
    _, idx = distance_transform_edt(search_mask, return_indices=True)

    # Convert all skin pixels to HSV
    skin_rgb_flat = rgb_image.reshape(-1, 3).astype(np.float32)
    skin_hsv_flat = rgb_to_hsv(skin_rgb_flat)
    skin_hsv_img = skin_hsv_flat.reshape(H, W, 3)

    # Build target map: each wound pixel gets H,S of nearest skin pixel
    skin_hs_map = np.zeros((H, W, 2), np.float32)
    w = wound_mask > 0
    skin_hs_map[w, 0] = skin_hsv_img[idx[0][w], idx[1][w], 0]  # H
    skin_hs_map[w, 1] = skin_hsv_img[idx[0][w], idx[1][w], 1]  # S

    # Also populate outward band (periwound) pixels with their own skin H,S
    # These are already skin pixels, so use their actual values
    outward = (wound_mask == 0)
    skin_hs_map[outward, 0] = skin_hsv_img[outward, 0]
    skin_hs_map[outward, 1] = skin_hsv_img[outward, 1]

    return skin_hs_map


# ── Parameters ─────────────────────────────────────────────────────────

@dataclass
class WoundParams:
    k: np.ndarray = field(default_factory=lambda: K_GOOD.copy())
    alpha: float = 2.0
    beta: float = 2.0
    M_bact: float = 0.0
    A: float = 0.5
    B: float = 0.5
    Q: float = 0.2
    D_gf: float = 10.0
    lam_gf: float = 1.0
    P_gf: float = 5.0
    D_lac: float = 50.0
    lam_lac: float = 0.5
    P_lac: float = 0.0
    n: float = 0.21
    p_death: float = 0.001
    dt: float = 0.05
    gf_sub: int = 10
    lac_sub: int = 20
    blend_rate: float = 0.10
    REF_RADIUS: float = 50.0

    @classmethod
    def from_gsn(cls, G, S, N, wound_px=0.0):
        t = G + S + N
        if t > 0:
            G, S, N = G / t, S / t, N / t
        wg, wp = G, S + N
        if wg + wp < 1e-6:
            wg, wp = 0.5, 0.5
        k = wg * K_GOOD + wp * K_POOR
        k[7] *= (1 - 0.5 * N)
        k[10] *= (1 - 0.5 * N)
        k[11] *= (1 + 0.2 * S)
        Mb = 10.0 * S
        A0 = 1.0 * G + 0.0 * S - 0.5 * N
        B = 0.5 * G - 0.2 * S - 0.5 * N
        Q = 0.15 * G + 0.35 * S + 0.40 * N
        sf = 1.0
        if wound_px > 0:
            sf = np.clip(cls.REF_RADIUS / max(np.sqrt(wound_px / np.pi), 1),
                         0.3, 2.5)
        return cls(k=k, M_bact=Mb, A=A0 * sf, B=B, Q=Q,
                   P_lac=3 * S, p_death=0.001 + 0.01 * (S + N),
                   blend_rate=0.06 + 0.05 * max(G, N))


# ── Helpers ────────────────────────────────────────────────────────────

def compute_ref_colors(rgb, rykw, mask):
    """Average RGB per RYKW class from the actual wound image."""
    ref = {}
    for lb in [1, 2, 3, 4]:
        m = (rykw == lb) & (mask > 0)
        ref[lb] = (rgb[m].astype(np.float32).mean(0)
                   if m.sum() >= 10 else DEFAULT_RYKW_COLORS[lb].copy())
    return ref


# ── Model ──────────────────────────────────────────────────────────────

class WoundModel:
    def __init__(self, mask, rgb, rykw, ref_colors, G, S, N,
                 skin_hs_map, seed=0):
        """
        Args:
            mask: (H,W) uint8 wound mask
            rgb: (H,W,3) uint8 original image
            rykw: (H,W) uint8 RYKW classification
            ref_colors: dict {1:RGB, 2:RGB, 3:RGB, 4:RGB}
            G, S, N: granulation/slough/necrosis fractions
            skin_hs_map: (H,W,2) float32 — H,S from foot segmentation
            seed: random seed
        """
        self.shape = mask.shape
        self.p = WoundParams.from_gsn(G, S, N, float(mask.sum()))
        self.rng = np.random.default_rng(seed)
        self.ref = ref_colors
        self.rgb = rgb.astype(np.float32).copy()
        self.mask0 = mask.copy()

        # Original V channel (brightness) — preserved throughout
        orig_flat = rgb.reshape(-1, 3).astype(np.float32)
        orig_hsv = rgb_to_hsv(orig_flat)
        self.orig_V = orig_hsv[:, 2].reshape(self.shape)

        # Skin H,S targets from foot segmentation
        self.skin_hs = skin_hs_map.copy()

        # Per-pixel noise for natural variation
        self.b_noise = np.clip(
            1 + 0.2 * self.rng.standard_normal(self.shape), 0.5, 1.5
        ).astype(np.float32)
        self.c_noise = (self.rng.standard_normal((*self.shape, 3)) * 8
                        ).astype(np.float32)

        # ODE state fields
        self.M = np.zeros(self.shape, np.float32)
        self.T = np.zeros(self.shape, np.float32)
        self.E = np.zeros(self.shape, np.float32)
        self.f = np.zeros(self.shape, np.float32)
        for lb, (m, t, e, fv) in INIT_STATE.items():
            ix = rykw == lb
            self.M[ix] = m
            self.T[ix] = t
            self.E[ix] = e
            self.f[ix] = fv

        # Level-set field w: smooth wound boundary
        self.w = gaussian_filter(mask.astype(np.float32), sigma=1.5)
        self.area0 = float(mask.sum())
        self.wound_radius0 = max(np.sqrt(self.area0 / np.pi), 1.0)

        # Outward band: extend w beyond mask0 for smooth periwound transition
        outward_px = max(int(self.wound_radius0 * OUTWARD_BAND_RATIO), 2)
        d_outside = distance_transform_edt(mask == 0)
        outward_band = (mask == 0) & (d_outside <= outward_px)
        # Give outward pixels a small w value that decays with distance
        self.w[outward_band] = np.clip(
            1.0 - d_outside[outward_band] / (outward_px + 1), 0, 0.4)
        # Re-smooth to blend the extension naturally
        self.w = gaussian_filter(self.w, sigma=1.0)

        # Extended processing mask: mask0 + outward band
        self.process_mask = (mask > 0) | outward_band

        # Chemical fields
        self.c_gf = np.zeros(self.shape, np.float32)
        self.c_lac = np.zeros(self.shape, np.float32)

        # Cell death
        self.dead = np.zeros(self.shape, bool)

        # RYKW classification (updated each step)
        self.rykw = rykw.copy().astype(np.uint8)

        # Epithelial front: fraction of wound_radius reached by keratinocytes
        # Starts at 0, grows by EPI_MIGRATION_RATE per day
        self.epi_front_ratio = 0.0

    @staticmethod
    def _lap(a):
        return (np.roll(a, 1, 0) + np.roll(a, -1, 0) +
                np.roll(a, 1, 1) + np.roll(a, -1, 1) - 4 * a)

    def _step_gf(self):
        p = self.p
        rim = 4 * self.w * (1 - self.w)
        src = p.P_gf * rim
        sd = p.dt / p.gf_sub
        for _ in range(p.gf_sub):
            self.c_gf += sd * (p.D_gf * self._lap(self.c_gf)
                               - p.lam_gf * self.c_gf + src)
        np.clip(self.c_gf, 0, None, out=self.c_gf)

    def _step_lac(self):
        p = self.p
        src = p.P_lac * (self.rykw == 2).astype(np.float32)
        sd = p.dt / p.lac_sub
        for _ in range(p.lac_sub):
            self.c_lac += sd * (p.D_lac * self._lap(self.c_lac)
                                - p.lam_lac * self.c_lac + src)
        np.clip(self.c_lac, 0, None, out=self.c_lac)

    def _step_ode(self):
        p = self.p
        k = p.k
        dt = p.dt
        ins = self.w > 0.5
        mu = np.exp(-p.n * self.c_lac)
        Ma = np.power(self.M + 1e-8, p.alpha)
        Ta = np.power(self.T + 1e-8, p.beta)
        Mh = Ma / (k[1] ** p.alpha + Ma)
        Th = Ta / (k[5] ** p.beta + Ta)
        fe = self.f * mu
        dM = k[0] * Mh * fe - k[2] * self.M - k[3] * self.M * self.T + p.M_bact
        dT = k[4] * Th * fe * self.M - k[6] * self.T - k[3] * self.M * self.T
        dE = k[7] * fe * (1 - self.E) - k[8] * self.M * self.E - k[9] * self.E
        df = k[10] * self.f * (1 - self.f) - k[11] * self.f
        o = ~ins
        dM[o] = dT[o] = dE[o] = df[o] = 0
        self.M = np.clip(self.M + dt * dM, 0, 5).astype(np.float32)
        self.T = np.clip(self.T + dt * dT, 0, 5).astype(np.float32)
        self.E = np.clip(self.E + dt * dE, 0, 1).astype(np.float32)
        self.f = np.clip(self.f + dt * df, 0, 1).astype(np.float32)

    def _step_bnd(self):
        p = self.p
        gy, gx = np.gradient(self.w)
        nm = np.sqrt(gx ** 2 + gy ** 2) + 1e-6
        _, dnx = np.gradient(gx / nm)
        dny, _ = np.gradient(gy / nm)
        kap = np.clip(dnx + dny, -2, 2)
        H = gaussian_filter((self.c_gf > p.Q).astype(np.float32), sigma=1.0)
        v = np.clip((p.A + p.B * kap) * H, -0.5, 0.5)
        rim = 4 * self.w * (1 - self.w)
        dw = -v * np.sqrt(gx ** 2 + gy ** 2) * rim
        vm = np.mean(v[self.w > 0.5]) if (self.w > 0.5).any() else 0
        dw += -0.008 * max(0, vm) * (self.w > 0.3) * self.w
        self.w = np.clip(self.w + p.dt * dw, 0, 1)

    def _step_death(self):
        p = self.p
        wound = self.w > 0.3
        self.dead |= ((self.rng.random(self.shape) < p.p_death * p.dt)
                       & wound & ~self.dead)
        mu = np.exp(-p.n * self.c_lac)
        self.dead &= ~(self.rng.random(self.shape) < mu * 0.02 * p.dt)

    def _classify(self):
        o = np.zeros(self.shape, np.uint8)
        w = self.w > 0.3
        o[w] = 2
        o[w & (self.f > 0.5) & (self.E > 0.2)] = 1
        o[w & (self.f < 0.1)] = 3
        o[w & (self.c_lac > 0.5) & (o != 3)] = 2
        o[w & self.dead] = 4
        return o

    def _step_epithelial_front(self):
        """Advance the epithelial front ratio.
        Keratinocytes migrate inward from wound edge at EPI_MIGRATION_RATE.
        Only advances in healing conditions (positive A → wound is closing)."""
        if self.p.A > 0:
            self.epi_front_ratio += EPI_MIGRATION_RATE * self.p.dt
        # Clamp to 1.0 (full coverage = fully epithelialized)
        self.epi_front_ratio = min(self.epi_front_ratio, 1.0)

    def _step_blend(self):
        """3-zone HSV blending with epithelial front.

        Operates on process_mask (mask0 + outward band) so periwound
        pixels also participate in color blending.
        """
        p = self.p
        proc = self.process_mask
        if not proc.any():
            return

        wv = np.clip(self.w[proc], 0, 1)

        # Current wound geometry
        current_wound = self.w > 0.5
        current_area = float(current_wound.sum())
        wound_radius = max(np.sqrt(current_area / np.pi), 1.0)

        # Distance from current wound boundary (inward from edge)
        if current_wound.any():
            d_edge = distance_transform_edt(current_wound)
        else:
            d_edge = np.zeros(self.shape, np.float32)
        d_edge_px = d_edge[proc]
        epi_ratio = d_edge_px / wound_radius  # normalized to wound size

        # ── Zone classification ──
        is_healed = wv < 0.5
        is_epithelial = (~is_healed) & (epi_ratio < self.epi_front_ratio)
        is_deep = ~is_healed & ~is_epithelial

        # ── ODE-driven wound color target (for deep wound + epithelial) ──
        fv = self.f[proc]
        Ev = self.E[proc]
        dv = self.dead[proc].astype(np.float32)
        wr = np.clip(fv * np.clip(Ev / 0.4, 0, 1), 0, 1)
        wk = np.clip((1 - fv / 0.15), 0, 1) * np.clip(1 - Ev, 0, 1)
        ww = dv * 0.8
        wy = np.clip(1 - wr - wk - ww, 0, 1)
        s = wr + wy + wk + ww + 1e-8
        wr /= s; wy /= s; wk /= s; ww /= s
        wound_rgb_tgt = (wr[:, None] * self.ref[1][None, :]
                         + wy[:, None] * self.ref[2][None, :]
                         + wk[:, None] * self.ref[3][None, :]
                         + ww[:, None] * self.ref[4][None, :]
                         + self.c_noise[proc])
        wound_hsv_tgt = rgb_to_hsv(np.clip(wound_rgb_tgt, 0, 255))

        # ── Skin H,S target (from foot segmentation) ──
        skin_h = self.skin_hs[proc, 0]
        skin_s = self.skin_hs[proc, 1]

        # ── Original V (brightness from photo) ──
        orig_v = self.orig_V[proc]

        # ── Build per-pixel target H, S, V ──
        n_px = proc.sum()
        tgt_h = np.zeros(n_px, np.float32)
        tgt_s = np.zeros(n_px, np.float32)
        tgt_v = np.zeros(n_px, np.float32)

        # Zone 1: HEALED — fully skin H,S, original V
        if is_healed.any():
            tgt_h[is_healed] = skin_h[is_healed]
            tgt_s[is_healed] = skin_s[is_healed]
            tgt_v[is_healed] = orig_v[is_healed]

        # Zone 2: EPITHELIAL — blend from wound → white/pink → skin
        if is_epithelial.any():
            # Progress within the epithelial band: 0 at wound edge, 1 at front
            epi_progress = np.zeros(n_px, np.float32)
            front = max(self.epi_front_ratio, 1e-6)
            epi_progress[is_epithelial] = np.clip(
                1.0 - epi_ratio[is_epithelial] / front, 0, 1)

            # Two-stage blend:
            #   progress 0.0-0.5: wound → epithelial white/pink
            #   progress 0.5-1.0: epithelial white/pink → skin
            ep = epi_progress[is_epithelial]

            # Stage weights
            stage1 = np.clip(2.0 * ep, 0, 1)        # 0→1 over first half
            stage2 = np.clip(2.0 * (ep - 0.5), 0, 1) # 0→1 over second half

            # H blend: wound_h → epi_h → skin_h
            w_h = wound_hsv_tgt[is_epithelial, 0]
            e_h = EPITHELIAL_HSV[0]
            s_h = skin_h[is_epithelial]
            h_mid = (1 - stage1) * w_h + stage1 * e_h
            tgt_h[is_epithelial] = (1 - stage2) * h_mid + stage2 * s_h

            # S blend: wound_s → epi_s (low) → skin_s
            w_s = wound_hsv_tgt[is_epithelial, 1]
            e_s = EPITHELIAL_HSV[1]
            s_s = skin_s[is_epithelial]
            s_mid = (1 - stage1) * w_s + stage1 * e_s
            tgt_s[is_epithelial] = (1 - stage2) * s_mid + stage2 * s_s

            # V: mostly original, slight influence from epithelial brightness
            e_v = EPITHELIAL_HSV[2]
            tgt_v[is_epithelial] = (0.85 * orig_v[is_epithelial]
                                    + 0.15 * ((1 - stage2) * e_v
                                              + stage2 * orig_v[is_epithelial]))

        # Zone 3: DEEP WOUND — ODE-driven H,S, original V
        if is_deep.any():
            tgt_h[is_deep] = wound_hsv_tgt[is_deep, 0]
            tgt_s[is_deep] = wound_hsv_tgt[is_deep, 1]
            tgt_v[is_deep] = 0.85 * orig_v[is_deep] + 0.15 * wound_hsv_tgt[is_deep, 2]

        # ── Reconstruct RGB from H,S,V ──
        final_hsv = np.stack([tgt_h, tgt_s, tgt_v], axis=1)
        final_rgb = hsv_to_rgb(final_hsv)

        # ── Blend toward target at blend_rate ──
        # Healed zone blends faster to make shrinkage visible
        alpha = p.blend_rate * p.dt * self.b_noise[proc]
        alpha[is_healed] *= 3.0   # faster skin transition in healed zone
        alpha[is_epithelial] *= 2.0  # moderately faster in epithelial zone

        cur = self.rgb[proc]
        self.rgb[proc] = cur + alpha[:, None] * (final_rgb - cur)
        np.clip(self.rgb, 0, 255, out=self.rgb)

    def _postprocess_median_hs(self):
        """Conditional median H,S filter at wound boundary.

        Checks each pixel's 8 neighbors. If circular H gap > threshold,
        pulls H,S toward median. V is never touched. Runs on the
        transition band (0.1 < w < 0.9) where boundary artifacts appear.
        Called only at snapshot times, not every substep.
        """
        transition = (self.w > 0.1) & (self.w < 0.9) & self.process_mask
        if not transition.any():
            return
        for _ in range(MEDIAN_ITERATIONS):
            self.rgb = _conditional_median_hs(
                self.rgb, transition, MEDIAN_H_THRESHOLD)

    def step(self):
        self._step_gf()
        self._step_lac()
        self._step_ode()
        self._step_bnd()
        self._step_death()
        self.rykw = self._classify()
        self._step_epithelial_front()
        self._step_blend()

    def run(self, weeks=4, snap_days=7):
        steps = int(weeks * 7 / self.p.dt)
        sn = max(1, int(snap_days / self.p.dt))
        out = [self._snap(0.0)]
        for i in range(1, steps + 1):
            self.step()
            if i % sn == 0 or i == steps:
                self._postprocess_median_hs()
                out.append(self._snap(i * self.p.dt))
        return out

    def _snap(self, day):
        a = float((self.w > 0.5).sum())
        par = 100 * (1 - a / self.area0) if self.area0 > 0 else 0
        w = self.w > 0.3
        rw = self.rykw[w]
        t = max(1, len(rw))
        return dict(
            day=day,
            rgb=np.clip(self.rgb, 0, 255).astype(np.uint8).copy(),
            rykw=self.rykw.copy(),
            w=self.w.copy(),
            area_px=a,
            par=par,
            epi_front_ratio=self.epi_front_ratio,
            rykw_pct=dict(
                R=(rw == 1).sum() / t * 100,
                Y=(rw == 2).sum() / t * 100,
                K=(rw == 3).sum() / t * 100,
                W=(rw == 4).sum() / t * 100,
            ),
        )


# ── Dual Trajectory ────────────────────────────────────────────────────

TRAJECTORY_PRESETS = {
    "healing":   {"G": 0.85, "S": 0.10, "N": 0.05},
    "worsening": {"G": 0.05, "S": 0.25, "N": 0.70},
}


def run_dual_trajectory(mask, rgb_image, rykw_initial,
                        foot_mask=None,
                        weeks=4, snapshot_every_days=7, seed=0,
                        healing_gsn=None, worsening_gsn=None):
    """Run healing and worsening trajectories.

    Args:
        mask: (H,W) uint8 wound mask
        rgb_image: (H,W,3) uint8
        rykw_initial: (H,W) uint8 RYKW map
        foot_mask: (H,W) uint8 foot segmentation mask (from Roboflow).
                   If None, falls back to non-wound pixels for skin color.
        weeks: simulation duration
        snapshot_every_days: snapshot interval
        seed: random seed
        healing_gsn: override healing G/S/N dict
        worsening_gsn: override worsening G/S/N dict

    Returns:
        dict {"healing": [snapshots], "worsening": [snapshots]}
    """
    ref = compute_ref_colors(rgb_image, rykw_initial, mask)
    skin_hs = compute_skin_hsv_map(rgb_image, mask, foot_mask)

    res = {}
    for name, dflt in TRAJECTORY_PRESETS.items():
        gsn = (healing_gsn if name == "healing" and healing_gsn
               else worsening_gsn if name == "worsening" and worsening_gsn
               else dflt)
        m = WoundModel(mask, rgb_image, rykw_initial, ref,
                       gsn["G"], gsn["S"], gsn["N"],
                       skin_hs, seed)
        res[name] = m.run(weeks, snapshot_every_days)
    return res