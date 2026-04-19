"""Healing-only wound progression model.

This is the original stable Batch-1 style model file before the later
experimental geometry changes. It keeps:
  - stable cumulative inward closure from the initial signed distance (phi0)
  - outward skin recovery with gap-based slowdown
  - simpler 3-zone appearance blending
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import binary_dilation

from wound_healing_config import (
    DEFAULT_RYKW_COLORS,
    EPITHELIAL_HSV,
    HEALTHY_WOUND_HSV,
    HealingParams,
)
from wound_healing_color import (
    circular_h_lerp,
    compute_skin_hsv_map,
    compute_skin_rgb_map,
    conditional_boundary_smooth,
    hsv_to_rgb,
    rgb_to_hsv,
)
from wound_healing_geometry import (
    binary_from_field,
    signed_distance,
    smooth_field,
    wound_radius,
)


def compute_ref_colors(rgb: np.ndarray, rykw: np.ndarray, mask: np.ndarray) -> dict[int, np.ndarray]:
    """Average RGB per RYKW class from the initial wound image."""
    ref: dict[int, np.ndarray] = {}
    for lb in (1, 2, 3, 4):
        m = (rykw == lb) & (mask > 0)
        ref[lb] = (rgb[m].astype(np.float32).mean(axis=0)
                   if np.count_nonzero(m) >= 10 else DEFAULT_RYKW_COLORS[lb].copy())
    return ref


@dataclass
class Snapshot:
    day: float
    rgb: np.ndarray
    wound_mask: np.ndarray
    area_px: int
    inward_front_px: float
    outward_front_px: float
    inward_speed_px: float
    epi_fraction: float
    gap_gate: float


class HealingWoundModel:
    def __init__(
        self,
        mask: np.ndarray,
        rgb: np.ndarray,
        rykw: np.ndarray,
        G: float | None = None,
        S: float | None = None,
        N: float | None = None,
        foot_mask: np.ndarray | None = None,
        ref_colors: dict[int, np.ndarray] | None = None,
        params: HealingParams | None = None,
        seed: int = 0,
    ):
        self.mask0 = np.asarray(mask > 0, dtype=bool)
        self.rgb = np.asarray(rgb, dtype=np.float32).copy()
        self.rgb0 = np.asarray(rgb, dtype=np.float32).copy()
        self.rykw0 = np.asarray(rykw, dtype=np.uint8)
        self.params = params or HealingParams()
        self.rng = np.random.default_rng(seed)

        self.shape = self.mask0.shape
        self.phi0 = signed_distance(self.mask0)
        self.w = smooth_field(self.mask0.astype(np.float32), sigma=self.params.smooth_sigma_init)
        self.current_mask = binary_from_field(self.w)
        self.area0 = int(np.count_nonzero(self.mask0))
        self.radius0 = wound_radius(self.mask0)

        self.ref_colors = ref_colors or compute_ref_colors(self.rgb0, self.rykw0, self.mask0)
        self.ref_hsv = {k: rgb_to_hsv(v[None, :])[0] for k, v in self.ref_colors.items()}
        self.skin_rgb = compute_skin_rgb_map(self.rgb0, self.mask0, foot_mask=foot_mask)
        self.skin_hsv = compute_skin_hsv_map(self.rgb0, self.mask0, foot_mask=foot_mask)
        self.orig_hsv = rgb_to_hsv(self.rgb0)

        self.max_outward_band_px = self.params.max_outward_band_px(self.radius0)
        self.outward_front_px = 0.0
        self.inward_front_px = 0.0
        self.last_gap_gate = 1.0

        outside_distance = self.phi0
        outward_band = (~self.mask0) & (outside_distance > 0) & (outside_distance <= self.max_outward_band_px)
        self.process_mask = self.mask0 | outward_band

        self.alpha_noise = np.clip(
            1.0 + self.params.noise_strength_alpha * self.rng.standard_normal(self.shape),
            0.78, 1.24,
        ).astype(np.float32)
        self.rgb_noise = (
            self.params.noise_strength_rgb * self.rng.standard_normal((*self.shape, 3))
        ).astype(np.float32)

        self.initial_wound_hsv = np.zeros((*self.shape, 3), dtype=np.float32)
        bg_hsv = self.skin_hsv.copy()
        self.initial_wound_hsv[:] = bg_hsv
        for lb, hsv in self.ref_hsv.items():
            m = self.rykw0 == lb
            self.initial_wound_hsv[m] = hsv

        vals = [v for v in (G, S, N) if v is not None]
        self.global_health_bias = 0.0
        if len(vals) == 3:
            s = float(G) + float(S) + float(N) + 1e-8
            self.global_health_bias = float(max(0.0, (float(G) - 0.5 * float(S) - 0.8 * float(N)) / s))

    # -------------------------- geometry --------------------------
    def _hybrid_inward_speed(self) -> float:
        """Blend lamellipodial crawling (large wounds) with purse-string-like closure (small wounds)."""
        if not np.any(self.current_mask):
            return 0.0
        p = self.params
        r = wound_radius(self.current_mask)
        r_norm = r / max(self.radius0, 1e-6)
        lo = p.purse_small_radius_ratio
        hi = p.crawl_large_radius_ratio
        if hi <= lo:
            crawl_w = 0.5
        else:
            crawl_w = float(np.clip((r_norm - lo) / (hi - lo), 0.0, 1.0))
        purse_w = 1.0 - crawl_w
        purse_term = p.purse_string_speed_px_per_day + p.purse_curvature_gain / max(r, 1.0)
        return float(crawl_w * p.crawl_speed_px_per_day + purse_w * purse_term)

    def _boundary_ring(self, mask: np.ndarray) -> np.ndarray:
        dil = binary_dilation(mask, iterations=1)
        ero = binary_dilation(~mask, iterations=1)
        return dil & ero

    def _estimate_gap_gate(self) -> float:
        """Measure residual boundary mismatch between current edge and skin."""
        boundary = self._boundary_ring(self.current_mask)
        if not np.any(boundary):
            return self.params.min_gap_gate
        cur = np.clip(self.rgb[boundary], 0, 255)
        skin = np.clip(self.skin_rgb[boundary], 0, 255)
        gap = np.linalg.norm((cur - skin) / 255.0, axis=1)
        if gap.size == 0:
            return self.params.min_gap_gate
        g = float(np.percentile(gap, 70))
        lo = self.params.gap_stop_threshold
        hi = self.params.gap_full_threshold
        if hi <= lo:
            return 1.0
        x = np.clip((g - lo) / (hi - lo), 0.0, 1.0)
        gate = self.params.min_gap_gate + (1.0 - self.params.min_gap_gate) * (x ** self.params.outward_saturation_power)
        return float(np.clip(gate, self.params.min_gap_gate, 1.0))

    def _step_geometry(self):
        """Apply inward closure as a controllable front displacement from phi0."""
        if not np.any(self.mask0):
            self.current_mask[:] = False
            self.w[:] = 0.0
            return

        new_mask = self.phi0 <= (-self.inward_front_px)
        if np.any(new_mask):
            self.current_mask = new_mask
            self.w = smooth_field(self.current_mask.astype(np.float32), sigma=self.params.smooth_sigma_init)
        else:
            self.current_mask[:] = False
            self.w[:] = 0.0

    def _step_fronts(self):
        inward_speed = self._hybrid_inward_speed()
        self.inward_front_px += inward_speed * self.params.dt

        gap_gate = self._estimate_gap_gate()
        saturation = 1.0 - min(1.0, self.outward_front_px / max(1.0, self.max_outward_band_px))
        outward_speed = self.params.outward_speed_px_per_day * gap_gate * max(0.0, saturation)
        self.outward_front_px += outward_speed * self.params.dt
        self.outward_front_px = float(min(self.outward_front_px, self.max_outward_band_px))
        self.last_gap_gate = float(gap_gate)

    # -------------------------- appearance --------------------------
    def _deep_wound_target_hsv(self, heal_global: np.ndarray) -> np.ndarray:
        """Shift deep wound colors gradually toward healthier granulation."""
        base = self.initial_wound_hsv.copy()
        healthy = np.broadcast_to(HEALTHY_WOUND_HSV, base.shape).copy()
        alpha = np.clip(heal_global, 0.0, 1.0)
        H = circular_h_lerp(base[..., 0], healthy[..., 0], alpha)
        S = (1.0 - alpha) * base[..., 1] + alpha * healthy[..., 1]
        V = (1.0 - alpha) * base[..., 2] + alpha * healthy[..., 2]
        return np.stack([H, S, V], axis=-1)

    def _blend_appearance(self):
        proc = self.process_mask
        if not np.any(proc):
            return

        phi = signed_distance(self.current_mask)
        inside = phi <= 0
        outside = phi > 0
        d_in = np.clip(-phi, 0.0, None)
        d_out = np.clip(phi, 0.0, None)

        epi_band_px = max(self.params.epi_band_px(self.radius0), 0.72 * self.inward_front_px)
        prog_in = np.zeros(self.shape, dtype=np.float32)
        prog_out = np.zeros(self.shape, dtype=np.float32)
        if self.inward_front_px > 1e-6:
            prog_in[inside] = np.clip(1.0 - d_in[inside] / max(epi_band_px, self.inward_front_px), 0.0, 1.0)
        if self.outward_front_px > 1e-6:
            prog_out[outside] = np.clip(1.0 - d_out[outside] / self.outward_front_px, 0.0, 1.0)

        area = max(1.0, float(np.count_nonzero(self.current_mask)))
        area_frac = 1.0 - min(1.0, area / max(1.0, self.area0))
        heal_global = np.clip(area_frac + self.global_health_bias, 0.0, 1.0) ** self.params.health_gain_power
        heal_global_map = np.full(self.shape, heal_global, dtype=np.float32)

        cur_hsv = rgb_to_hsv(self.rgb)
        skin_hsv = self.skin_hsv
        deep_hsv = self._deep_wound_target_hsv(heal_global_map)
        orig_v = self.orig_hsv[..., 2]
        skin_v = skin_hsv[..., 2]

        tgt_h = cur_hsv[..., 0].copy()
        tgt_s = cur_hsv[..., 1].copy()
        tgt_v = cur_hsv[..., 2].copy()

        deep_zone = inside & proc & (prog_in < 1e-3)
        if np.any(deep_zone):
            tgt_h[deep_zone] = deep_hsv[..., 0][deep_zone]
            tgt_s[deep_zone] = deep_hsv[..., 1][deep_zone]
            tgt_v[deep_zone] = (
                (1.0 - self.params.v_blend_inside) * orig_v[deep_zone]
                + self.params.v_blend_inside * deep_hsv[..., 2][deep_zone]
            )

        epi_in = inside & proc & (prog_in > 0)
        if np.any(epi_in):
            ep = prog_in[epi_in]
            stage1 = np.clip(self.params.epi_stage1_gain * ep, 0.0, 1.0)
            stage2 = np.clip((ep - self.params.epi_stage2_start) / max(self.params.epi_stage2_width, 1e-6), 0.0, 1.0)

            h1 = circular_h_lerp(deep_hsv[..., 0][epi_in], EPITHELIAL_HSV[0], stage1)
            s1 = (1.0 - stage1) * deep_hsv[..., 1][epi_in] + stage1 * EPITHELIAL_HSV[1]
            v1 = (1.0 - stage1) * deep_hsv[..., 2][epi_in] + stage1 * EPITHELIAL_HSV[2]

            tgt_h[epi_in] = circular_h_lerp(h1, skin_hsv[..., 0][epi_in], stage2)
            tgt_s[epi_in] = (1.0 - stage2) * s1 + stage2 * skin_hsv[..., 1][epi_in]
            skin_mix_v = (1.0 - self.params.skin_v_mix) * orig_v[epi_in] + self.params.skin_v_mix * skin_v[epi_in]
            tgt_v[epi_in] = (1.0 - stage2) * v1 + stage2 * skin_mix_v

        out_zone = outside & proc & (prog_out > 0)
        if np.any(out_zone):
            po = prog_out[out_zone]
            tgt_h[out_zone] = circular_h_lerp(cur_hsv[..., 0][out_zone], skin_hsv[..., 0][out_zone], 0.85 * po)
            tgt_s[out_zone] = (1.0 - 0.88 * po) * cur_hsv[..., 1][out_zone] + (0.88 * po) * skin_hsv[..., 1][out_zone]
            target_v = (1.0 - self.params.skin_v_mix) * orig_v[out_zone] + self.params.skin_v_mix * skin_v[out_zone]
            tgt_v[out_zone] = (1.0 - self.params.v_blend_outside * po) * cur_hsv[..., 2][out_zone] + (self.params.v_blend_outside * po) * target_v

        tgt_rgb = hsv_to_rgb(np.stack([tgt_h, tgt_s, tgt_v], axis=-1))
        tgt_rgb = np.clip(tgt_rgb + 0.10 * self.rgb_noise, 0, 255)

        alpha = np.full(self.shape, self.params.base_blend_rate * self.params.dt, dtype=np.float32)
        alpha *= self.alpha_noise
        alpha[deep_zone] *= self.params.deep_blend_boost
        alpha[epi_in] *= self.params.epi_blend_boost
        alpha[out_zone] *= self.params.healed_blend_boost
        alpha = np.clip(alpha, 0.0, 0.95)

        self.rgb[proc] = self.rgb[proc] + alpha[proc, None] * (tgt_rgb[proc] - self.rgb[proc])
        np.clip(self.rgb, 0, 255, out=self.rgb)

    # -------------------------- public API --------------------------
    def step(self):
        self._step_fronts()
        self._step_geometry()
        self._blend_appearance()

    def snapshot(self, day: float) -> Snapshot:
        rgb_out = self.rgb.copy()
        boundary_band = self.process_mask & (np.abs(signed_distance(self.current_mask)) <= max(1.0, self.outward_front_px + 1.5))
        rgb_out = conditional_boundary_smooth(
            rgb_out,
            boundary_band,
            hue_threshold_deg=self.params.boundary_hue_threshold_deg,
            iterations=self.params.boundary_smooth_iters,
        )
        area_px = int(np.count_nonzero(self.current_mask))
        epi_fraction = float(min(1.0, self.inward_front_px / max(1e-6, self.radius0)))
        return Snapshot(
            day=float(day),
            rgb=np.clip(rgb_out, 0, 255).astype(np.uint8),
            wound_mask=self.current_mask.astype(np.uint8),
            area_px=area_px,
            inward_front_px=float(self.inward_front_px),
            outward_front_px=float(self.outward_front_px),
            inward_speed_px=float(self._hybrid_inward_speed()),
            epi_fraction=epi_fraction,
            gap_gate=float(self.last_gap_gate),
        )

    def run(self, weeks: float = 4.0, snapshot_every_days: float = 7.0):
        steps = int(np.ceil((weeks * 7.0) / self.params.dt))
        snap_every = max(1, int(round(snapshot_every_days / self.params.dt)))
        out = [self.snapshot(0.0)]
        for i in range(1, steps + 1):
            self.step()
            if i % snap_every == 0 or i == steps:
                out.append(self.snapshot(i * self.params.dt))
        return out


def run_healing_trajectory(
    mask: np.ndarray,
    rgb_image: np.ndarray,
    rykw_initial: np.ndarray,
    foot_mask: np.ndarray | None = None,
    weeks: float = 4.0,
    snapshot_every_days: float = 7.0,
    seed: int = 0,
    G: float | None = None,
    S: float | None = None,
    N: float | None = None,
    params: HealingParams | None = None,
):
    model = HealingWoundModel(
        mask=mask,
        rgb=rgb_image,
        rykw=rykw_initial,
        G=G,
        S=S,
        N=N,
        foot_mask=foot_mask,
        params=params,
        seed=seed,
    )
    return model.run(weeks=weeks, snapshot_every_days=snapshot_every_days)
