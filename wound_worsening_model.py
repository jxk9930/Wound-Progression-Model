"""Ischemia + infection worsening-only wound progression model.

Design:
- outward geometry expansion: wound boundary enlarges outward
- outward infection: periwound turns yellow/slough from current edge outward
- inward edge infection: wound-bed rim turns yellow from edge inward
- core ischemia: deepest central core progresses yellow -> black faster
- IMPORTANT:
  inside progression is per-pixel and local-anchor based, not fixed-paint based
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import binary_dilation

from wound_worsening_config import (
    WorseningParams,
    INFECTED_YELLOW_RGB,
    NECROTIC_BLACK_RGB,
)
from wound_worsening_utils import (
    circular_h_lerp,
    compute_skin_hsv_map,
    compute_skin_rgb_map,
    conditional_boundary_smooth,
    hsv_to_rgb,
    rgb_to_hsv,
    signed_distance,
    smooth_field,
    wound_radius,
)


@dataclass
class Snapshot:
    day: float
    rgb: np.ndarray
    wound_mask: np.ndarray
    area_px: int
    outward_geom_px: float
    outward_infection_px: float
    inward_edge_infection_px: float
    core_ischemia_px: float


class WorseningWoundModel:
    def __init__(
        self,
        mask: np.ndarray,
        rgb: np.ndarray,
        rykw: np.ndarray | None = None,
        foot_mask: np.ndarray | None = None,
        params: WorseningParams | None = None,
        seed: int = 0,
    ):
        self.mask0 = np.asarray(mask > 0, dtype=bool)
        self.rgb = np.asarray(rgb, dtype=np.float32).copy()
        self.rgb0 = np.asarray(rgb, dtype=np.float32).copy()
        self.params = params or WorseningParams()
        self.rng = np.random.default_rng(seed)

        self.shape = self.mask0.shape
        self.current_mask = self.mask0.copy()
        self.w = smooth_field(self.current_mask.astype(np.float32), sigma=1.2)
        self.area0 = int(np.count_nonzero(self.mask0))
        self.radius0 = wound_radius(self.mask0)

        self.skin_rgb = compute_skin_rgb_map(self.rgb0, self.mask0, foot_mask=foot_mask)
        self.skin_hsv = compute_skin_hsv_map(self.rgb0, self.mask0, foot_mask=foot_mask)
        self.orig_hsv = rgb_to_hsv(self.rgb0)

        self.max_outward_band_px = self.params.max_outward_band_px(self.radius0)
        self.geom_max_expand_px = self.params.outward_geom_max_px(self.radius0)

        # Progress fronts
        self.outward_geom_px = 0.0
        self.outward_infection_px = 0.0
        self.inward_edge_infection_px = 0.0
        self.core_ischemia_px = 0.0
        self.geom_carry_px = 0.0

        # Stable local variation
        self.alpha_noise = np.clip(
            1.0 + self.params.noise_strength_alpha * self.rng.standard_normal(self.shape),
            0.82, 1.18,
        ).astype(np.float32)
        self.rgb_noise = (
            self.params.noise_strength_rgb
            * self.rng.standard_normal((*self.shape, 3))
        ).astype(np.float32)
        # Low-frequency spatial heterogeneity so worsening does not look like
        # perfectly symmetric paint layers.
        from scipy.ndimage import gaussian_filter
        raw_noise = self.rng.standard_normal(self.shape).astype(np.float32)
        self.progress_noise = gaussian_filter(raw_noise, sigma=5.0)
        self.progress_noise = np.clip(1.0 + 0.18 * self.progress_noise, 0.72, 1.28)
        # --- Local per-pixel wound-bed base state ---
        # Use the original wound-bed as the baseline red state.
        self.base_wound_hsv = rgb_to_hsv(self.rgb0)

        yellow_const_hsv = rgb_to_hsv(INFECTED_YELLOW_RGB[None, :])[0]
        black_const_hsv = rgb_to_hsv(NECROTIC_BLACK_RGB[None, :])[0]

        # --- Local per-pixel yellow anchor ---
        # Move each pixel toward infected/slough-like yellow, but keep local variability.
        yh = circular_h_lerp(self.base_wound_hsv[..., 0], yellow_const_hsv[0], 0.48)
        ys = 0.34 * self.base_wound_hsv[..., 1] + 0.66 * yellow_const_hsv[1]
        yv = 0.60 * self.base_wound_hsv[..., 2] + 0.40 * yellow_const_hsv[2]
        self.local_yellow_hsv = np.stack([yh, ys, yv], axis=-1)

        # --- Local per-pixel black/necrotic anchor ---
        # Move each pixel toward necrosis, but still preserve local hue/texture structure.
        bh = circular_h_lerp(self.base_wound_hsv[..., 0], black_const_hsv[0], 0.58)
        bs = 0.42 * self.base_wound_hsv[..., 1] + 0.58 * black_const_hsv[1]
        bv = 0.32 * self.base_wound_hsv[..., 2] + 0.68 * black_const_hsv[2]
        self.local_black_hsv = np.stack([bh, bs, bv], axis=-1)

    # --------------------- progression logic ---------------------
    def _step_fronts(self):
        p = self.params

        # outward geometry expansion
        if self.outward_geom_px < self.geom_max_expand_px:
            self.outward_geom_px += p.outward_geom_speed_px_per_day * p.dt
            self.outward_geom_px = min(self.outward_geom_px, float(self.geom_max_expand_px))

        # outward periwound infection from current edge
        self.outward_infection_px += p.outward_infection_speed_px_per_day * p.dt
        self.outward_infection_px = min(self.outward_infection_px, float(self.max_outward_band_px))

        # inward superficial infection/slough from edge
        self.inward_edge_infection_px += p.inward_edge_infection_speed_px_per_day * p.dt
        self.inward_edge_infection_px = min(self.inward_edge_infection_px, 0.95 * self.radius0)

        # core ischemia / necrosis from the deepest center
        self.core_ischemia_px += p.core_ischemia_speed_px_per_day * p.dt
        self.core_ischemia_px = min(self.core_ischemia_px, 0.98 * self.radius0)

    def _step_geometry(self):
        if not np.any(self.current_mask):
            return

        p = self.params
        self.geom_carry_px += p.outward_geom_speed_px_per_day * p.dt
        if self.geom_carry_px < 1.0:
            self.w = smooth_field(self.current_mask.astype(np.float32), sigma=1.0)
            return

        expand_px = int(self.geom_carry_px)
        self.geom_carry_px -= expand_px
        if expand_px <= 0:
            return

        self.current_mask = binary_dilation(self.current_mask, iterations=expand_px)
        self.w = smooth_field(self.current_mask.astype(np.float32), sigma=1.0)

    # --------------------- appearance logic ---------------------
    def _blend_appearance(self):
        p = self.params
        phi = signed_distance(self.current_mask)
        inside = phi <= 0
        outside = phi > 0

        # process current wound + outward infection band
        proc = inside | (outside & (phi <= self.max_outward_band_px))
        if not np.any(proc):
            return

        d_in = np.clip(-phi, 0.0, None)
        d_out = np.clip(phi, 0.0, None)

        current_inside = inside & proc

        # distance from current edge for inside pixels
        d_edge = d_in
        max_d = float(d_edge[current_inside].max()) if np.any(current_inside) else 1.0
        core_rank = np.zeros(self.shape, dtype=np.float32)
        if np.any(current_inside):
            core_rank[current_inside] = d_edge[current_inside] / max(max_d, 1e-6)

        # ---------- progress fields ----------
        # outward infection in periwound
        prog_out = np.zeros(self.shape, dtype=np.float32)
        if self.outward_infection_px > 1e-6:
            m = outside & proc
            prog_out[m] = np.clip(1.0 - d_out[m] / self.outward_infection_px, 0.0, 1.0)

        # inward edge infection/slough
        edge_infect = np.zeros(self.shape, dtype=np.float32)
        if self.inward_edge_infection_px > 1e-6:
            m = current_inside
            edge_infect[m] = np.clip(1.0 - d_edge[m] / self.inward_edge_infection_px, 0.0, 1.0)

        # core ischemia / necrosis progression
        core_nec = np.zeros(self.shape, dtype=np.float32)
        if np.any(current_inside):
            thresh = np.clip(max_d - self.core_ischemia_px, 0.0, max_d)
            m = current_inside
            core_nec[m] = np.clip(
                (d_edge[m] - thresh) / max(self.core_ischemia_px, 1e-6),
                0.0, 1.0
            )
        # Apply low-frequency spatial heterogeneity
        edge_infect *= self.progress_noise
        core_nec *= self.progress_noise

        edge_infect = np.clip(edge_infect, 0.0, 1.0)
        core_nec = np.clip(core_nec, 0.0, 1.0)
        # global worsening burden with geometry expansion
        area_frac = min(2.2, float(np.count_nonzero(self.current_mask)) / max(1.0, self.area0))
        global_worse = np.clip((area_frac - 1.0) / 0.7, 0.0, 1.0)

        cur_hsv = rgb_to_hsv(self.rgb)
        orig_v = self.orig_hsv[..., 2]
        skin_hsv = self.skin_hsv
        skin_v = skin_hsv[..., 2]

        tgt_h = cur_hsv[..., 0].copy()
        tgt_s = cur_hsv[..., 1].copy()
        tgt_v = cur_hsv[..., 2].copy()

        # ---------- inside current wound ----------
        if np.any(current_inside):
            m = current_inside

            # Per-pixel progression:
            # red(local base) -> yellow(local infected) from edge
            # yellow -> black(local necrotic) from core
            yellow_prog = np.clip(
                0.10 * global_worse
                + 1.10 * edge_infect[m]
                + 0.24 * core_nec[m],
                0.0, 1.0
            )

            black_prog = np.clip(
                0.04 * global_worse
                + 0.72 * core_nec[m]
                + 0.06 * edge_infect[m] * core_rank[m],
                0.0, 1.0
            )

            base_h = self.base_wound_hsv[..., 0][m]
            base_s = self.base_wound_hsv[..., 1][m]
            base_v = self.base_wound_hsv[..., 2][m]

            y_h = self.local_yellow_hsv[..., 0][m]
            y_s = self.local_yellow_hsv[..., 1][m]
            y_v = self.local_yellow_hsv[..., 2][m]

            b_h = self.local_black_hsv[..., 0][m]
            b_s = self.local_black_hsv[..., 1][m]
            b_v = self.local_black_hsv[..., 2][m]

            # stage 1: local base -> local yellow
            h_y = circular_h_lerp(base_h, y_h, yellow_prog)
            s_y = (1.0 - yellow_prog) * base_s + yellow_prog * y_s
            v_y = (1.0 - yellow_prog) * base_v + yellow_prog * y_v

            # stage 2: local yellow state -> local black
            tgt_h[m] = circular_h_lerp(h_y, b_h, black_prog)
            tgt_s[m] = (1.0 - black_prog) * s_y + black_prog * b_s

            depth_shadow = (
                p.depth_shadow_gain
                * (core_rank[m] ** 1.65)
                * (0.25 + 0.75 * core_nec[m])
            )
            infected_drop = 0.70 * p.infected_v_drop * edge_infect[m]
            nec_drop = p.core_darkening_gain * core_nec[m]

            tgt_v[m] = np.clip(
                (1.0 - black_prog) * v_y + black_prog * b_v
                - depth_shadow
                - infected_drop
                - nec_drop,
                0.02, 1.0
            )

        # ---------- outside current wound / periwound infection ----------
        periwound = outside & proc
        if np.any(periwound):
            m = periwound
            po = prog_out[m]

            # periwound starts from existing skin/orig and moves toward yellow infection
            yellow_out_h = circular_h_lerp(skin_hsv[..., 0][m], self.local_yellow_hsv[..., 0][m], 0.78 * po)
            yellow_out_s = (1.0 - 0.78 * po) * skin_hsv[..., 1][m] + (0.78 * po) * self.local_yellow_hsv[..., 1][m]
            yellow_out_v = (1.0 - 0.65 * po) * orig_v[m] + (0.65 * po) * self.local_yellow_hsv[..., 2][m]

            tgt_h[m] = circular_h_lerp(cur_hsv[..., 0][m], yellow_out_h, 0.85 * po)
            tgt_s[m] = (1.0 - 0.85 * po) * cur_hsv[..., 1][m] + (0.85 * po) * yellow_out_s
            tgt_v[m] = np.clip(
                (1.0 - 0.72 * po) * cur_hsv[..., 2][m] + (0.72 * po) * yellow_out_v - 0.03 * po,
                0.02, 1.0
            )

        tgt_rgb = hsv_to_rgb(np.stack([tgt_h, tgt_s, tgt_v], axis=-1))
        tgt_rgb = np.clip(tgt_rgb + 0.08 * self.rgb_noise, 0, 255)

        alpha = np.full(self.shape, p.base_blend_rate * p.dt, dtype=np.float32)
        alpha *= self.alpha_noise

        # inside should worsen gradually, not be instantly repainted
        if np.any(current_inside):
            local_drive = np.clip(edge_infect[current_inside] + core_nec[current_inside], 0.0, 1.0)
            alpha[current_inside] *= 0.88 + 0.10 * local_drive

        # periwound infection can still be more visible
        alpha[periwound] *= p.periwound_blend_boost
        alpha = np.clip(alpha, 0.0, 0.95)

        self.rgb[proc] = self.rgb[proc] + alpha[proc, None] * (tgt_rgb[proc] - self.rgb[proc])
        np.clip(self.rgb, 0, 255, out=self.rgb)

    # --------------------- public API ---------------------
    def step(self):
        self._step_fronts()
        self._step_geometry()
        self._blend_appearance()

    def snapshot(self, day: float) -> Snapshot:
        rgb_out = self.rgb.copy()
        phi = signed_distance(self.current_mask)
        process_mask = (np.abs(phi) <= max(1.0, self.outward_infection_px + 1.5))
        rgb_out = conditional_boundary_smooth(
            rgb_out,
            process_mask,
            hue_threshold_deg=self.params.boundary_hue_threshold_deg,
            iterations=self.params.boundary_smooth_iters,
        )
        return Snapshot(
            day=float(day),
            rgb=np.clip(rgb_out, 0, 255).astype(np.uint8),
            wound_mask=self.current_mask.astype(np.uint8),
            area_px=int(np.count_nonzero(self.current_mask)),
            outward_geom_px=float(self.outward_geom_px),
            outward_infection_px=float(self.outward_infection_px),
            inward_edge_infection_px=float(self.inward_edge_infection_px),
            core_ischemia_px=float(self.core_ischemia_px),
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


def run_worsening_trajectory(
    mask: np.ndarray,
    rgb_image: np.ndarray,
    rykw_initial: np.ndarray | None = None,
    foot_mask: np.ndarray | None = None,
    weeks: float = 4.0,
    snapshot_every_days: float = 7.0,
    seed: int = 0,
    params: WorseningParams | None = None,
):
    model = WorseningWoundModel(
        mask=mask,
        rgb=rgb_image,
        rykw=rykw_initial if rykw_initial is not None else np.zeros_like(mask, dtype=np.uint8),
        foot_mask=foot_mask,
        params=params,
        seed=seed,
    )
    return model.run(weeks=weeks, snapshot_every_days=snapshot_every_days)
