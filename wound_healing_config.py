"""Configuration and defaults for the healing-only wound progression model.

Revision: added Vermolen boundary-law parameters (vermolen_B, vermolen_Q,
growth_factor_sigma) so that curvature-dependent boundary smoothing is
applied on top of the hybrid crawl / purse-string inward speed.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

DEFAULT_RYKW_COLORS = {
    1: np.array([180, 60, 60], dtype=np.float32),
    2: np.array([200, 180, 100], dtype=np.float32),
    3: np.array([50, 35, 30], dtype=np.float32),
    4: np.array([220, 215, 200], dtype=np.float32),
}

EPITHELIAL_HSV = np.array([9.5, 0.20, 0.78], dtype=np.float32)
HEALTHY_WOUND_HSV = np.array([9.0, 0.62, 0.73], dtype=np.float32)


@dataclass
class HealingParams:
    """Parameters for healing-only bidirectional wound progression."""

    dt: float = 0.20  # days per update

    # ── Vermolen boundary law: v = (A + B·κ) · H(c - Q) ──────────────
    # A is computed dynamically by _hybrid_inward_speed() each step.
    # B controls how strongly curvature affects local boundary speed.
    # Q is the growth-factor threshold for the Heaviside gate.
    vermolen_B: float = 0.25          # curvature gain (px²/day)
    vermolen_Q: float = 0.15          # growth-factor threshold [0-1]
    growth_factor_sigma: float = 4.0  # spatial smoothing of GF field
    vermolen_clamp: float = 0.8       # max |v| to prevent instability

    # ── Hybrid inward speed (feeds into Vermolen A) ──────────────────
    crawl_speed_px_per_day: float = 0.75
    purse_string_speed_px_per_day: float = 0.45
    purse_curvature_gain: float = 0.35
    crawl_large_radius_ratio: float = 0.70
    purse_small_radius_ratio: float = 0.25

    # Legacy baseline (kept for outward band speed)
    inward_speed_px_per_day: float = 0.22
    outward_speed_px_per_day: float = 0.90

    max_outward_band_ratio: float = 0.28
    min_outward_band_px: int = 4

    # Geometry control
    smooth_sigma_init: float = 1.35

    # Appearance control
    base_blend_rate: float = 0.34
    healed_blend_boost: float = 2.6
    epi_blend_boost: float = 2.25
    deep_blend_boost: float = 0.9

    # V-channel / luminance blending
    v_blend_inside: float = 0.32
    v_blend_outside: float = 0.58
    skin_v_mix: float = 0.75

    # Progression zones
    min_epi_band_px: float = 1.6
    epi_band_ratio: float = 0.12
    health_gain_power: float = 0.85
    epi_stage1_gain: float = 2.5
    epi_stage2_start: float = 0.50
    epi_stage2_width: float = 0.70

    # Outward speed damping from residual skin-gap
    gap_stop_threshold: float = 0.035
    gap_full_threshold: float = 0.16
    min_gap_gate: float = 0.08
    outward_saturation_power: float = 1.3

    # Post-smoothing
    boundary_hue_threshold_deg: float = 24.0
    boundary_smooth_iters: int = 1

    # Stochastic realism
    noise_strength_rgb: float = 5.0
    noise_strength_alpha: float = 0.12

    def max_outward_band_px(self, wound_radius0: float) -> int:
        return max(self.min_outward_band_px,
                   int(np.ceil(self.max_outward_band_ratio * wound_radius0)))

    def epi_band_px(self, wound_radius0: float) -> float:
        return max(self.min_epi_band_px, self.epi_band_ratio * wound_radius0)
