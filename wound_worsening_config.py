"""Configuration and defaults for ischemia + infection worsening model."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# Color anchors (RGB 0-255)
INFECTED_YELLOW_RGB = np.array([196, 168, 72], dtype=np.float32)   # slough / infected tissue
NECROTIC_BLACK_RGB = np.array([42, 33, 28], dtype=np.float32)      # ischemic necrosis
DEEP_RED_RGB = np.array([132, 42, 48], dtype=np.float32)           # inflamed / deep wound bed

INFECTED_YELLOW_HSV = np.array([46.0, 0.64, 0.77], dtype=np.float32)
NECROTIC_BLACK_HSV = np.array([18.0, 0.34, 0.17], dtype=np.float32)
DEEP_RED_HSV = np.array([356.0, 0.68, 0.54], dtype=np.float32)


@dataclass
class WorseningParams:
    """Parameters for ischemia + infection worsening.

    Assumed biology / image logic:
      - outward: infection spreads centrifugally from the current wound edge into
        the periwound region, turning surrounding tissue yellow/slough-like first.
      - inward (edge-driven): superficial infected/slough changes also move inward
        from the wound margin over the wound bed.
      - inward (core-driven): ischemic necrosis starts in the deepest / most poorly
        perfused central core and propagates outward, making the core darker,
        blacker, and visually deeper.
    """

    dt: float = 0.20  # days per update

    # Geometry: wound expands outward as tissue breaks down.
    outward_geom_speed_px_per_day: float = 0.22
    outward_geom_max_ratio: float = 0.18

    # Appearance fronts
    outward_infection_speed_px_per_day: float = 0.82
    inward_edge_infection_speed_px_per_day: float = 1.10
    core_ischemia_speed_px_per_day: float = 0.8

    max_outward_band_ratio: float = 0.35
    min_outward_band_px: int = 4

    # Rendering / blending
    base_blend_rate: float = 0.30
    periwound_blend_boost: float = 1.70
    inner_infection_blend_boost: float = 1.6
    core_necrosis_blend_boost: float = 1.8

    # Depth cue
    depth_shadow_gain: float = 0.24
    core_darkening_gain: float = 0.42
    infected_v_drop: float = 0.14

    # Post smoothing
    boundary_hue_threshold_deg: float = 24.0
    boundary_smooth_iters: int = 1

    # Stable noise
    noise_strength_rgb: float = 4.0
    noise_strength_alpha: float = 0.10

    def max_outward_band_px(self, wound_radius0: float) -> int:
        return max(self.min_outward_band_px,
                   int(np.ceil(self.max_outward_band_ratio * wound_radius0)))

    def outward_geom_max_px(self, wound_radius0: float) -> int:
        return max(self.min_outward_band_px,
                   int(np.ceil(self.outward_geom_max_ratio * wound_radius0)))
