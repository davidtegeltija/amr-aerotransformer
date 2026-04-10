from dataclasses import dataclass
from typing import Dict, Optional

from typing import Dict

import numpy as np

from src.amr.physics_metrics import (
    compute_velocity_gradient,
    compute_vorticity,
    compute_momentum_magnitude,
    compute_kelvin_helmholtz_shear,
    compute_channel_variance,
    compute_channel_entropy,
)
from src.amr.geometry_metrics import (
    compute_surface_curvature,
    compute_leading_trailing_edge,
    compute_thickness_gradient,
    compute_distance_to_wall,
)
 
@dataclass
class RefinementCriteria:
    """
    Thresholds that control which cells get subdivided.
 
    Set a field to "None" to disable that metric entirely (it will not be computed or checked).
 
    OR-logic applies: a cell is subdivided if *any* enabled metric exceeds
    its threshold.
 
    Geometry thresholds
    ---------------------------------------
    curvature_threshold      : mean discrete curvature magnitude.
                               Primary signal for curved surfaces
                               (LE nose, high-camber regions).
    le_te_threshold          : leading / trailing edge indicator.
                               Detects chordwise curvature peaks
                               near LE and TE.
    thickness_grad_threshold : thickness-gradient magnitude.
                               Detects rapid thickness changes
                               (flap gaps, blunt TEs, thin tips).
    wall_distance_threshold  : inverse distance to wall.
                               Ensures near-wall cells are resolved
                               (boundary-layer region).

    Physics thresholds (from AMR-Transformer paper)
    ------------------------------------------------
    grad_threshold      : velocity gradient magnitude (Eq. 2).
                          Primary signal for boundary layers, shocks, wakes.
    vorticity_threshold : vorticity magnitude (Eq. 3).
                          Detects rotating flow, vortex cores.
    momentum_threshold  : momentum per unit area (Eq. 4).
                          Useful for high-speed regions.
    kh_shear_threshold  : max Kelvin-Helmholtz shear (Eq. 5).
                          Detects shear-layer instabilities.
    variance_threshold  : mean per-channel variance.
                          Generic signal for any kind of local variation.
    entropy_threshold   : Shannon entropy of value distribution.
                          Detects information-rich patches regardless of
                          which physical quantity drives the complexity.
 
    Examples
    --------
    # Use only velocity gradient and vorticity:
    cfg = RefinementCriteria(
        grad_threshold=0.05,
        vorticity_threshold=0.04,
        momentum_threshold=None,
        kh_shear_threshold=None,
        variance_threshold=None,
        entropy_threshold=None,
    )
 
    # Coarser mesh (raise all thresholds by 2x):
    cfg = AERODYNAMIC_CRITERIA.scale(2.0)
    """

    # Geometry thresholds
    curvature_threshold:      Optional[float] = 0.10
    le_te_threshold:          Optional[float] = 0.15
    thickness_grad_threshold: Optional[float] = 0.08
    wall_distance_threshold:  Optional[float] = 5.00
 
    # Physics thresholds
    grad_threshold:      Optional[float] = 0.08
    vorticity_threshold: Optional[float] = 0.06
    momentum_threshold:  Optional[float] = 0.80
    kh_shear_threshold:  Optional[float] = 0.60
    variance_threshold:  Optional[float] = 0.02
    entropy_threshold:   Optional[float] = 2.50
 
    def scale(self, factor: float) -> "RefinementCriteria":
        """
        Return a new config with all enabled thresholds multiplied by factor.
 
        factor < 1.0 -> lower thresholds -> finer mesh (more tokens)
        factor > 1.0 -> higher thresholds -> coarser mesh (fewer tokens)
        """
        def _s(v):
            return v * factor if v is not None else None
        return RefinementCriteria(
            curvature_threshold=      _s(self.curvature_threshold),
            le_te_threshold=          _s(self.le_te_threshold),
            thickness_grad_threshold= _s(self.thickness_grad_threshold),
            wall_distance_threshold=  _s(self.wall_distance_threshold),

            grad_threshold=           _s(self.grad_threshold),
            vorticity_threshold=      _s(self.vorticity_threshold),
            momentum_threshold=       _s(self.momentum_threshold),
            kh_shear_threshold=       _s(self.kh_shear_threshold),
            variance_threshold=       _s(self.variance_threshold),
            entropy_threshold=        _s(self.entropy_threshold),
        )
 
    def geometry_kwargs(self) -> Dict:
        """Keyword arguments for the geometry metrics only."""
        return {
            "curvature_threshold":      self.curvature_threshold,
            "le_te_threshold":          self.le_te_threshold,
            "thickness_grad_threshold": self.thickness_grad_threshold,
            "wall_distance_threshold":  self.wall_distance_threshold,
        }
    
    def physics_kwargs(self) -> Dict:
        """Keyword arguments for the physics metrics only."""
        return {
            "grad_threshold":      self.grad_threshold,
            "vorticity_threshold": self.vorticity_threshold,
            "momentum_threshold":  self.momentum_threshold,
            "kh_shear_threshold":  self.kh_shear_threshold,
            "variance_threshold":  self.variance_threshold,
            "entropy_threshold":   self.entropy_threshold,
        }
    
    def to_kwargs(self) -> Dict:
        """
        Return a dict of keyword arguments accepted by build_adaptive_mesh()
        and should_subdivide().  None-valued thresholds are included so
        those metrics are explicitly skipped.
        """
        return {**self.physics_kwargs(), **self.geometry_kwargs()}

    def compute_enabled_metrics(self, region: np.ndarray) -> Dict[str, float]:
        """Compute and return metric values for all enabled (non-None) thresholds."""
        metrics = {}
        # Geometry thresholds
        if self.curvature_threshold is not None:
            metrics["surface_curvature"] = compute_surface_curvature(region)
        if self.le_te_threshold is not None:
            metrics["leading_trailing_edge"] = compute_leading_trailing_edge(region)
        if self.thickness_grad_threshold is not None:
            metrics["thickness_gradient"] = compute_thickness_gradient(region)
        if self.wall_distance_threshold is not None:
            metrics["distance_to_wall"] = compute_distance_to_wall(region)

        # Physics thresholds
        if self.grad_threshold is not None:
            metrics["velocity_gradient"] = compute_velocity_gradient(region)
        if self.vorticity_threshold is not None:
            metrics["vorticity"] = compute_vorticity(region)
        if self.momentum_threshold is not None:
            metrics["momentum"] = compute_momentum_magnitude(region)
        if self.kh_shear_threshold is not None:
            metrics["kh_shear"] = compute_kelvin_helmholtz_shear(region)
        if self.variance_threshold is not None:
            metrics["variance"] = compute_channel_variance(region)
        if self.entropy_threshold is not None:
            metrics["entropy"] = compute_channel_entropy(region)
        
        return metrics
    
    def threshold_checks(self) -> list[tuple[str, float]]:
        """Return (metric_name, threshold) pairs for all enabled metrics."""
        return [
            (name, t) for name, t in [
                ("surface_curvature",     self.curvature_threshold),
                ("leading_trailing_edge", self.le_te_threshold),
                ("thickness_gradient",    self.thickness_grad_threshold),
                ("distance_to_wall",      self.wall_distance_threshold),
                ("velocity_gradient",     self.grad_threshold),
                ("vorticity",             self.vorticity_threshold),
                ("momentum",              self.momentum_threshold),
                ("kh_shear",              self.kh_shear_threshold),
                ("variance",              self.variance_threshold),
                ("entropy",               self.entropy_threshold),
            ]
            if t is not None
        ]
