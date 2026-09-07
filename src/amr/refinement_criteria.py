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

    Every threshold defaults to "None", which disables that metric entirely (it
    is neither computed nor checked). A RefinementCriteria that names
    nothing subdivides nothing and collapses to the "min_depth" mesh.

    OR-logic applies: a cell is subdivided if *any* enabled metric exceeds
    its threshold.

    Note the metrics are evaluated on the first three input channels, which on
    the wing dataset are the (x, y, z) surface coordinates in metres -- not a
    normalised flow field. Thresholds calibrated for normalised fields are one to
    three orders of magnitude off there.

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
    # Use only velocity gradient and vorticity; the other eight stay disabled:
    cfg = RefinementCriteria(grad_threshold=0.05, vorticity_threshold=0.04)

    # Coarser mesh (raise all enabled thresholds by 2x):
    cfg = GEOMETRY_BALANCED_CRITERIA.scale(2.0)
    """

    # Geometry thresholds
    curvature_threshold:      Optional[float] = None
    le_te_threshold:          Optional[float] = None
    thickness_grad_threshold: Optional[float] = None
    wall_distance_threshold:  Optional[float] = None

    # Physics thresholds
    grad_threshold:      Optional[float] = None
    vorticity_threshold: Optional[float] = None
    momentum_threshold:  Optional[float] = None
    kh_shear_threshold:  Optional[float] = None
    variance_threshold:  Optional[float] = None
    entropy_threshold:   Optional[float] = None
 
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


# ---------------------------------------------------------------------------
# Available RefinementCriteria
# ---------------------------------------------------------------------------
# higher threshold = fewer patches
# lower threshold  = more patches


# ~1390 tokens. The best geometry-only criterion found: beats a uniform mesh of
# the same size on 88% of wings
GEOMETRY_BALANCED_CRITERIA = RefinementCriteria(
    entropy_threshold        = 4.10,     # uniform depth-5 floor
    wall_distance_threshold  = 450.0,    # one extra level at LE/TE
    le_te_threshold          = 3.0e-4,   # and at chordwise curvature peaks
)

# ~1070 tokens. Same family, cheapest setting that still beats uniform
GEOMETRY_BALANCED_LIGHT = RefinementCriteria(
    entropy_threshold        = 4.10,
    wall_distance_threshold  = 800.0,
)

# ~3730 tokens (cap is 4096). Dense: 97% of leaves at the finest depth
GEOMETRY_BALANCED_DENSE = RefinementCriteria(
    entropy_threshold        = 3.95,
    wall_distance_threshold  = 90.0,
    le_te_threshold          = 1.2e-4,
)

# ~264 tokens (floor is 256). Coarse: 97% of leaves at the coarsest depth
GEOMETRY_COARSE_CRITERIA = RefinementCriteria(
    wall_distance_threshold  = 450.0,
)

# Unstable on purpose: the same threshold gives 256 tokens on one wing and 3673
# on another
GEOMETRY_FRAGILE_CRITERIA = RefinementCriteria(
    grad_threshold           = 0.016,
)

# Wrong on purpose: refines the flat mid-chord and coarsens the leading edge
GEOMETRY_MISALIGNED_CRITERIA = RefinementCriteria(
    vorticity_threshold      = 0.0094,
)

# ~850 tokens. Stable budget but the mesh quality swings 7x between wings
GEOMETRY_LETE_CRITERIA = RefinementCriteria(
    le_te_threshold = 2.2e-4
)

# ~800 tokens. The best single metric by rank correlation and still 0.57x 
# a uniform mesh, because it can only ever build a two-level mesh
GEOMETRY_WALL_CRITERIA = RefinementCriteria(
    wall_distance_threshold = 150.0
)


CRITERIA_REGISTRY: Dict[str, RefinementCriteria] = {
    "GEOMETRY_BALANCED_CRITERIA":   GEOMETRY_BALANCED_CRITERIA,
    "GEOMETRY_BALANCED_LIGHT":      GEOMETRY_BALANCED_LIGHT,
    "GEOMETRY_BALANCED_DENSE":      GEOMETRY_BALANCED_DENSE,
    "GEOMETRY_COARSE_CRITERIA":     GEOMETRY_COARSE_CRITERIA,
    "GEOMETRY_FRAGILE_CRITERIA":    GEOMETRY_FRAGILE_CRITERIA,
    "GEOMETRY_MISALIGNED_CRITERIA": GEOMETRY_MISALIGNED_CRITERIA,
    "GEOMETRY_LETE_CRITERIA":       GEOMETRY_LETE_CRITERIA,
    "GEOMETRY_WALL_CRITERIA":       GEOMETRY_WALL_CRITERIA,
}