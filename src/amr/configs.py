# higher threshold = fewer patches
# lower threshold  = more patches

from src.amr.physics_metrics import RefinementCriteria


AERODYNAMIC_CRITERIA = RefinementCriteria(
    grad_threshold      = 0.08,   # primary: catches shocks, BL, wakes
    vorticity_threshold = 0.06,   # secondary: vortex cores
    momentum_threshold  = 0.80,   # conservative: avoid over-refining fast flow
    kh_shear_threshold  = 0.60,   # conservative
    variance_threshold  = 0.02,   # general fallback
    entropy_threshold   = 2.50,   # general fallback
)


AERODYNAMIC_CRITERIA_2 = RefinementCriteria(
    grad_threshold      = 0.15,   # primary: catches shocks, BL, wakes
    vorticity_threshold = 0.10,   # secondary: vortex cores
    momentum_threshold  = 1.20,   # conservative: avoid over-refining fast flow
    kh_shear_threshold  = 0.80,   # conservative
    variance_threshold  = 0.05,   # general fallback
    entropy_threshold   = 7.50,   # general fallback
)
 

DEFAULT_CRITERIA = RefinementCriteria(
    grad_threshold      = 0.05,
    vorticity_threshold = 0.035,
    momentum_threshold  = 0.40,
    kh_shear_threshold  = 0.25,
    variance_threshold  = 0.015,
    entropy_threshold   = 2.50,
)


FIRST_DEFAULT_CRITERIA = RefinementCriteria(
    # Individual metric thresholds — calibrated for normalised physical fields
    # in range ~[-2, 2].  Tune these to control coarseness vs. fine resolution.
    grad_threshold      = 1.15,   # mean velocity-gradient magnitude
    vorticity_threshold = 0.10,   # mean vorticity magnitude
    momentum_threshold  = 0.50,   # momentum per unit area
    kh_shear_threshold  = 0.80,   # max KH shear strength
    variance_threshold  = 0.05,   # mean per-channel variance
    entropy_threshold   = 4.5,    # mean Shannon entropy (bits)
)