

import numpy as np

from src.amr.quadtree import collect_leaves
from src.amr.refinement_criteria import AERODYNAMIC_CRITERIA, AERODYNAMIC_CRITERIA_2, GEOMETRY_ONLY_COMBINED_CONFIG, AERODYNAMIC_COMBINED_CONFIG
from src.amr.adaptive_mesh import build_adaptive_mesh
from src.utils.mesh_visualization import (
    visualize_mesh,
    visualize_mesh_by_depth,
    visualize_metric_heatmap,
    visualize_patch_features,
)


if __name__ == "__main__":
    data = np.load("data/crmmdata.npy")
    sample = data[0]
    sample = sample.transpose(2, 1, 0)

    token_list = build_adaptive_mesh(
        sample,
        max_depth=8,
        min_cell_size=2,
        refinement_criteria=AERODYNAMIC_CRITERIA_2,
    )
 
    save_path = "outputs/plots3"
    # save_path = None

    visualize_mesh(sample, token_list, save_path=save_path)
    # visualize_mesh_by_depth(sample, token_list, save_path=save_path)
    # visualize_patch_features(sample, token_list, channel=0, save_path=save_path)
