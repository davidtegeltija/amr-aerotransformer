import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.amr.refinement_criteria import RefinementCriteria, CRITERIA_REGISTRY
from src.data.synthetic_dataset import make_synthetic_field
from src.amr.adaptive_mesh import build_adaptive_mesh, mesh_statistics
from src.utils.plot import (
    plot_mesh,
    plot_mesh_by_depth,
    plot_metric_heatmap,
    plot_patch_features,
)


def create_mesh(data, sample_index, max_depth, min_depth, refinement_criteria: RefinementCriteria):

    # 1. Generate synthetic batch data
    if data is None:
        data = make_synthetic_field(n_samples=9, channels=3, height=128, width=256)
        print(f"Generating synthetic fluid field of shape {data.shape} ...")
    else:
        print(f"Reading your data of shape {data.shape} ...")


    # 2. Build adaptive mesh for one sample
    print(f"\nBuilding adaptive mesh for sample {sample_index} ...")
    sample = data[sample_index]   # shape: (3, 128, 256)
    sample = sample.transpose(2, 1, 0)

    # Ad-hoc demo: depths are hardcoded here. Production config is patch-size based
    # (min_patch_size / max_patch_size, converted by patch_sizes_to_depth_bounds).
    mesh = build_adaptive_mesh(
        sample,
        max_depth=max_depth,
        min_depth=min_depth,
        refinement_criteria=refinement_criteria,
    )

    stats = mesh_statistics(mesh)
    print(f"Total patches : {stats['total_patches']}")
    print(f"Depth range   : {stats['depth_range']}")

    # Show a few example patches
    print("\nFirst 3 patches:")
    for p in mesh[:3]:
        print(f"depth={p.depth}  bbox={p.bbox} size={p.area()}  center=({p.center[0]:.1f},{p.center[1]:.1f})")
    return sample, mesh


if __name__ == "__main__":
    data = np.load("data/crmmdata.npy")
    sample_index = 0
    max_depth = 6
    min_depth = 2
    criteria_name = "AERODYNAMIC_CRITERIA_2"

    sample, mesh = create_mesh(data=data, sample_index=sample_index, max_depth=max_depth, min_depth=min_depth, refinement_criteria=CRITERIA_REGISTRY[criteria_name])

    # --- Plotting ---
    show_plots = True
    # save_path = "outputs/plots"
    save_path = None
    prefix = f"{save_path}/{criteria_name}" if save_path else None

    # Main mesh overlay
    save_path_mesh = f"{prefix}_adaptive_mesh.png" if save_path else None
    title_mesh = "Adaptive Mesh (threshold=0.15, max_depth=6)"
    plot_mesh(sample, mesh, title=title_mesh, show=show_plots, save_path=save_path_mesh)

    # Per-depth subplot
    save_path_depth = f"{prefix}_mesh_by_depth.png" if save_path else None
    title_depth = "Adaptive Mesh by Depth"
    plot_mesh_by_depth(sample, mesh, title=title_depth, show=show_plots, save_path=save_path_depth)

    # Metric heatmap
    save_path_heatmap = f"{prefix}_velocity_gradient.png" if save_path else None
    title_heatmap = "Velocity Gradient Magnitude per Patch"
    plot_metric_heatmap(sample, mesh, metric_name="velocity_gradient", title=title_heatmap, show=show_plots, save_path=save_path_heatmap)

    # Patch feature reconstruction
    save_path_reconstruction = f"{prefix}_reconstruction.png" if save_path else None
    title_features = "AMR Patch Reconstruction  (velocity_x)"
    plot_patch_features(sample, mesh, channel=0, title=title_features, show=show_plots, save_path=save_path_reconstruction)


    # 5. Demonstrate configurable thresholds
    # print("\n[5] Comparing coarse vs. fine threshold ...")

    # mesh_coarse = build_adaptive_mesh(sample, max_depth=4, threshold=0.30)
    # mesh_fine   = build_adaptive_mesh(sample, max_depth=6, threshold=0.08)

    # print(f"    Coarse (threshold=0.30, max_depth=4): {mesh_statistics(mesh_coarse)['total_patches']} patches")
    # print(f"    Fine   (threshold=0.08, max_depth=6): {mesh_statistics(mesh_fine)['total_patches']} patches")

    # fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # bg = sample[0]   # velocity_x channel for background

    # for ax, m, label in zip(axes, [mesh_coarse, mesh_fine], ["Coarse (thresh=0.30, depth≤4)", "Fine   (thresh=0.08, depth≤6)"]):
    #     ax.imshow(bg, cmap="viridis", origin="upper")
    #     for patch in m:
    #         r0, c0, r1, c1 = patch["bbox"]
    #         rect = plt.Rectangle(
    #             (c0, r0), c1 - c0, r1 - r0,
    #             linewidth=0.5, edgecolor="white", facecolor="none", alpha=0.8,
    #         )
    #         ax.add_patch(rect)
    #     ax.set_title(f"{label}\n{len(m)} patches")
    #     ax.set_aspect("equal")

    # fig.suptitle("Threshold Sensitivity", fontsize=12)
    # plt.tight_layout()
    # if save_path:            
    #     fig.savefig(f"{save_path}/06_threshold_comparison-{timestamp}.png", dpi=150, bbox_inches="tight")
    #     print(f"    Saved -> {save_path}/06_threshold_comparison.png")

    # 6. Multi-channel demo
    # print("\n[6] Demonstrating 5-channel support ...")
    # data_5ch = make_synthetic_field(n_samples=1, channels=5, height=64, width=128)
    # # Channels: 0=density, 1=pressure, 2=temp, 3=vel_x, 4=vel_y  (hypothetical)
    # mesh_5ch = build_adaptive_mesh(data_5ch[0], max_depth=5, refinement_criteria=refinement_criteria)
    # st5 = mesh_statistics(mesh_5ch)
    # print(f"    5-channel field (1, 5, 64, 128): "
    #       f"{st5['total_patches']} patches, depth {st5['depth_range']}")
