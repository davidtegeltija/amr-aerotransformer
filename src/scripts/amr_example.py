"""
========================================================================
Demonstrates the full adaptive mesh pipeline on synthetic fluid field data.
========================================================================

Because of the project
In order to treat the whole src/ folder as a package, and therefore correctly 
resolve imports relative to the project root, you must run this script with the following 
bash command:
    python -m src.scripts.amr_example

The script generates a realistic synthetic field with:
    - A shockwave-like velocity discontinuity (high gradient → fine mesh)
    - A vortex region (high vorticity → fine mesh)
    - A smooth background (low variability → coarse mesh)
    - Additive noise in one quadrant

Or use your own data and produce the results with that.

Then builds an adaptive mesh using different configurations and visualises
the results.
"""

import os

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from src.data.synthetic_dataset import make_synthetic_field
from src.amr.adaptive_mesh import build_adaptive_mesh, process_batch, mesh_statistics
from src.utils.mesh_visualization import (
    visualize_mesh,
    visualize_mesh_by_depth,
    visualize_metric_heatmap,
    visualize_patch_features,
)


def main(refinement_criteria=None, data=None, sample_number=None, batch=False, show_plots=False, save_path=None):

    # 1. Generate synthetic batch data
    if data is None:
        np.random.seed(42)
        data = make_synthetic_field(n_samples=9, channels=3, height=128, width=256)
        print(f"[1] Generating synthetic fluid field of shape {data.shape} ...")
    else:
        print(f"[1] Reading your data of shape {data.shape} ...")


    # 2. Build adaptive mesh for one sample
    sample_number = 0
    print(f"\n[2] Building adaptive mesh for sample {sample_number} ...")
    sample = data[sample_number]   # shape: (3, 128, 256)
    sample = sample.transpose(0, 2, 1)

    mesh = build_adaptive_mesh(
        sample,
        max_depth=6,
        min_cell_size=4,
        refinement_criteria=refinement_criteria,
    )

    stats = mesh_statistics(mesh)
    print(f"    Total patches      : {stats['total_patches']}")
    print(f"    Depth range        : {stats['depth_range']}")

    # Show a few example patches
    # print("\n    First 3 patches:")
    # for p in mesh[:3]:
    #     print(f"      depth={p['depth']}  bbox={p['bbox']}  "
    #           f"size={p['size']}  center=({p['center'][0]:.1f},{p['center'][1]:.1f})")


    # 3. Process full batch
    if batch:
        print(f"\n    Processing full batch of {data.shape[0]} samples ...")
        all_meshes = process_batch(data, max_depth=5, refinement_criteria=refinement_criteria)
        for i, m in enumerate(all_meshes):
            st = mesh_statistics(m)
            print(f"    sample {i}: {st['total_patches']:4d} patches,  "
                f"depth {st['depth_range'][0]}–{st['depth_range'][1]}")


    # ---------------------------------------------------------------------------
    # 4. Visualizations
    # ---------------------------------------------------------------------------
    print("\n[3] Generating visualizations")
    if save_path:
        true_save_path = os.path.join(save_path, "")
        os.makedirs(os.path.dirname(true_save_path), exist_ok=True)
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M")
        save_path_mesh = f"{save_path}/01_adaptive_mesh-{timestamp}.png"
        save_path_depth = f"{save_path}/02_mesh_by_depth-{timestamp}.png"
        save_path_velocity_heatmap = f"{save_path}/03_velocity_gradient-{timestamp}.png"
        save_path_vorticity_heatmap = f"{save_path}/04_vorticity-{timestamp}.png"
        save_path_reconstruction = f"{save_path}/05_reconstruction-{timestamp}.png"

    # 4a. Main mesh overlay
    visualize_mesh(
        sample, mesh,
        title="Adaptive Mesh  (threshold=0.15, max_depth=6)",
        show=show_plots,
        save_path=save_path_mesh if save_path is not None else None, # type: ignore
    )
    plt.close("all")

    # 4b. Per-depth subplot
    visualize_mesh_by_depth(
        sample, mesh,
        title="Adaptive Mesh by Depth",
        show=show_plots,
        save_path=save_path_depth if save_path is not None else None, # type: ignore
    )
    plt.close("all")

    # 4c. Metric heatmap
    visualize_metric_heatmap(
        sample, mesh,
        metric_name="velocity_gradient",
        title="Velocity Gradient Magnitude per Patch",
        show=show_plots,
        save_path=save_path_velocity_heatmap if save_path is not None else None, # type: ignore
    )
    plt.close("all")

    # 4e. Patch feature reconstruction
    visualize_patch_features(
        sample, mesh,
        channel=0,
        title="AMR Patch Reconstruction  (velocity_x)",
        show=show_plots,
        save_path=save_path_reconstruction if save_path is not None else None, # type: ignore
    )
    plt.close("all")
    if save_path:
        print(f"[Done]  All outputs saved to {save_path}")


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


if __name__ == "__main__":

    from src.amr.refinement_criteria import AERODYNAMIC_CRITERIA, AERODYNAMIC_CRITERIA_2, GEOMETRY_ONLY_COMBINED_CONFIG
    
    # data = np.load("data/crmmdata.npy")
    data = np.load("data/crmmgeom.npy")

    show_plots = True
    save_path = "outputs/plots"
    # save_path = None

    main(refinement_criteria=GEOMETRY_ONLY_COMBINED_CONFIG, data=data, show_plots=show_plots, save_path=save_path, batch=True)
