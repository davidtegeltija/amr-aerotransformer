import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.amr.refinement_criteria import RefinementCriteria, CRITERIA_REGISTRY
from src.data.synthetic_dataset import make_synthetic_field
from src.amr.adaptive_mesh import build_adaptive_mesh, mesh_statistics
from src.amr.learned_adaptive_mesh import build_depth_guided_mesh
from src.models.refinement_net import RefinementNet
from src.utils.checkpoint import build_model_from_checkpoint
from src.utils.plot import (
    plot_mesh,
    plot_mesh_by_depth,
    plot_mesh_by_depth_cumulative,
    plot_mesh_comparison,
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


def load_wing_row(row):
    """Scorer input (geometry + broadcast operating conditions) and target of one SuperWing row, read through mmap."""
    index = np.load("/mnt/data/tegeltija/index.npy")
    geometry = np.load("/mnt/data/tegeltija/geom0.npy", mmap_mode="r")[int(index[row, 0])].transpose(2, 1, 0)
    conditions = np.broadcast_to(index[row, [2, 3]], (*geometry.shape[:2], 2))
    target = np.load("/mnt/data/tegeltija/data.npy", mmap_mode="r")[row].transpose(2, 1, 0)
    return np.concatenate([geometry, conditions], axis=-1), np.asarray(target)


def build_learned_mesh(x, scorer, min_depth, max_depth):
    """Mesh built from the scorer's predicted depth map of input x."""
    with torch.no_grad():
        depth_map = scorer(torch.from_numpy(x.astype(np.float32))[None]).squeeze(1)[0].numpy()
    return build_depth_guided_mesh(x, depth_map, min_depth=min_depth, max_depth=max_depth,
                                   patch_sides_ratio=scorer.patch_sides_ratio)


def build_threshold_meshes(row, absolute, relative, scorer, min_depth, max_depth):
    """Ground truth and the absolute, relative and learned mesh of one SuperWing row, read through mmap."""
    x, target = load_wing_row(row)
    meshes = {
        "Absolute thresholds": build_adaptive_mesh(x, absolute, min_depth=min_depth, max_depth=max_depth),
        "Relative thresholds": build_adaptive_mesh(x, relative, min_depth=min_depth, max_depth=max_depth),
        "Learned (RefinementNet)": build_learned_mesh(x, scorer, min_depth, max_depth),
    }
    return target, meshes


def criteria_demo(data, sample_index, criteria_name, min_depth, max_depth, show_plots, save_path, scorer=None):
    """Mesh plots of one sample under a registry preset. save_path is a directory, or None to not save.

    With a scorer the mesh is the learned one instead: sample_index is then a SuperWing
    row (see load_wing_row), and data and criteria_name are unused.
    """
    if scorer is None:
        sample, mesh = create_mesh(data=data, sample_index=sample_index, max_depth=max_depth, min_depth=min_depth, refinement_criteria=CRITERIA_REGISTRY[criteria_name])
    else:
        x, sample = load_wing_row(sample_index)
        mesh = build_learned_mesh(x, scorer, min_depth, max_depth)
        print(f"Learned mesh for row {sample_index}: {len(mesh)} patches")

    # --- Plotting ---
    name = criteria_name if scorer is None else "learned"
    prefix = f"{save_path}/{name}" if save_path else None

    # Main mesh overlay
    save_path_mesh = f"{prefix}_adaptive_mesh.png" if save_path else None
    title_mesh = f"Adaptive Mesh ({len(mesh)} cells)"
    plot_mesh(sample, mesh, channel=0, title=title_mesh, show=show_plots, save_path=save_path_mesh)

    # Token-matched uniform mesh: the uniform preset and depth whose patch count
    # is closest to the adaptive mesh's actual patch count.
    uniform_meshes = [build_adaptive_mesh(sample, CRITERIA_REGISTRY[name], min_depth=depth, max_depth=depth)
                      for name in ("UNIFORM", "UNIFORM_1x1", "UNIFORM_1x2") for depth in range(max_depth + 1)]
    mesh_uniform = min(uniform_meshes, key=lambda m: abs(len(m) - len(mesh)))

    save_path_uniform = f"{prefix}_uniform_mesh.png" if save_path else None
    title_uniform = f"Uniform Mesh ({len(mesh_uniform)} cells)"
    plot_mesh(sample, mesh_uniform, channel=0, title=title_uniform, show=show_plots, save_path=save_path_uniform)

    # Per-depth subplot
    save_path_depth = f"{prefix}_mesh_by_depth.png" if save_path else None
    title_depth = "Adaptive Mesh by Depth"
    plot_mesh_by_depth(sample, mesh, title=title_depth, show=show_plots, save_path=save_path_depth)

    # Cumulative per-depth subplot
    save_path_depth_cumulative = f"{prefix}_mesh_by_depth_cumulative.png" if save_path else None
    title_depth_cumulative = "Adaptive Mesh by Cumulative Depth"
    plot_mesh_by_depth_cumulative(sample, mesh, title=title_depth_cumulative, show=show_plots, save_path=save_path_depth_cumulative)

    # Metric heatmap (learned leaves carry no metrics)
    if scorer is None:
        save_path_heatmap = f"{prefix}_velocity_gradient.png" if save_path else None
        title_heatmap = "Velocity Gradient Magnitude per Patch"
        plot_metric_heatmap(sample, mesh, metric_name="velocity_gradient", title=title_heatmap, show=show_plots, save_path=save_path_heatmap)

    # Patch feature reconstruction
    save_path_reconstruction = f"{prefix}_reconstruction.png" if save_path else None
    title_features = "AMR Patch Reconstruction  (velocity_x)"
    plot_patch_features(sample, mesh, channel=0, title=title_features, show=show_plots, save_path=save_path_reconstruction)


if __name__ == "__main__":
    min_depth, max_depth = 2, 6
    show_plots = True

    # Registry preset on one wing sample
    data = np.load("data/crmmdata.npy")
    sample_index = 0
    criteria_name = "AERODYNAMIC_CRITERIA_2"
    save_path = "outputs/plots/"
    # criteria_demo(data, sample_index, criteria_name, min_depth, max_depth, show_plots, save_path)

    # Learned mesh on one SuperWing row (17612: depth-5 cells trace the upper-surface shock)
    scorer = build_model_from_checkpoint(RefinementNet, "outputs/checkpoints/2026-11-09_scorer_n=512_rectangular.pt").eval()
    # criteria_demo(None, 20337, None, min_depth, max_depth, show_plots, save_path, scorer=scorer)

    # Compare the meshes of different refinement rules on one wing sample
    row = 4287
    absolute = RefinementCriteria(le_te_threshold=2.4e-4)
    relative = RefinementCriteria(wall_distance_threshold=1.1, le_te_threshold=1.1, relative_thresholds=True)
    scorer = build_model_from_checkpoint(RefinementNet, "outputs/checkpoints/2026-10-11_scorer_n=512_quad.pt").eval()
    compare_save_path = "outputs/plots/threshold_meshes.png"   # a file, e.g. "outputs/plots/threshold_meshes.png"

    target, meshes = build_threshold_meshes(row, absolute, relative, scorer, min_depth, max_depth)
    plot_mesh_comparison(target[..., 0], meshes, label="$C_p$", show=show_plots, save_path=compare_save_path)
