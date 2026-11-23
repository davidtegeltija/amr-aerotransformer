"""Reconstruction floor of a mesh and its comparison to a uniform mesh of the same size.

The floor is the NMSE left after fitting the per-cell polynomial of the given
affine_output order to the ground truth by least squares: the lowest error any
model with that output order can reach on the mesh. Since the floor drops with
every added token, a mesh is judged by the uniform floor at its own token count
divided by its floor, so above 1 means the mesh placed its cells better than a
uniform one would.

Run from the project root: python scripts/mesh_floor.py
"""
import os
import sys
from collections import defaultdict

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.amr.adaptive_mesh import build_adaptive_mesh
from src.amr.learned_adaptive_mesh import build_depth_guided_mesh
from src.amr.refinement_criteria import CRITERIA_REGISTRY, RefinementCriteria
from src.data.dataset_factory import build_dataset
from src.data.split import test_row_indices
from src.evaluation.metrics import l2_error
from src.inference.predict import resolve_mesh_source
from src.models.reconstruction import basis_size, cell_basis
from src.models.refinement_net import RefinementNet
from src.utils.checkpoint import build_model_from_checkpoint
from src.utils.config import load_config, resolve_depth_bounds
from visualize_mesh import build_learned_mesh, build_threshold_meshes, load_wing_row


def mesh_fit(target, leaves, order=2):
    """Best per-cell polynomial fit of the target, by least squares.

    Args:
        target: [H, W, C] ground truth.
        leaves: Leaf QuadNodes tiling the grid.
        order: affine_output order of the fitted polynomial, 0, 1 or 2.

    Returns:
        [H, W, C] fitted field.
    """
    C = target.shape[-1]
    by_shape = defaultdict(list)
    for node in leaves:
        by_shape[(node.height, node.width)].append(node)

    fit = np.empty(target.shape, dtype=np.float64)
    for (h, w), nodes in by_shape.items():
        basis = cell_basis(h, w, basis_size(order))                                         # [h*w, K]
        # The columns are orthogonal, so the least-squares coefficients are plain projections.
        # A term the cell cannot resolve is a zero column and drops out.
        norms = (basis ** 2).sum(axis=0)
        norms[norms == 0] = np.inf
        cells = np.stack([target[n.r0:n.r1, n.c0:n.c1].reshape(-1, C) for n in nodes])     # [n, h*w, C]
        coefficients = np.einsum("pk,npc->nkc", basis, cells) / norms[:, None]
        for node, cell in zip(nodes, np.einsum("pk,nkc->npc", basis, coefficients)):
            fit[node.r0:node.r1, node.c0:node.c1] = cell.reshape(h, w, C)
    return fit


def mesh_floor(target, leaves, order=2):
    """NMSE of the best per-cell polynomial fit of the target, normalised per channel like the training loss."""
    mse = ((target - mesh_fit(target, leaves, order)) ** 2).mean(axis=(0, 1))
    return float(np.mean(mse / target.var(axis=(0, 1))))


@torch.no_grad()
def mesh_l2_floors(model_config, data_config="configs/data/wing.yaml", orders=(0, 1, 2), overrides=None):
    """Per-channel relative L2 floor of every output order on the mesh of a config, over its test split.

    Averaged per sample like evaluate_error_rate, so the floor reads on the same
    scale as the l2 table of the evaluation files.

    """
    args = load_config(model_config, data_config)
    args.update(overrides or {})
    dataset, dataset_type = build_dataset(args)
    args = resolve_depth_bounds(args, dataset)
    criteria, scorer = resolve_mesh_source(args)
    test_idx = test_row_indices(dataset, dataset_type, args.get("val_split"), args.get("seed", 42))

    errors = defaultdict(list)
    for i, index in enumerate(test_idx, 1):
        sample = dataset[index]
        input_grid = np.asarray(sample["input"], dtype=np.float32)
        if scorer is None:
            leaves = build_adaptive_mesh(input_grid, criteria, max_depth=args["max_depth"], min_depth=args["min_depth"])
        else:
            depth_map = scorer(torch.from_numpy(input_grid).unsqueeze(0)).squeeze(1)[0].numpy()
            leaves = build_depth_guided_mesh(input_grid, depth_map, max_depth=args["max_depth"], min_depth=args["min_depth"],
                                             offset=args.get("offset", 0.0), patch_sides_ratio=scorer.patch_sides_ratio)
        target = np.asarray(sample["target"], dtype=np.float64)
        for order in orders:
            errors[order].append(l2_error(mesh_fit(target, leaves, order), target)[0])
        if i % 250 == 0:
            print(f"{i}/{len(test_idx)} test samples", flush=True)

    return {order: np.mean(values, axis=0) for order, values in errors.items()}


def learned_token_counts(scorer_file, rows, min_depth=2, max_depth=6):
    """Token count of the learned mesh of a scorer on every given SuperWing row."""
    scorer = build_model_from_checkpoint(RefinementNet, scorer_file).eval()
    return np.array([len(build_learned_mesh(load_wing_row(row)[0], scorer, min_depth, max_depth)) for row in rows])


def uniform_floors(target, min_depth, max_depth, order=2, criteria="UNIFORM"):
    """Token counts and floors of the uniform meshes at every depth from min_depth to max_depth.

    criteria names the uniform preset, which sets the cell shape.
    """
    # A uniform mesh depends only on the grid shape, so the target can stand in for the input
    meshes = [build_adaptive_mesh(target, CRITERIA_REGISTRY[criteria], min_depth=d, max_depth=d)
              for d in range(min_depth, max_depth + 1)]
    return [len(m) for m in meshes], [mesh_floor(target, m, order) for m in meshes]


def cell_shape_floors(rows, min_depth=2, max_depth=5, order=2):
    """Token counts and mean floors of the uniform 2:1, 1:1 and 1:2 meshes over the given SuperWing rows."""
    presets = {"2:1": "UNIFORM_2x1", "1:1": "UNIFORM_1x1", "1:2": "UNIFORM_1x2"}
    # A uniform mesh has the same token count on every row, so only the floors are averaged
    counts, floors = {}, defaultdict(list)
    for i, row in enumerate(rows, 1):
        _, target = load_wing_row(row)
        for shape, criteria in presets.items():
            counts[shape], row_floors = uniform_floors(target, min_depth, max_depth, order, criteria)
            floors[shape].append(row_floors)
        if i % 25 == 0:
            print(f"{i}/{len(rows)} rows", flush=True)
    return {shape: (counts[shape], np.mean(floors[shape], axis=0)) for shape in presets}


def uniform_floor_at(n_tokens, counts, floors):
    """Uniform floor at n_tokens, interpolated log-log between the uniform depths."""
    return float(np.exp(np.interp(np.log(n_tokens), np.log(counts), np.log(floors))))


def sample_rows(n_samples, seed=0):
    """One random simulation row of each of n_samples randomly drawn wings of the SuperWing index."""
    geometry_ids = np.load("/mnt/data/tegeltija/index.npy")[:, 0].astype(int)
    rng = np.random.default_rng(seed)
    wings = rng.choice(np.unique(geometry_ids), size=n_samples, replace=False)
    return [int(rng.choice(np.flatnonzero(geometry_ids == wing))) for wing in wings]


if __name__ == "__main__":
    # n_samples counts wings (4239), one random operating condition each.
    # For every simulation of every wing use rows = list(range(28856)) instead
    n_samples = 300
    rows = sample_rows(n_samples)
    min_depth, max_depth = 2, 6
    order = 2
    absolute = RefinementCriteria(le_te_threshold=2.4e-4)
    relative = RefinementCriteria(wall_distance_threshold=1.1, le_te_threshold=1.1, relative_thresholds=True)
    scorer = build_model_from_checkpoint(RefinementNet, "outputs/checkpoints/2026-10-11_scorer_n=512_quad.pt").eval()

    # Mesh name -> per-row token count, floor and uniform floor / floor
    stats = defaultdict(lambda: defaultdict(list))
    for i, row in enumerate(rows, 1):
        target, meshes = build_threshold_meshes(row, absolute, relative, scorer, min_depth, max_depth)
        counts, floors = uniform_floors(target, min_depth, max_depth, order)
        for name, leaves in meshes.items():
            floor = mesh_floor(target, leaves, order)
            stats[name]["tokens"].append(len(leaves))
            stats[name]["floor"].append(floor)
            stats[name]["vs_uniform"].append(uniform_floor_at(len(leaves), counts, floors) / floor)
        if i % 25 == 0:
            print(f"{i}/{len(rows)} rows", flush=True)

    print(f"\n{'Mesh':26s} {'Samples':>7s} {'Tokens':>12s} {'Floor (1e-3)':>14s} {'vs. uniform':>13s} {'> uniform':>9s}")
    for name, s in stats.items():
        tokens, floor, ratio = (np.array(s[key]) for key in ("tokens", "floor", "vs_uniform"))
        print(f"{name:26s} {len(rows):7d} {tokens.mean():6.0f} +- {tokens.std():3.0f} "
              f"{floor.mean() * 1e3:6.1f} +- {floor.std() * 1e3:4.1f} {ratio.mean():5.2f} +- {ratio.std():4.2f} "
              f"{(ratio > 1).mean() * 100:8.0f}%")
