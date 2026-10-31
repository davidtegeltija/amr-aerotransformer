"""
Pick the mesh_perturbation amplitude before training, without training anything.

For each candidate amplitude, perturbs the scorer depth map of a few training rows
the way LearnedCollateFn does and reports, against the clean mesh:

* the fraction of pixels whose leaf depth changes. Aim for ~10-20 %: a scorer
  mesh from another condition of the same wing changes 16 % and took the
  learned model's CL error from 2.9 to 42 counts.
* the mean token count, which should stay near the clean one so the run
  trains on the same budget it is evaluated on.
* the time to build and reduce one perturbed mesh, which every training row now
  pays every epoch.
"""

import os
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.amr.learned_adaptive_mesh import build_depth_guided_mesh
from src.data.collate_fn import PERTURBATION_BLOCK, LearnedCollateFn, perturb_depth_map
from src.data.dataset_factory import build_dataset
from src.data.split import test_row_indices
from src.models.refinement_net import RefinementNet
from src.utils.checkpoint import build_model_from_checkpoint
from src.utils.config import load_config, resolve_depth_bounds


def leaf_depth_map(leaves, H: int, W: int) -> np.ndarray:
    """Paint each leaf's depth over its pixels.

    Args:
        leaves: Leaf ``QuadNode`` s tiling the grid.
        H, W: Grid size.

    Returns:
        ``[H, W]`` int array of leaf depths.
    """
    depth = np.zeros((H, W), dtype=np.int64)
    for leaf in leaves:
        depth[leaf.r0:leaf.r1, leaf.c0:leaf.c1] = leaf.depth
    return depth


def sample_train_rows(dataset, args, dataset_type: str, n_rows: int, seed: int) -> list:
    """Draw random rows outside the test split.

    Args:
        dataset: The full dataset.
        args: Merged config, for the split settings.
        dataset_type: As returned by ``build_dataset``.
        n_rows: Rows to draw.
        seed: RNG seed.

    Returns:
        List of dataset samples.
    """
    test_idx = set(test_row_indices(dataset, dataset_type, args.get("val_split"), args.get("seed", 42)))
    candidates = [i for i in range(len(dataset)) if i not in test_idx]
    rows = np.random.default_rng(seed).choice(candidates, n_rows, replace=False)
    return [dataset[int(i)] for i in rows]


@torch.no_grad()
def score_rows(scorer, samples: list, chunk: int = 32) -> np.ndarray:
    """Run the frozen scorer on the rows, ``chunk`` rows per CPU forward.

    Args:
        scorer: Trained ``RefinementNet``.
        samples: Dataset samples.
        chunk: Rows per forward, bounding the activation memory.

    Returns:
        ``[n_rows, H, W]`` depth maps.
    """
    scorer.eval()
    depth_maps = []
    for i in range(0, len(samples), chunk):
        grids = torch.stack([torch.from_numpy(np.asarray(s["input"], dtype=np.float32))
                             for s in samples[i:i + chunk]])
        depth_maps.append(scorer(grids).squeeze(1).numpy())
    return np.concatenate(depth_maps)


def check_amplitude(amplitude: float, samples: list, depth_maps: np.ndarray, collate: LearnedCollateFn,
                    draws: int, rng: np.random.Generator) -> None:
    """Print the depth-change fraction, token count and build time of one amplitude.

    Args:
        amplitude: Candidate ``mesh_perturbation``.
        samples: Dataset samples.
        depth_maps: Their clean scorer depth maps.
        collate: The training collate, used for the mesh build settings and timing.
        draws: Perturbed meshes per row.
        rng: Noise generator.
    """
    changed, clean_tokens, perturbed_tokens, seconds = [], [], [], []
    for s, depth_map in zip(samples, depth_maps):
        input = np.asarray(s["input"], dtype=np.float32)
        H, W = input.shape[:2]
        mesh_kwargs = dict(max_depth=collate.max_depth, min_depth=collate.min_depth, offset=collate.offset,
                           patch_sides_ratio=collate.scorer.patch_sides_ratio)

        clean = build_depth_guided_mesh(input, depth_map, **mesh_kwargs)
        clean_depth = leaf_depth_map(clean, H, W)
        clean_tokens.append(len(clean))

        for _ in range(draws):
            perturbed_map = perturb_depth_map(depth_map, amplitude, rng)
            perturbed = build_depth_guided_mesh(input, perturbed_map, **mesh_kwargs)
            changed.append((leaf_depth_map(perturbed, H, W) != clean_depth).mean())
            perturbed_tokens.append(len(perturbed))

            start = time.perf_counter()
            collate._tokenize(s, perturbed_map)
            seconds.append(time.perf_counter() - start)

    print(f"{amplitude:9.2f} | {100 * np.mean(changed):6.1f} % (p10 {100 * np.percentile(changed, 10):4.1f}, "
          f"p90 {100 * np.percentile(changed, 90):4.1f}) | {np.mean(clean_tokens):7.1f} -> {np.mean(perturbed_tokens):7.1f} "
          f"| {1000 * np.mean(seconds):6.1f} ms")


if __name__ == "__main__":
    data_config = "configs/data/wing.yaml"
    model_config = "configs/learned_transformer/tokens=512/learned_n=512_affine=1-2_quad_mesh_perturb.yaml"
    amplitudes = [0.5, 0.55, 0.6, 0.65]
    n_rows = 200   # Random rows ~ distinct wings; this sets the precision, not draws
    draws = 2

    args = load_config(model_config, data_config)
    dataset, dataset_type = build_dataset(args)
    args = resolve_depth_bounds(args, dataset)

    scorer = build_model_from_checkpoint(RefinementNet, args["checkpoint_file"])
    collate = LearnedCollateFn(scorer, min_depth=args["min_depth"], max_depth=args["max_depth"],
                               offset=args.get("offset", 0.0), affine_input=args["affine_input"],
                               affine_output=args["affine_output"], dense_loss=args["affine_output"] > 0)

    samples = sample_train_rows(dataset, args, dataset_type, n_rows, args.get("seed", 42))
    depth_maps = score_rows(scorer, samples)

    print(f"{n_rows} train rows x {draws} draws, {PERTURBATION_BLOCK}x{PERTURBATION_BLOCK} px noise blocks\n")
    print("amplitude | pixels changing depth          | tokens clean -> perturbed | build + reduce")
    rng = np.random.default_rng(0)
    for amplitude in amplitudes:
        check_amplitude(amplitude, samples, depth_maps, collate, draws, rng)
