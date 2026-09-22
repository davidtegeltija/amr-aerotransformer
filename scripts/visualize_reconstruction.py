import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.amr.adaptive_mesh import build_adaptive_mesh
from src.amr.learned_adaptive_mesh import build_depth_guided_mesh
from src.data.dataset_factory import build_dataset
from src.data.split import test_row_indices
from src.inference.predict import resolve_mesh_source
from src.utils.config import load_config, resolve_depth_bounds
from src.utils.plot import plot_reconstruction_by_depth_cumulative


# ------------------------------------------------------------------------
# One test-split sample and the mesh its config prescribes for it
# ------------------------------------------------------------------------
@torch.no_grad()
def build_mesh(model_config: str, data_config: str, sample_idx: int = -1):
    """Draw a test-split sample and build the mesh the config prescribes for it.

    No transformer takes part: the figure is a property of the mesh and the
    target alone, and the config already says which mesh that is -- the scorer
    checkpoint or the deterministic criteria, under the configured patch sizes.

    Args:
        model_config: Model config (``configs/*.yaml``), which names the mesh
            source and the patch-size bounds.
        data_config: Data config (``configs/data/*.yaml``) the sample is drawn from.
        sample_idx: Which sample of the test split to take.

    Returns:
        ``(sample, mesh, sample_index)``: the dataset row, the leaf ``QuadNode`` s
        tiling it, and that row's index in the full dataset.
    """
    args = load_config(model_config, data_config)
    dataset, dataset_type = build_dataset(args)
    args = resolve_depth_bounds(args, dataset)

    # The test split is replayed from the config, as in evaluate_model.py
    sample_index = test_row_indices(dataset, dataset_type, args.get("val_split"), args.get("seed", 42))[sample_idx]
    sample = dataset[sample_index]
    input_grid = np.asarray(sample["input"], dtype=np.float32)

    refinement_criteria, scorer = resolve_mesh_source(args)
    if scorer is None:
        mesh = build_adaptive_mesh(
            input_grid,
            refinement_criteria,
            max_depth=args["max_depth"],
            min_depth=args["min_depth"],
        )
    else:
        depth_map = scorer(torch.from_numpy(input_grid).unsqueeze(0)).squeeze(1)[0].numpy()
        mesh = build_depth_guided_mesh(
            input_grid,
            depth_map,
            max_depth=args["max_depth"],
            min_depth=args["min_depth"],
            offset=args.get("offset", 0.0),
        )

    return sample, mesh, sample_index


if __name__ == "__main__":
    model_config = "configs/learned_transformer_n=800_affine=2.yaml"
    data_config = "configs/data/wing.yaml"
    sample_idx = -1

    sample, mesh, sample_index = build_mesh(model_config, data_config, sample_idx)

    plot_reconstruction_by_depth_cumulative(
        sample["target"],
        mesh,
        orders=[0, 2],
        title=f"Mesh reconstruction of the training target  ({len(mesh)} cells, sample {sample_index})",
        save_path=f"outputs/plots/{Path(model_config).stem}_reconstruction_sample={sample_index}.png",
    )
