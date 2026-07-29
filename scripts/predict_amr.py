import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.config_utils import load_config
from src.utils.data_utils import build_dataset, test_row_indices
from src.utils.geometry_utils import patch_sizes_to_depth_bounds
from src.amr.adaptive_mesh import build_adaptive_mesh
from src.amr.learned_adaptive_mesh import build_depth_guided_mesh
from src.amr.quadtree import nodes_to_token_array
from src.amr.refinement_criteria import CRITERIA_REGISTRY
from src.model.amr_model import AdaptiveMeshAeroModel
from src.model.refinement_net import RefinementNet
from src.model.reconstruction import tokens_to_grid, tokens_to_grid_affine
from src.utils.mesh_visualization import plot_mesh
from src.utils.model_utils import build_model_from_checkpoint
from src.utils.prediction_visualization import plot_flow_comparison, plot_3d_prediction


def create_model(args, checkpoint_file, dataset):
    """Rebuild the trained transformer model from its checkpoint.

    The architecture (d_model, n_layers, affine_output, ...) comes from the
    checkpoint itself, so it always matches the weights. ``args`` still supplies
    the *mesh* settings, which are not part of the model: the depth bounds
    derived here are what ``predict_single`` rebuilds the quadtree with.
    """
    # Configs express bounds as patch sizes; convert once to integer depths (mirrors
    # main.py) and inject back into args so predict_single reuses the same values.
    H, W = dataset.H, dataset.W
    min_depth, max_depth = patch_sizes_to_depth_bounds(H, W, args.get("min_patch_size"), args.get("max_patch_size"))
    args["min_depth"] = min_depth
    args["max_depth"] = max_depth

    model = build_model_from_checkpoint(AdaptiveMeshAeroModel, checkpoint_file)
    model.eval()
    return model, args


def create_scorer(scorer_checkpoint):
    """Load a standalone frozen ``RefinementNet`` for learned-mesh inference."""
    if not scorer_checkpoint:
        raise ValueError(
            "learned-mesh inference needs a scorer checkpoint; set 'scorer_checkpoint_file' "
            "in the config to a trained RefinementNet checkpoint.")
    return build_model_from_checkpoint(RefinementNet, scorer_checkpoint).eval()


@torch.no_grad()
def predict_single(model, args, sample):
    """Run one forward pass on a single sample and reconstruct the full-grid prediction.

    The mesh is built here (mirroring the collate functions) and the transformer
    only consumes packed tokens:
      * deterministic -> build_adaptive_mesh (physics AMR criterion)
      * learned       -> ``scorer`` depth map -> build_depth_guided_mesh
    """
    input_grid = sample["input"]                  # [H, W, C] numpy
    H, W, C = input_grid.shape
    output_channels = model.output_channels

    if args.get("model_trained") == "deterministic_transformer":
        leaves = build_adaptive_mesh(
            input_grid,
            max_depth=args.get("max_depth"),
            min_depth=args.get("min_depth"),
            refinement_criteria=CRITERIA_REGISTRY[args.get("refinement_criteria")],
        )
        token_array = nodes_to_token_array(leaves, H, W, C)
    else:
        # Learned mesh needs a frozen scorer to build the mesh at inference time.
        scorer = create_scorer(args.get("checkpoint_file"))
        grid = torch.from_numpy(np.asarray(input_grid, dtype=np.float32)).unsqueeze(0)  # [1, H, W, C]
        depth_map = scorer(grid).squeeze(1)[0].numpy()                                  # [H, W]
        leaves = build_depth_guided_mesh(
            data=np.asarray(input_grid, dtype=np.float32), depth_map=depth_map,
            max_depth=args.get("max_depth"), min_depth=args.get("min_depth"),
            offset=args.get("offset", 0.0))
        token_array = nodes_to_token_array(leaves, H, W, C)

    packed_tokens = torch.from_numpy(token_array).float()
    out = model(packed_tokens, [len(leaves)])

    token_preds = out["token_preds"]
    if model.affine_output:
        # token_preds is [N, C, 3] = (value, gx, gy); decode the per-cell ramps.
        grid = tokens_to_grid_affine(token_preds, leaves, H, W, output_channels)
    else:
        grid = tokens_to_grid(token_preds, leaves, H, W, output_channels, mode="fill")
    return {
        "input_grid": input_grid,
        "ground_truth": sample["target"],
        "prediction": grid.cpu().numpy(),
        "token_preds": token_preds,
        "mesh": leaves,
    }


if __name__ == "__main__":
    model_config = "configs/learned_transformer.yaml"
    data_config = "configs/data/wing.yaml"
    checkpoint_file = "outputs/checkpoints/transformer_on_learned_mesh.pt"

    args = load_config(model_config, data_config)
    dataset, dataset_type = build_dataset(args)

    model, args = create_model(args, checkpoint_file, dataset)

    # Predict on the held-out test split, replayed from the config, so the plots
    # show a geometry (or cavity case) the model never trained on. The last test
    # row is the final frame of its case: the flow has fully developed, so the
    # quadtree actually refines. Early frames are near rest and collapse to a
    # single root token.
    test_idx = test_row_indices(dataset, dataset_type, args.get("val_split"), args.get("seed", 42))
    sample_index = test_idx[-1]
    sample = dataset[sample_index]

    result = predict_single(model, args, sample)
    input_grid = result["input_grid"]
    ground_truth = result["ground_truth"]
    prediction = result["prediction"]
    token_preds = result["token_preds"]
    mesh = result["mesh"]
    
    # --- Plotting Mesh ---
    plot_mesh(input_grid, mesh, show=False, save_path=f"outputs/plots/amr_mesh_sample={sample_index}.png")

    # --- Plotting Flow ---
    plot_flow_comparison(ground_truth, prediction, save_path=f"outputs/plots/prediction_test_sample={sample_index}.png")
    # plot_3d_prediction(sample["input"], prediction)

