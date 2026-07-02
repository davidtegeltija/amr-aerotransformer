import os
import random
import sys

import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.geometry_utils import patch_sizes_to_depth_bounds
from src.amr.quadtree_tokenizer import QuadtreeTokenizer
from src.amr.refinement_criteria import CRITERIA_REGISTRY
from src.data.dataset import AeroDataset
from src.model.amr_model import AdaptiveMeshAeroModel
from src.model.reconstruction import tokens_to_grid, tokens_to_grid_affine
from src.utils.mesh_visualization import plot_mesh
from src.utils.prediction_visualization import plot_flow_comparison, plot_3d_prediction


def create_model(args, checkpoint_file, dataset, input_channels=5, output_channels=3):
    """Instantiate the model from a config file and load weights from a checkpoint.

    Args:
        args: Arguments from a read YAML config (see configs/overfit.yaml).
        checkpoint_file: Path to a .pt checkpoint produced during training.
        input_channels: Number of input channels (including AoA/Mach added by the dataset).
        output_channels: Number of predicted channels.

    Returns:
        Tuple of (model, args) where args is the parsed YAML config dict.
    """

    refinement_mode = args.get("refinement_mode")
    criteria = None
    if refinement_mode == "deterministic":
        criteria = CRITERIA_REGISTRY[args.get("refinement_criteria")]

    # Configs express bounds as patch sizes; convert once to integer depths (mirrors
    # main.py) and inject back into args so predict_single reuses the same values.
    H, W = dataset.H, dataset.W
    min_depth, max_depth = patch_sizes_to_depth_bounds(H, W, args.get("min_patch_size"), args.get("max_patch_size"))
    args["min_depth"] = min_depth
    args["max_depth"] = max_depth

    model = AdaptiveMeshAeroModel(
        input_channels=input_channels,
        output_channels=output_channels,
        d_model=args.get("d_model"),
        n_layers=args.get("n_layers"),
        n_heads=args.get("n_heads"),
        d_ff=args.get("d_ff"),
        dropout=args.get("dropout"),
        min_depth=min_depth,
        max_depth=max_depth,
        refinement_mode=refinement_mode,
        refinement_criteria=criteria,
        affine_output=args.get("affine_output", False),
        continuous_output=args.get("continuous_output", False),
    )

    checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    # Drop shape-mismatched params (strict=False still errors on shape clashes)
    # so a constant-head checkpoint can warm-start an affine_output model.
    model_sd = model.state_dict()
    state = {k: v for k, v in state.items()
             if not (k in model_sd and v.shape != model_sd[k].shape)}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, args


@torch.no_grad()
def predict_single(model, args, sample):
    """Run one forward pass on a single sample and reconstruct the full-grid prediction.

    Args:
        model: A loaded AdaptiveMeshAeroModel in eval mode.
        args: Parsed YAML config dict (needed for the deterministic tokenizer settings).
        sample: Dataset sample dict with keys 'input' [H, W, C] and 'target' [H, W, output_channels].

    Returns:
        Dict with keys:
            input_grid:   [H, W, input_channels] input grid (numpy).
            ground_truth: [H, W, output_channels] target grid (numpy).
            prediction:   [H, W, output_channels] reconstructed flow field (numpy).
            token_preds:  [N, output_channels] per-token predictions (tensor), or
                          [N, output_channels, 3] = (value, gx, gy) when affine_output.
            mesh:         List[QuadNode] leaves of the adaptive mesh.
    """
    input_grid = sample["input"]                  # [H, W, C] numpy
    H, W, _ = input_grid.shape
    output_channels = model.output_channels

    if model.refinement_mode == "deterministic":
        tokenizer = QuadtreeTokenizer(
            min_depth=args.get("min_depth"),
            max_depth=args.get("max_depth"),
            refinement_criteria=CRITERIA_REGISTRY[args.get("refinement_criteria")],
        )
        token_array, leaves = tokenizer.tokenize(input_grid)
        packed_tokens = torch.from_numpy(token_array).float()
        out = model(packed_tokens, [len(leaves)])
    else:
        grids = torch.from_numpy(input_grid).float().unsqueeze(0)   # [1, H, W, C]
        out = model(grids)
        leaves = out["token_lists"][0]

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
    config_file = "configs/train_learned_transformer.yaml"
    checkpoint_file = "outputs/checkpoints/transformer_on_learned_mesh.pt"

    with open(config_file, "r") as f:
        args = yaml.safe_load(f)

    dataset = AeroDataset(
        input_path=args.get("input_file"),
        target_path=args.get("target_file"),
        index_path=args.get("index_file"),
    )

    model = create_model(args, checkpoint_file, dataset)

    sample_index = random.randint(0, dataset.__len__())
    # sample_index = dataset.__len__() - 1
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

