import numpy as np
import torch

from src.amr.adaptive_mesh import build_adaptive_mesh
from src.amr.learned_adaptive_mesh import build_depth_guided_mesh
from src.amr.quadtree import nodes_to_token_array, token_feature_width
from src.models.reconstruction import tokens_to_grid, tokens_to_grid_affine


@torch.no_grad()
def predict_single_amr(model, sample, *, max_depth, min_depth, refinement_criteria, scorer, offset):
    """Run one forward pass on a single sample and reconstruct the full-grid prediction.

    The mesh is built here (mirroring the collate functions) and the transformer
    only consumes packed tokens. Which mesh is built follows from which of the
    two mesh arguments is given — pass exactly one:
    * deterministic -> **refinement_criteria** -> build_adaptive_mesh (physics AMR criterion)
    * learned       -> **scorer** depth map    -> build_depth_guided_mesh (**offset** applies here only)
    """

    if (refinement_criteria is None) == (scorer is None):
        raise ValueError("Pass exactly one of **refinement_criteria** or **scorer**")

    input_grid = sample["input"]                  # [H, W, C] numpy
    H, W, C = input_grid.shape
    output_channels = model.output_channels

    if scorer is None:
        leaves = build_adaptive_mesh(input_grid, refinement_criteria, max_depth=max_depth, min_depth=min_depth)
    else:
        # Learned mesh needs a frozen scorer to build the mesh at inference time.
        input_grid = np.asarray(input_grid, dtype=np.float32)
        grid = torch.from_numpy(input_grid).unsqueeze(0)  # [1, H, W, C]
        depth_map = scorer(grid).squeeze(1)[0].numpy()    # [H, W]
        leaves = build_depth_guided_mesh(input_grid, depth_map, max_depth=max_depth, min_depth=min_depth, offset=offset)

    # The model's token width is what says whether it was trained with affine_input,
    # so the checkpoint decides this rather than a config that could disagree with it.
    affine_input = model.input_channels == token_feature_width(C)
    token_array = nodes_to_token_array(leaves, H, W, C, affine_input)
    packed_tokens = torch.from_numpy(token_array).float()
    out = model(packed_tokens, [len(leaves)])
    token_preds = out["token_preds"]

    if model.affine_output:
        # token_preds is [N, C, K] where K defines the cell's polynomial.
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


@torch.no_grad()
def predict_single_vit(model, sample):
    """Run one forward pass on a single sample and return the full-grid prediction."""
    input_grid = sample["input"]                                                        # [H, W, C] numpy
    grid = torch.from_numpy(np.asarray(input_grid, dtype=np.float32))
    grid = grid.permute(2, 0, 1).unsqueeze(0)                                           # [1, C, H, W]

    prediction = model(grid)                                                            # [1, output_channels, H, W]
    prediction = prediction.squeeze(0).permute(1, 2, 0)                                 # [H, W, output_channels]

    return {
        "input_grid": input_grid,
        "ground_truth": sample["target"],
        "prediction": prediction.cpu().numpy(),
    }
