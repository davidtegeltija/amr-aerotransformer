import numpy as np
import torch

from src.amr.adaptive_mesh import build_adaptive_mesh
from src.amr.learned_adaptive_mesh import build_depth_guided_mesh
from src.amr.quadtree import nodes_to_token_array, token_feature_width
from src.amr.refinement_criteria import CRITERIA_REGISTRY
from src.models.reconstruction import tokens_to_grid, tokens_to_grid_affine
from src.models.refinement_net import RefinementNet
from src.utils.checkpoint import build_model_from_checkpoint


# ---------------------------------------------------------------------------
# Mesh source
# ---------------------------------------------------------------------------
def resolve_mesh_source(args):
    """Build the pair of mesh arguments ``predict_single_amr`` chooses between.

    Loading the scorer reads its checkpoint from disk, so a caller that predicts
    the same config more than once resolves the pair once here and passes it on
    rather than calling this per prediction.

    Args:
        args: Merged config. ``refinement_criteria`` selects the deterministic
            path, ``checkpoint_file`` the learned one; a config carries one.

    Returns:
        ``(refinement_criteria, scorer)``, with the unused one ``None``.
    """
    refinement_criteria = CRITERIA_REGISTRY[args["refinement_criteria"]] if args.get("refinement_criteria") else None
    scorer = build_model_from_checkpoint(RefinementNet, args["checkpoint_file"]).eval() if args.get("checkpoint_file") else None

    return refinement_criteria, scorer


# ---------------------------------------------------------------------------
# Single-sample prediction
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_single_amr(model, sample, *, max_depth, min_depth, refinement_criteria, scorer, offset,
                       mesh_cache=None):
    """Run one forward pass on a single sample and reconstruct the full-grid prediction.

    The mesh is built here (mirroring the collate functions) and the transformer
    only consumes packed tokens. Which mesh is built follows from which of the
    two mesh arguments is given — pass exactly one:
    * deterministic -> **refinement_criteria** -> build_adaptive_mesh (physics AMR criterion)
    * learned       -> **scorer** depth map    -> build_depth_guided_mesh (**offset** applies here only)

    Args:
        mesh_cache: Optional dict reused across calls to share deterministic
            meshes between rows of the same geometry; see the mesh block below.
            Applies to the deterministic path alone — a learned mesh comes from
            the scorer, which reads the whole grid including the condition
            channels, so two rows of one geometry can get different meshes.
            Omit it (the default) to build every mesh from scratch, which a
            caller predicting a single row wants anyway.
    """

    if (refinement_criteria is None) == (scorer is None):
        raise ValueError("Pass exactly one of **refinement_criteria** or **scorer**")

    input_grid = sample["input"]                  # [H, W, C] numpy
    H, W, C = input_grid.shape
    output_channels = model.output_channels

    if scorer is None:
        # Rows that share a geometry share a deterministic mesh: the criteria read
        # only the leading (x, y, z) channels, and the wing data's two condition
        # channels are constant over the grid. Naming the geometry lets a sweep
        # build one tree per wing instead of one per simulation -- 6.8 rows per
        # geometry on the full SuperWing set. Same contract as
        # DeterministicCollateFn._mesh_cache, and a dataset offering no mesh_key
        # (or a caller passing no cache) simply builds every mesh as before.
        mesh_key = sample.get("mesh_key") if mesh_cache is not None else None
        leaves = mesh_cache.get(mesh_key) if mesh_key is not None else None
        reused = leaves is not None
        if not reused:
            leaves = build_adaptive_mesh(input_grid, refinement_criteria, max_depth=max_depth, min_depth=min_depth)
            if mesh_key is not None:
                mesh_cache[mesh_key] = leaves
    else:
        reused = False
        # Learned mesh needs a frozen scorer to build the mesh at inference time.
        input_grid = np.asarray(input_grid, dtype=np.float32)
        grid = torch.from_numpy(input_grid).unsqueeze(0)  # [1, H, W, C]
        depth_map = scorer(grid).squeeze(1)[0].numpy()    # [H, W]
        leaves = build_depth_guided_mesh(input_grid, depth_map, max_depth=max_depth, min_depth=min_depth, offset=offset)

    # The model's token width is what says whether it was trained with affine_input,
    # so the checkpoint decides this rather than a config that could disagree with it.
    affine_input = model.input_channels == token_feature_width(C)
    token_array = nodes_to_token_array(leaves, H, W, C, affine_input)
    if reused:
        # The cached leaves carry the channel means of whichever row built the mesh.
        # Rows sharing a mesh differ only in channels the dataset holds constant over
        # the grid (the wing data's angle of attack and Mach), and a constant
        # channel's cell mean is that constant, so rewriting those columns -- which
        # lead the token, one per input channel -- is the whole correction. The
        # gradient columns need none: they cover the geometry channels alone.
        constant = (input_grid == input_grid[0, 0]).all(axis=(0, 1))
        token_array[:, np.flatnonzero(constant)] = input_grid[0, 0, constant]
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
