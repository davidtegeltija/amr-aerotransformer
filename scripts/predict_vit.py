import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.model.vit import ViT
from src.utils.config_utils import load_config
from src.utils.data_utils import build_dataset, test_row_indices
from src.utils.model_utils import build_model_from_checkpoint
from src.utils.prediction_visualization import plot_flow_comparison


def create_model(checkpoint_file):
    """Rebuild the trained ViT baseline from its checkpoint.

    Unlike the AMR model, the ViT consumes dense [B, C, H, W] grids directly, so
    there is no quadtree, tokenizer or scorer involved — the checkpoint's own
    hyperparameter record is all that is needed.
    """
    model = build_model_from_checkpoint(ViT, checkpoint_file)
    model.eval()
    return model


@torch.no_grad()
def predict_single(model, sample):
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


if __name__ == "__main__":
    model_config = "configs/vit.yaml"
    data_config = "configs/data/wing.yaml"
    checkpoint_file = "outputs/checkpoints/vit.pt"

    args = load_config(model_config, data_config)
    dataset, dataset_type = build_dataset(args)

    model = create_model(checkpoint_file)

    # Predict on the held-out test split, replayed from the config, so the plots
    # show a geometry (or cavity case) the model never trained on. The last test
    # row is the final frame of its case, where the flow has fully developed.
    # Early frames are near rest and nearly featureless.
    test_idx = test_row_indices(dataset, dataset_type, args.get("val_split"), args.get("seed", 42))
    sample_index = test_idx[-1]
    sample = dataset[sample_index]

    result = predict_single(model, sample)

    # --- Plotting Flow ---
    plot_flow_comparison(result["ground_truth"], result["prediction"],
                         save_path=f"outputs/plots/vit_prediction_test_sample={sample_index}.png")
