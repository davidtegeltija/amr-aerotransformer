import numpy as np
import torch

from src.model.amr_model import AdaptiveMeshAeroModel
from src.utils.flow_visualization import plot_flow_comparison


if __name__ == "__main__":
    model = AdaptiveMeshAeroModel(
        input_channels=3,
        output_dim=3,
        d_model=256,
        n_layers=6,
        n_heads=4,
        d_ff=1024,
        min_depth=2,
        max_depth=6,
    )
    model.load_state_dict(torch.load("outputs/checkpoints/best_model_13-04-2026.pt"))
    model.eval()

    input_data = np.load("data/crmmgeom.npy")
    sample = input_data[0]   # shape: (3, 128, 256) — channel-first (C, H, W)
    sample = sample.transpose(2, 1, 0).astype(np.float32)  # (H, W, C) = (128, 256, 3)
    input = torch.from_numpy(sample)
    prediction = model.forward_single(input)
    prediction = prediction["prediction"].cpu().numpy()  # [H, W, output_dim]

    target_data = np.load("data/crmmdata.npy")
    ground_truth = target_data[0]
    ground_truth = ground_truth.transpose(2, 1, 0).astype(np.float32)
    plot_flow_comparison(ground_truth, prediction, show=True)