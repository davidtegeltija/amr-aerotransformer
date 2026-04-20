import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch

from src.data.dataset import AeroDataset
from src.model.amr_model import AdaptiveMeshAeroModel
from src.utils.flow_visualization import plot_flow_comparison
from src.utils.visualization_3d import visualize_3d_prediction


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
    checkpoint = torch.load("outputs/checkpoints/best_model_16-04-2026.pt")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = AeroDataset(input_path="data/crmmgeom.npy", target_path="data/crmmdata.npy")
    first_sample = dataset[0]
    input = first_sample["input"]
    input = torch.from_numpy(input)
    prediction = model.forward_single(input)
    prediction  = prediction["prediction"].cpu().numpy()
    
    ground_truth = first_sample["target"]


    # input_data = np.load("data/crmmgeom.npy")
    # sample = input_data[0]   # shape: (3, 128, 256) — channel-first (C, H, W)
    # sample = sample.transpose(2, 1, 0).astype(np.float32)  # (H, W, C) = (128, 256, 3)
    # input = torch.from_numpy(sample)
    # prediction = model.forward_single(input)
    # prediction = prediction["prediction"].cpu().numpy()  # [H, W, output_dim]

    # target_data = np.load("data/crmmdata.npy")
    # ground_truth = target_data[0]
    # ground_truth = ground_truth.transpose(2, 1, 0).astype(np.float32)

    # plot_flow_comparison(ground_truth, prediction, show=True)
    visualize_3d_prediction(first_sample["input"], prediction)
