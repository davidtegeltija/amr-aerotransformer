import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.amr.refinement_criteria import CRITERIA_REGISTRY
from src.data.dataset_factory import build_dataset
from src.data.split import test_row_indices
from src.models.amr_model import AMRTransformer
from src.models.refinement_net import RefinementNet
from src.models.vit_model import ViT
from src.evaluation.evaluate import evaluate_aero_coefficients, evaluate_error_rate
from src.inference.predict import predict_single_amr, predict_single_vit
from src.utils.config import load_config, resolve_depth_bounds
from src.utils.checkpoint import build_model_from_checkpoint
from src.utils.plot import plot_mesh, plot_flow_comparison


if __name__ == "__main__":
    model_config = "configs/learned_transformer.yaml"
    data_config = "configs/data/wing.yaml"
    checkpoint_file = "outputs/checkpoints/transformer_on_learned_mesh.pt"

    print(f"\nEvaluating checkpoint: {checkpoint_file}\n")

    args = load_config(model_config, data_config)
    dataset, dataset_type = build_dataset(args)

    # Predict on the test split, replayed from the config
    test_idx = test_row_indices(dataset, dataset_type, args.get("val_split"), args.get("seed", 42))
    sample_index = test_idx[-1]
    sample = dataset[sample_index]

    # Build a model from a checkpoint
    if args.get("model_trained") == "vit":
        model = build_model_from_checkpoint(ViT, checkpoint_file).eval()
        result = predict_single_vit(model, sample)
    else:
        model = build_model_from_checkpoint(AMRTransformer, checkpoint_file).eval()

        # Add min_depth/max_depth to args. Only the AMR path builds a quadtree,
        # and only its configs carry the patch sizes the bounds come from.
        args = resolve_depth_bounds(args, dataset)

        refinement_criteria = CRITERIA_REGISTRY[args["refinement_criteria"]] if args.get("refinement_criteria") else None
        scorer = build_model_from_checkpoint(RefinementNet, args["checkpoint_file"]).eval() if args.get("checkpoint_file") else None

        result = predict_single_amr(
            model,
            sample,
            max_depth=args["max_depth"],
            min_depth=args["min_depth"],
            refinement_criteria=refinement_criteria,
            scorer=scorer,
            offset=args.get("offset", 0.0)
        )

    # ------------------------
    # Plotting
    # ------------------------
    model_name = Path(checkpoint_file).stem

    # --- Mesh --- (AMR only; the ViT predicts the dense grid, so there is none)
    if "mesh" in result:
        plot_mesh(result["input_grid"], result["mesh"], show=False, save_path=f"outputs/plots/{model_name}_sample={sample_index}.png")

    # ---Flow ---
    plot_flow_comparison(result["ground_truth"], result["prediction"], save_path=f"outputs/plots/{model_name}_prediction_sample={sample_index}.png")
    # plot_3d_prediction(sample["input"], prediction)


    # ------------------------
    # Model Accuracy
    # ------------------------
    # --- Prediction --- 
    metrics_l2 = evaluate_error_rate(model, args, dataset, test_idx, "l2")
    metrics_cae = evaluate_error_rate(model, args, dataset, test_idx, "mae")

    # --- Aero Coefficients ---
    index_array = np.load(args["index_file"])
    geometry_array = np.load("/mnt/data/tegeltija/origingeom.npy", mmap_mode="r")

    metrics_coef = evaluate_aero_coefficients(model, args, dataset, test_idx, index_array, geometry_array)
    