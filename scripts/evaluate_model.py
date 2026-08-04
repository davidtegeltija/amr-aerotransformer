import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.config_utils import load_config, resolve_depth_bounds
from src.utils.data_utils import build_dataset, test_row_indices
from src.utils.mesh_visualization import plot_mesh
from src.utils.model_utils import build_model_from_checkpoint
from src.utils.prediction_visualization import plot_flow_comparison
from src.amr.refinement_criteria import CRITERIA_REGISTRY
from src.model.amr_model import AMRTransformer
from src.model.refinement_net import RefinementNet
from src.model.vit_model import ViT
from src.evaluate import evaluate_aero_coefficients, evaluate_error_rate
from src.inference import predict_single_amr, predict_single_vit


if __name__ == "__main__":
    model_config = "configs/learned_transformer.yaml"
    data_config = "configs/data/wing.yaml"
    checkpoint_file = "outputs/checkpoints/transformer_on_learned_mesh.pt"

    args = load_config(model_config, data_config)
    dataset, dataset_type = build_dataset(args)

    # Add min_depth/max_depth to args
    args = resolve_depth_bounds(args, dataset)

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

    # --- Mesh ---
    plot_mesh(result["input_grid"], result["mesh"], show=False, save_path=f"outputs/plots/amr_mesh_sample={sample_index}.png")

    # ---Flow ---
    plot_flow_comparison(result["ground_truth"], result["prediction"], save_path=f"outputs/plots/prediction_test_sample={sample_index}.png")
    # plot_3d_prediction(sample["input"], prediction)


    # ------------------------
    # Model Accuracy
    # ------------------------
    
    # --- Prediction --- 
    metrics_l2 = evaluate_error_rate(model, args, dataset, test_idx, "l2")
    metrics_cae = evaluate_error_rate(model, args, dataset, test_idx, "mae")

    # --- Aero Coefficients ---
    # The index file the dataset was built from, so its rows line up with the
    # dataset rows; the geometry file stays memory-mapped (~3 GB) and is read
    # one wing at a time through column 0 of the index.
    index_array = np.load(args["index_file"])
    geometry_array = np.load("/mnt/data/tegeltija/origingeom.npy", mmap_mode="r")

    metrics_coef = evaluate_aero_coefficients(model, args, dataset, test_idx, index_array, geometry_array)
    