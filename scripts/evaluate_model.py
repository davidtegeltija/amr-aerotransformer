import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data.collate_fn import perturb_depth_map
from src.data.dataset_factory import TIMESTEP_DATASETS, build_dataset
from src.data.split import test_row_indices
from src.models.amr_model import AMRTransformer
from src.models.vit_model import ViT
from src.evaluation.evaluate import evaluate_aero_coefficients, evaluate_error_rate
from src.evaluation.evaluate_autoregressive import evaluate_next_timestep
from src.inference.predict import resolve_mesh_source, predict_single_amr, predict_single_vit
from src.utils.config import load_config, resolve_depth_bounds
from src.utils.checkpoint import build_model_from_checkpoint
from src.utils.plot import plot_mesh, plot_flow_comparison, plot_error_by_depth, plot_next_timestep


class PerturbedScorer:
    """Scorer whose depth map gets the train-time ``perturb_depth_map`` noise, to test a model on perturbed meshes.

    The noise is seeded by the input grid, so every evaluation pass and every model sees the same mesh of a sample.
    """

    def __init__(self, scorer, amplitude: float):
        self.scorer = scorer
        self.amplitude = amplitude
        self.patch_sides_ratio = scorer.patch_sides_ratio

    def __call__(self, grid):
        depth_map = self.scorer(grid)                                               # [1, 1, H, W]
        rng = np.random.default_rng(grid.numpy().view(np.uint32).ravel())
        noise = perturb_depth_map(np.zeros(depth_map.shape[-2:]), self.amplitude, rng)
        return depth_map + depth_map.new_tensor(noise)


def check_if_not_evaluated(result_path: Path) -> None:
    """Exit the script if an evaluation is already saved."""
    if result_path.exists():
        print(f"Evaluation for this configuration already exists: {result_path}")
        sys.exit(0)


def save_evaluation(buffer: io.StringIO, result_path: Path, model_config: str, checkpoint_file: str, data_config: str) -> None:
    """Write the captured evaluation output, including the configs it was run with."""
    result_path.parent.mkdir(parents=True, exist_ok=True)
    header = f"model_config: {model_config}\ncheckpoint_file: {checkpoint_file}\ndata_config: {data_config}\n"
    result_path.write_text(header + buffer.getvalue(), encoding="utf-8")
    print(f"Saved evaluation to {result_path}")


if __name__ == "__main__":
    data_config = "configs/data/wing.yaml"
    model_config = "configs/deterministic_transformer/deterministic_UNIFORM_16x16.yaml"
    checkpoint_file = "outputs/checkpoints/2026-11-16_uniform_n=128.pt"
    
    test_perturbation = 0.0   # Amplitude of the depth map noise on the learned test mesh, 0 for the clean mesh

    suffix = f"_test_perturb={test_perturbation}" if test_perturbation else ""
    result_path = Path("outputs/evaluation") / f"{Path(checkpoint_file).stem}{suffix}.txt"

    print(f"\nEvaluating checkpoint: {checkpoint_file}\n")

    args = load_config(model_config, data_config)
    dataset, dataset_type = build_dataset(args)

    # Predict on the test split, replayed from the config
    test_idx = test_row_indices(dataset, dataset_type, args.get("val_split"), args.get("seed", 42))
    sample_index = test_idx[-1]
    sample = dataset[sample_index]

    # Build a model from a checkpoint. The mesh source is resolved once here and
    # handed to every prediction below, so the scorer loaded once
    mesh_source = None

    if args.get("model_trained") == "vit":
        model = build_model_from_checkpoint(ViT, checkpoint_file).eval()
        result = predict_single_vit(model, sample)
    else:
        model = build_model_from_checkpoint(AMRTransformer, checkpoint_file).eval()

        # Add min_depth/max_depth to args. Only the AMR path builds a quadtree,
        # and only its configs carry the patch sizes the bounds come from
        args = resolve_depth_bounds(args, dataset)

        mesh_source = resolve_mesh_source(args)
        refinement_criteria, scorer = mesh_source
        if test_perturbation:
            scorer = PerturbedScorer(scorer, test_perturbation)
            mesh_source = (refinement_criteria, scorer)

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
    model_name = result_path.stem

    # --- Mesh --- (AMR only; the ViT predicts the dense grid, so there is none)
    if "mesh" in result:
        plot_mesh(result["input_grid"], result["mesh"], show=False, save_path=f"outputs/plots/{model_name}_sample={sample_index}.png")

        # Error by patch depth
        meshes, predictions, targets = [], [], []
        for index in test_idx[:50]:
            sample_i = dataset[index]
            result_i = predict_single_amr(model, sample_i, max_depth=args["max_depth"], min_depth=args["min_depth"],
                                          refinement_criteria=refinement_criteria, scorer=scorer,
                                          offset=args.get("offset", 0.0))
            meshes.append(result_i["mesh"])
            predictions.append(result_i["prediction"])
            targets.append(result_i["ground_truth"])

        plot_error_by_depth(meshes, predictions, targets, channel=0, show=False,
                            save_path=f"outputs/plots/{model_name}_error_by_depth.png")
        
    # ---Flow ---
    n_patches = len(result["mesh"]) if "mesh" in result else model.nh * model.nw
    plot_flow_comparison(result["ground_truth"], result["prediction"],
                         title=f"Ground Truth vs Prediction on a Learned Mesh ({n_patches} patches)",
                         save_path=f"outputs/plots/{model_name}_prediction_sample={sample_index}.png")

    # --- Next timestep --- (the dataset's showcase case at t, t+n, t+m; see TIMESTEP_EXAMPLES)
    if dataset_type in TIMESTEP_DATASETS:
        plot_next_timestep(model, args, dataset, mesh_source, show_mesh=True,
                           show=False, save_path=f"outputs/plots/{model_name}_next_timestep.png")


    # ------------------------
    # Model Accuracy
    # ------------------------
    check_if_not_evaluated(result_path)
    
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        if dataset_type in TIMESTEP_DATASETS:
            # --- Next timestep, rebuilt from the predicted residual ---
            metrics_next_timestep = evaluate_next_timestep(model, args, dataset, test_idx, mesh_source)
        else:
            # --- Prediction ---
            metrics_l2 = evaluate_error_rate(model, args, dataset, test_idx, "l2", mesh_source)
            metrics_cae = evaluate_error_rate(model, args, dataset, test_idx, "mae", mesh_source)

            # --- Aero Coefficients ---
            index_array = np.load(args["index_file"])
            geometry_array = np.load("/mnt/data/tegeltija/origingeom.npy", mmap_mode="r")

            metrics_coef = evaluate_aero_coefficients(model, args, dataset, test_idx, index_array, geometry_array, mesh_source)

    print(buffer.getvalue(), end="")

    save_evaluation(buffer, result_path, model_config, checkpoint_file, data_config)
    