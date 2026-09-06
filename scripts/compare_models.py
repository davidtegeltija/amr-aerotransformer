import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from src.data.dataset_factory import build_dataset
from src.data.split import test_row_indices
from src.models.amr_model import AMRTransformer
from src.models.vit_model import ViT
from src.evaluation.evaluate import evaluate_aero_coefficients
from src.inference.predict import resolve_mesh_source, predict_single_amr, predict_single_vit
from src.utils.config import load_config, resolve_depth_bounds
from src.utils.checkpoint import build_model_from_checkpoint
from src.utils.plot import plot_coefficient_correlation, plot_channel_sections


def build_model(args, checkpoint_file, dataset):
    """Restore one model from its checkpoint."""
    if args.get("model_trained") == "vit":
        return build_model_from_checkpoint(ViT, checkpoint_file).eval(), args

    model = build_model_from_checkpoint(AMRTransformer, checkpoint_file).eval()

    # Add min_depth/max_depth to args. Only the AMR path builds a quadtree,
    # and only its configs carry the patch sizes the bounds come from.
    return model, resolve_depth_bounds(args, dataset)


def predict_sample(model, args, sample, mesh_source):
    """Predict a single sample on whichever inference path the model selects."""
    if isinstance(model, ViT):
        return predict_single_vit(model, sample)

    refinement_criteria, scorer = mesh_source

    return predict_single_amr(
        model,
        sample,
        max_depth=args["max_depth"],
        min_depth=args["min_depth"],
        refinement_criteria=refinement_criteria,
        scorer=scorer,
        offset=args.get("offset", 0.0)
    )


if __name__ == "__main__":
    data_config = "configs/data/wing.yaml"

    # The models to overlay:. Each entry is name, model_config, checkpoint_file
    MODELS = [
        # ("Learned Transformer n=250", "configs/learned_transformer_n=250.yaml", "outputs/checkpoints/2026-08-08_learned_transformer_n=250_min=2.pt"),
        ("Learned Transformer n=800", "configs/learned_transformer_n=800.yaml", "outputs/checkpoints/2026-08-02_learned_transformer_n=800_min=2.pt"),
        ("Learned Transformer Discrete n=800", "configs/learned_transformer_n=800_discrete.yaml", "outputs/checkpoints/2026-08-14_learned_transformer_n=800_min=2_discrete.pt"),
        # ("Learned Transformer n=1000", "configs/learned_transformer_n=1000.yaml", "outputs/checkpoints/2026-08-08_learned_transformer_n=1000_min=2.pt"),
        # ("Learned Transformer n=2000", "configs/learned_transformer_n=2000.yaml", "outputs/checkpoints/2026-08-08_learned_transformer_n=2000_min=2.pt"),
        ("ViT", "configs/vit.yaml", "outputs/checkpoints/2026-08-02_vit_min=4.pt"),
    ]

    # All models are compared on the same sample, so the dataset and the test
    # split are built once from the first config (they only differ in the model
    # section, and the data config is shared).
    args = load_config(MODELS[0][1], data_config)
    dataset, dataset_type = build_dataset(args)

    # Predict on the test split, replayed from the config
    test_idx = test_row_indices(dataset, dataset_type, args.get("val_split"), args.get("seed", 42))
    sample_index = test_idx[-1]
    sample = dataset[sample_index]

    # The surface integrals run on the node grid of the original geometry file,
    # looked up per case through the index file.
    n_coefficient_samples = 100
    coefficient_idx = test_idx[:n_coefficient_samples]
    index_array = np.load(args["index_file"])
    geometry_array = np.load("/mnt/data/tegeltija/origingeom.npy", mmap_mode="r")

    predictions, coefficients, coefficients_target = {}, {}, None
    for name, model_config, checkpoint_file in MODELS:
        model_args = load_config(model_config, data_config)
        model, model_args = build_model(model_args, checkpoint_file, dataset)

        # Resolved once per model: the sample plot and the coefficients below
        # both predict with it, and a ViT config resolves to (None, None).
        mesh_source = resolve_mesh_source(model_args)

        predictions[name] = predict_sample(model, model_args, sample, mesh_source)["prediction"]

        # Both sides of the plot go through the same integral, so the gap it
        # shows is the model's rather than the integrator's. The target is the
        # same for every model, hence the single array kept from the last run.
        metrics = evaluate_aero_coefficients(model, model_args, dataset, coefficient_idx, index_array,
                                             geometry_array, mesh_source)
        coefficients[name] = metrics["coefficients_pred"]
        coefficients_target = metrics["coefficients_true"]

    # ------------------------
    # Plotting
    # ------------------------
    plot_channel_sections(
        sample["input"],
        predictions,
        sample["target"],
        channel=0,  # C_P
        title=f"Cp(x) across models  -  test sample {sample_index}",
        show=False,
        save_path=f"outputs/plots/cp_comparison_sample={sample_index}.png",
    )

    plot_coefficient_correlation(
        coefficients_target,
        coefficients,
        coefficient=1,  # CD
        title="Drag coefficient across models",
        show=False,
        save_path=f"outputs/plots/cd_comparison_first={len(coefficient_idx)}.png",
    )
