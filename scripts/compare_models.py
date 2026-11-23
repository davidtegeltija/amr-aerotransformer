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
from src.utils.plot import plot_coefficient_correlation, plot_channel_sections, plot_mesh_prediction_comparison


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

    # Each entry is name, model_config, checkpoint_file and optionally a dict of config overrides.
    # The cell-shape configs were not kept, so they share one config and name their scorer.
    MODELS = [
        # ("2:1 cells", "configs/learned_transformer/tokens=512/learned_n=512_affine=1-2.yaml", "outputs/checkpoints/2026-10-12_learned_n=512_affine=1-2_quad__token_embed-prediction_head-init.pt",
        #  {"checkpoint_file": "outputs/checkpoints/2026-10-11_scorer_n=512_quad.pt"}),
        # ("1:1 cells", "configs/learned_transformer/tokens=512/learned_n=512_affine=1-2.yaml", "outputs/checkpoints/2026-10-13_learned_n=512_affine=1-2_square_mesh__token_embed+prediction_head+init.pt",
        #  {"checkpoint_file": "outputs/checkpoints/2026-10-13_scorer_n=512_square_mesh.pt", "max_patch_size": 32}),
        # ("1:2 cells", "configs/learned_transformer/tokens=512/learned_n=512_affine=1-2.yaml", "outputs/checkpoints/2026-11-11_learned_n=512_affine=1-2_rectangular_mesh.pt",
        #  {"checkpoint_file": "outputs/checkpoints/2026-11-09_scorer_n=512_rectangular.pt"}),

        # ("n=128 order=1-0", "configs/learned_transformer/tokens=128/learned_n=128_affine=1-0.yaml", "outputs/checkpoints/2026-09-27_learned_n=128_affine=1-0.pt"),
        # ("n=128 order=1-1", "configs/learned_transformer/tokens=128/learned_n=128_affine=1-1.yaml", "outputs/checkpoints/2026-09-27_learned_n=128_affine=1-1.pt"),
        # ("n=128 order=1-2", "configs/learned_transformer/tokens=128/learned_n=128_affine=1-2.yaml", "outputs/checkpoints/2026-09-27_learned_n=128_affine=1-2.pt"),
        # ("Order=1-0", "configs/learned_transformer/tokens=512/learned_n=512_affine=1-0.yaml", "outputs/checkpoints/2026-09-28_learned_n=512_affine=1-0.pt"),
        # ("Order=1-1", "configs/learned_transformer/tokens=512/learned_n=512_affine=1-1.yaml", "outputs/checkpoints/2026-09-28_learned_n=512_affine=1-1.pt"),
        # ("Order=1-2", "configs/learned_transformer/tokens=512/learned_n=512_affine=1-2.yaml", "outputs/checkpoints/2026-09-28_learned_n=512_affine=1-2.pt"),
        # ("Learned n=512 affine=1-2 perturb", "configs/learned_transformer/tokens=512/learned_n=512_affine=1-2_mesh_jitter.yaml", "outputs/checkpoints/2026-10-19_learned_n=512_affine=1-2_jitter=0.55.pt"),
        # ("Order=1-2", "configs/learned_transformer/tokens=512/learned_n=512_affine=1-2_quad.yaml", "outputs/checkpoints/2026-10-12_learned_n=512_affine=1-2_quad__token_embed-prediction_head-init.pt"),
        # ("n=2048 order=1-0", "configs/learned_transformer/tokens=2048/learned_n=2048_affine=1-0.yaml", "outputs/checkpoints/2026-10-02_learned_n=2048_affine=1-0.pt"),
        # ("n=2048 order=1-1", "configs/learned_transformer/tokens=2048/learned_n=2048_affine=1-1.yaml", "outputs/checkpoints/2026-10-08_learned_n=2048_affine=1-1.pt"),
        # ("n=2048 order=1-2", "configs/learned_transformer/tokens=2048/learned_n=2048_affine=1-2.yaml", "outputs/checkpoints/2026-10-08_learned_n=2048_affine=1-2.pt"),
        
        ("ViT", "configs/vit/vit_8x8.yaml", "outputs/checkpoints/2026-10-11_vit_n=512_optimizer_trainable.pt"),
        ("Uniform AMT", "configs/deterministic_transformer/deterministic_UNIFORM_8x8.yaml", "outputs/checkpoints/2026-10-13_uniform_n=512_affine=1-2__token_embed+prediction_head+init.pt"),
        ("Learned AMT", "configs/learned_transformer/tokens=512/learned_n=512_affine=1-2_quad_square_mesh.yaml", "outputs/checkpoints/2026-10-13_learned_n=512_affine=1-2_square_mesh__token_embed+prediction_head+init.pt"),
        # ("n=512 AMT (learned)", "configs/learned_transformer/tokens=512/learned_n=512_affine=1-2_mesh_jitter.yaml", "outputs/checkpoints/2026-10-19_learned_n=512_affine=1-2_jitter=0.55.pt"),
      
        # ("a=0", "configs/learned_transformer/tokens=512/learned_n=512_affine=1-2_quad.yaml", "outputs/checkpoints/2026-10-12_learned_n=512_affine=1-2_quad__token_embed-prediction_head-init.pt"),
        # ("a=0.1", "configs/learned_transformer/tokens=512/learned_n=512_affine=1-2_mesh_perturb=0.1.yaml", "outputs/checkpoints/2026-11-09_learned_n=512_affine=1-2_perturb=0.1.pt"),
        # ("a=0.55", "configs/learned_transformer/tokens=512/learned_n=512_affine=1-2_mesh_jitter.yaml", "outputs/checkpoints/2026-10-19_learned_n=512_affine=1-2_jitter=0.55.pt"),
        # ("a=0.6", "configs/learned_transformer/tokens=512/learned_n=512_affine=1-2_mesh_jitter2.yaml", "outputs/checkpoints/2026-10-19_learned_n=512_affine=1-2_jitter=0.6.pt"),
        # ("a=0.9", "configs/learned_transformer/tokens=512/learned_n=512_affine=1-2_mesh_perturb=0.9.yaml", "outputs/checkpoints/2026-11-09_learned_n=512_affine=1-2_perturb=0.9.pt"),

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

    models, predictions, coefficients, coefficients_target = {}, {}, {}, None
    for name, model_config, checkpoint_file, *overrides in MODELS:
        model_args = load_config(model_config, data_config)
        model_args.update(*overrides)
        model, model_args = build_model(model_args, checkpoint_file, dataset)

        # Resolved once per model: the sample plot and the coefficients below
        # both predict with it, and a ViT config resolves to (None, None).
        mesh_source = resolve_mesh_source(model_args)
        models[name] = (model, model_args, mesh_source)

        result = predict_sample(model, model_args, sample, mesh_source)
        predictions[name] = result["prediction"]

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
        spanwise_position=[64],
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
        save_path=f"outputs/plots/cd_comparison2_first={len(coefficient_idx)}.png",
    )

    # The mesh comparison shows the first test sample whose C_p MAE drops from each model in MODELS to the next
    count = 0
    for showcase_index in coefficient_idx:
        showcase = dataset[showcase_index]
        showcase_results = {name: predict_sample(model, model_args, showcase, mesh_source)
                            for name, (model, model_args, mesh_source) in models.items()}
        mae = [np.abs(result["prediction"][..., 0] - showcase["target"][..., 0]).mean() for result in showcase_results.values()]
        if mae[0] > mae[1] and mae[1] > mae[2]:
            # break
            count += 1
            if count == 1:
                break
        # if all(a > b for a, b in zip(mae, mae[1:])):
        #     print(f"Showcase sample {showcase_index}: " + ", ".join(f"{name} {value:.2e}" for name, value in zip(models, mae)))
        #     break

    plot_mesh_prediction_comparison(
        showcase["target"][..., 0],  # C_P
        {name: result["prediction"][..., 0] for name, result in showcase_results.items()},
        {name: result.get("mesh") for name, result in showcase_results.items()},
        label="$C_p$",
        show=False,
        save_path=f"outputs/plots/mesh_prediction_comparison_sample={showcase_index}.png",
    )
