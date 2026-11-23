import glob
import os
import re
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from mesh_floor import cell_shape_floors, learned_token_counts, mesh_l2_floors, sample_rows
from src.data.split import geometry_disjoint_split
from src.models.amr_model import AMRTransformer
from src.models.refinement_net import RefinementNet
from src.models.vit_model import ViT
from src.utils.checkpoint import build_model_from_checkpoint
from src.utils.plot import (plot_cell_shape_floors, plot_error_vs_cost, plot_model_floor, plot_perturbation_error,
                            plot_perturbation_robustness, plot_polynomial_floor, plot_polynomial_order_error)

# Model, nominal token count, evaluation file, checkpoint and the scorer of its learned mesh (None for a fixed mesh)
PERFORMANCE_RUNS = [
    ("ViT", 128, "2026-11-16_vit_n=128.txt", "2026-11-16_vit_n=128.pt", None),
    ("ViT", 512, "2026-10-11_vit_n=512_optimizer_trainable.txt", "2026-10-11_vit_n=512_optimizer_trainable.pt", None),
    ("ViT", 2048, "2026-10-13_vit_n=2048_optimizer_trainable.txt", "2026-10-13_vit_n=2048_optimizer_trainable.pt", None),
    ("AMT (uniform)", 128, "2026-11-16_uniform_n=128.txt", "2026-11-16_uniform_n=128.pt", None),
    ("AMT (uniform)", 512, "2026-10-13_uniform_n=512_affine=1-2___token_embed+prediction_head+init.txt",
     "2026-10-13_uniform_n=512_affine=1-2__token_embed+prediction_head+init.pt", None),
    ("AMT (uniform)", 2048, "2026-10-13_uniform_n=2048_affine=1-2___token_embed+prediction_head+init.txt",
     "2026-10-13_uniform_n=2048_affine=1-2___token_embed+prediction_head+init.pt", None),
    ("AMT (learned)", 128, "2026-11-15_learned_n=128_affine=1-2_square_mesh.txt", "2026-11-15_learned_n=128_affine=1-2_square_mesh.pt",
     "2026-11-15_scorer_n=128_square.pt"),
    ("AMT (learned)", 256, "2026-11-15_learned_n=256_affine=1-2_square_mesh.txt", "2026-11-15_learned_n=256_affine=1-2_square_mesh.pt",
     "2026-11-15_scorer_n=256_square.pt"),
    ("AMT (learned)", 512, "2026-10-13_learned_n=512_affine=1-2_square_mesh___token_embed+prediction_head+init.txt",
     "2026-10-13_learned_n=512_affine=1-2_square_mesh__token_embed+prediction_head+init.pt", "2026-10-13_scorer_n=512_square_mesh.pt"),
    ("AMT (learned)", 1024, "2026-10-14_learned_n=1024_affine=1-2_square_mesh___token_embed+prediction_head+init.txt",
     "2026-10-14_learned_n=1024_affine=1-2_square_mesh___token_embed+prediction_head+init.pt", "2026-10-13_scorer_n=1024_square_mesh.pt"),
    ("AMT (learned)", 2048, "2026-10-14_learned_n=2048_affine=1-2_square_mesh___token_embed+prediction_head+init.txt",
     "2026-10-14_learned_n=2048_affine=1-2_square_mesh___token_embed+prediction_head+init.pt", "2026-10-13_scorer_n=2048_square_mesh.pt"),
]
WING_GRID = (256, 128)

# Learned mesh of 512 tokens, trained with mesh perturbation of amplitude a
PERTURBATION_EVALUATIONS = {
    0.1: "2026-11-09_learned_n=512_affine=1-2_perturb=0.1.txt",
    0.5: "2026-10-19_learned_n=512_affine=1-2_jitter=0.5.txt",
    0.55: "2026-10-19_learned_n=512_affine=1-2_jitter=0.55.txt",
    0.6: "2026-10-19_learned_n=512_affine=1-2_jitter=0.6.txt",
    0.9: "2026-11-09_learned_n=512_affine=1-2_perturb=0.9.txt",
}
# Best unperturbed learned mesh of 512 tokens with the same 2:1 cells, the a = 0 row of the thesis table
UNPERTURBED_EVALUATION = "2026-10-12_learned_n=512_affine=1-2_quad___token_embed+prediction_head+init.txt"

# Training amplitude a -> {test-time amplitude: evaluation file}, from evaluate_model.py --test-perturbation
TEST_AMPLITUDES = (0.25, 0.5, 0.75, 1.0)
ROBUSTNESS_CHECKPOINTS = {
    0.0: "2026-10-12_learned_n=512_affine=1-2_quad__token_embed-prediction_head-init",
    0.55: "2026-10-19_learned_n=512_affine=1-2_jitter=0.55",
}
ROBUSTNESS_EVALUATIONS = {
    a: {0.0: UNPERTURBED_EVALUATION if a == 0 else PERTURBATION_EVALUATIONS[a],
        **{test: f"{checkpoint}_test_perturb={test}.txt" for test in TEST_AMPLITUDES}}
    for a, checkpoint in ROBUSTNESS_CHECKPOINTS.items()
}

# Learned mesh with every (input order, output order) of the polynomial tokens, per token count
POLYNOMIAL_EVALUATIONS = {
    (0, 0): {
        128: "2026-09-27_learned_n=128_affine=0-0.txt",
        256: "2026-09-27_learned_n=256_affine=0-0.txt",
        512: "2026-09-28_learned_n=512_affine=0-0.txt",
        1024: "2026-09-29_learned_n=1024_affine=0-0.txt",
        2048: "2026-10-02_learned_n=2048_affine=0-0.txt",
    },
    (0, 1): {
        128: "2026-09-27_learned_n=128_affine=0-1.txt",
        256: "2026-09-27_learned_n=256_affine=0-1.txt",
        512: "2026-09-28_learned_n=512_affine=0-1.txt",
        1024: "2026-09-29_learned_n=1024_affine=0-1.txt",
        2048: "2026-10-02_learned_n=2048_affine=0-1.txt",
    },
    (0, 2): {
        128: "2026-09-27_learned_n=128_affine=0-2.txt",
        256: "2026-09-27_learned_n=256_affine=0-2.txt",
        512: "2026-09-28_learned_n=512_affine=0-2.txt",
        1024: "2026-09-29_learned_n=1024_affine=0-2.txt",
        2048: "2026-10-02_learned_n=2048_affine=0-2.txt",
    },
    (1, 0): {
        128: "2026-09-27_learned_n=128_affine=1-0.txt",
        256: "2026-09-28_learned_n=256_affine=1-0.txt",
        512: "2026-09-28_learned_n=512_affine=1-0.txt",
        1024: "2026-10-01_learned_n=1024_affine=1-0.txt",
        2048: "2026-10-02_learned_n=2048_affine=1-0.txt",
    },
    (1, 1): {
        128: "2026-09-27_learned_n=128_affine=1-1.txt",
        256: "2026-09-28_learned_n=256_affine=1-1.txt",
        512: "2026-09-28_learned_n=512_affine=1-1.txt",
        1024: "2026-10-01_learned_n=1024_affine=1-1.txt",
        2048: "2026-10-08_learned_n=2048_affine=1-1.txt",
    },
    (1, 2): {
        128: "2026-09-27_learned_n=128_affine=1-2.txt",
        256: "2026-09-28_learned_n=256_affine=1-2.txt",
        512: "2026-09-28_learned_n=512_affine=1-2.txt",
        1024: "2026-10-01_learned_n=1024_affine=1-2.txt",
        2048: "2026-10-08_learned_n=2048_affine=1-2.txt",
    },
}


def read_evaluation(evaluation_path: str):
    """Read the metric tables of an evaluation file row by row."""

    with open(os.path.join(PROJECT_ROOT, "outputs", "evaluation", evaluation_path), encoding="utf-8") as f:
        lines = f.read().splitlines()

    results, key = {}, None
    for line in lines:
        if line.startswith("Test split:"):
            key = line.rsplit("|", 1)[1].strip()
            results[key] = []
        elif key is not None and line.strip() and not line.strip().startswith(("channel", "coef.", "prediction vs solver")):
            channel, rest = line.split(maxsplit=1)
            values = rest.split("(")[0].split()
            results[key].append([float(i[:-1]) / 100 if i.endswith("%") else float(i) for i in values])


    return results


def overall_l2_error(evaluation_path: str) -> float:
    """Overall L2 error of an evaluation file, the second-to-last value of its overall row."""
    return read_evaluation(evaluation_path)["l2"][-1][-2]


def overall_coefficient_error(evaluation_path: str) -> float:
    """Overall aerodynamic coefficient error of an evaluation file, the mean over CL, CD and CMz."""
    return read_evaluation(evaluation_path)["aero coefficients"][-1][-2]


def plot_perturbation():
    """Overall L2 error against the mesh perturbation amplitude."""
    amplitudes = list(PERTURBATION_EVALUATIONS)
    errors = [overall_l2_error(path) for path in PERTURBATION_EVALUATIONS.values()]
    plot_perturbation_error(amplitudes, errors, overall_l2_error(UNPERTURBED_EVALUATION),
                            reference_label="No perturbation",
                            save_path="outputs/plots/mesh_perturbation_l2_error.png")


def plot_perturbation_coefficients():
    """Overall aerodynamic coefficient error against the mesh perturbation amplitude."""
    amplitudes = list(PERTURBATION_EVALUATIONS)
    errors = [overall_coefficient_error(path) for path in PERTURBATION_EVALUATIONS.values()]
    plot_perturbation_error(amplitudes, errors, overall_coefficient_error(UNPERTURBED_EVALUATION),
                            reference_label="No perturbation", ylabel="Relative error of coefficients (%)",
                            save_path="outputs/plots/mesh_perturbation_coefficient_error.png")


def plot_perturbation_robustness_curves():
    """Overall L2 and coefficient error against the test-time perturbation amplitude, per training amplitude."""
    for error_fn, ylabel, name in ((overall_l2_error, "Relative $L_2$ error of surface fields (%)", "l2"),
                                   (overall_coefficient_error, "Relative error of coefficients (%)", "coefficient")):
        errors = {a: {test: error_fn(path) for test, path in by_test.items()} for a, by_test in ROBUSTNESS_EVALUATIONS.items()}
        plot_perturbation_robustness(errors, ylabel=ylabel, save_path=f"outputs/plots/mesh_perturbation_robustness_{name}.png")


def plot_polynomial_orders():
    """Overall L2 error against the token count for every input and output order."""
    errors = {orders: {n: overall_l2_error(path) for n, path in by_tokens.items()}
              for orders, by_tokens in POLYNOMIAL_EVALUATIONS.items()}
    plot_polynomial_order_error(errors, save_path="outputs/plots/polynomial_order_error.png")


def plot_polynomial_coefficients():
    """Overall aerodynamic coefficient error against the token count for every input and output order."""
    errors = {orders: {n: overall_coefficient_error(path) for n, path in by_tokens.items()}
              for orders, by_tokens in POLYNOMIAL_EVALUATIONS.items()}
    plot_polynomial_order_error(errors, ylabel="Relative error of coefficients (%)",
                                save_path="outputs/plots/polynomial_order_coefficient_error.png")


def plot_polynomial_floor_comparison():
    """Overall L2 error of every input and output order at 512 tokens against the reconstruction floor."""
    # All six runs share one scorer, so any of their configs gives the mesh of every order
    floors = {order: floor.mean() for order, floor in
              mesh_l2_floors("configs/learned_transformer/tokens=512/learned_n=512_affine=1-2.yaml").items()}
    errors = {orders: overall_l2_error(by_tokens[512]) for orders, by_tokens in POLYNOMIAL_EVALUATIONS.items()}
    print("Floors: " + ", ".join(f"order {order} {100 * floor:.2f}%" for order, floor in floors.items()))
    plot_polynomial_floor(floors, errors, save_path="outputs/plots/polynomial_floor_n=512.png")


def table_errors(evaluation_path: str) -> np.ndarray:
    """Relative MAE of Cp, Cf_tau and Cf_z, then relative error of CL, CD and CMz of an evaluation file, as fractions."""
    results = read_evaluation(evaluation_path)
    return np.array([row[-2] for row in results["mae"][:3] + results["aero coefficients"][:3]])


def peak_gpu_gib(checkpoint: str) -> float:
    """Peak training GPU memory of a run (batch size 32) in GiB, the maximum over the epochs in its log."""
    date, name = checkpoint[:-3].split("_", 1)
    log = glob.glob(os.path.join(PROJECT_ROOT, "outputs", "logs", f"{date}_*_{name}", "*.log"))[0]
    with open(log, encoding="utf-8") as f:
        return max(float(gib) for gib in re.findall(r"gpu_peak=([\d.]+)GiB", f.read()))


def wing_test_rows() -> list:
    """SuperWing rows of the test split, which depends only on the seed."""
    geometry_ids = np.load("/mnt/data/tegeltija/index.npy")[:, 0].astype(int)
    return geometry_disjoint_split(geometry_ids, val_split=0.2, seed=42)[2]


def token_counts(n_tokens: int, scorer_file) -> np.ndarray:
    """Token count of a run on every test row: the nominal one for a fixed mesh, measured once and cached for a learned one."""
    if scorer_file is None:
        return np.array([n_tokens])
    cache = os.path.join(PROJECT_ROOT, "outputs", "evaluation", f"token_counts_{scorer_file[:-3]}.npy")
    if not os.path.exists(cache):
        np.save(cache, learned_token_counts(f"outputs/checkpoints/{scorer_file}", wing_test_rows()))
    return np.load(cache)


@torch.no_grad()
def inference_gflops(checkpoint: str, n_tokens: int, scorer_file) -> float:
    """Forward GFLOPs of one sample: the model at n_tokens, plus the scorer of a learned mesh."""
    checkpoint = f"outputs/checkpoints/{checkpoint}"
    saved = torch.load(checkpoint, map_location="cpu")
    if saved["model_class"] == "ViT":
        model = build_model_from_checkpoint(ViT, checkpoint).eval()
        inputs = (torch.zeros(1, saved["model_config"]["input_channels"], *WING_GRID),)
    else:
        model = build_model_from_checkpoint(AMRTransformer, checkpoint).eval()
        inputs = (torch.zeros(n_tokens, model.token_dim), [n_tokens])
    scorer = build_model_from_checkpoint(RefinementNet, f"outputs/checkpoints/{scorer_file}").eval() if scorer_file else None

    with FlopCounterMode(display=False) as counter:
        model(*inputs)
        if scorer:
            scorer(torch.zeros(1, *WING_GRID, 5))
    return counter.get_total_flops() / 1e9


def plot_performance():
    """Relative L2 error of the ViT and the uniform and learned AMT against the token count, GFLOPs and peak GPU memory."""
    errors = [overall_l2_error(path) for _, _, path, _, _ in PERFORMANCE_RUNS]
    counts = [token_counts(n, scorer) for _, n, _, _, scorer in PERFORMANCE_RUNS]
    # gflops = [inference_gflops(checkpoint, round(c.mean()), scorer)
    #           for (_, _, _, checkpoint, scorer), c in zip(PERFORMANCE_RUNS, counts)]
    # memory = [peak_gpu_gib(checkpoint) for _, _, _, checkpoint, _ in PERFORMANCE_RUNS]

    # The rows of the thesis table: tokens, Cp, Cf_tau, Cf_z, CL, CD, CMz in %, GFLOPs, peak GPU memory in GiB
    for (model, n, path, *_), count in zip(PERFORMANCE_RUNS, counts):
        print(f"{model:14s} {n:5d} {count.mean():6.0f} +- {count.std():4.0f} " + " ".join(f"{100 * e:5.2f}" for e in table_errors(path)))
    # for (model, n, path, *_), count, flops, gib in zip(PERFORMANCE_RUNS, counts, gflops, memory):
    #     print(f"{model:14s} {n:5d} {count.mean():6.0f} +- {count.std():4.0f} " + " ".join(f"{100 * e:5.2f}" for e in table_errors(path))
    #           + f" {flops:6.2f} {gib:5.2f}")

    costs = {"tokens": ([c.mean() for c in counts], [c.std() for c in counts], "Number of tokens", [128, 256, 512, 1024, 2048]),
            #  "gflops": (gflops, [0] * len(gflops), "Inference GFLOPs per sample", [1, 2, 5, 10, 20]),
            #  "gpu_memory": (memory, [0] * len(memory), "Peak training GPU memory (GiB)", [0.5, 1, 2, 5])
             }
    for cost_name, (cost, cost_std, xlabel, xticks) in costs.items():
        curves = {}
        for model in dict.fromkeys(run[0] for run in PERFORMANCE_RUNS):
            rows = [i for i, run in enumerate(PERFORMANCE_RUNS) if run[0] == model]
            curves[model] = ([cost[i] for i in rows], [errors[i] for i in rows], [cost_std[i] for i in rows])
        # The learned mesh of 512 tokens trained with mesh perturbation, only on the token axis
        reference = (512, overall_l2_error(PERTURBATION_EVALUATIONS[0.55]), "AMT (learned, perturbed)") if cost_name == "tokens" else None
        plot_error_vs_cost(curves, xlabel=xlabel, ylabel="Relative $L_2$ error of surface fields (%)", xticks=xticks,
                           reference=reference, save_path=f"outputs/plots/performance_l2_vs_{cost_name}.png")


def plot_performance_coefficients():
    """Overall aerodynamic coefficient error of the ViT and the uniform and learned AMT against the token count."""
    errors = [overall_coefficient_error(path) for _, _, path, _, _ in PERFORMANCE_RUNS]
    counts = [token_counts(n, scorer) for _, n, _, _, scorer in PERFORMANCE_RUNS]

    curves = {}
    for model in dict.fromkeys(run[0] for run in PERFORMANCE_RUNS):
        rows = [i for i, run in enumerate(PERFORMANCE_RUNS) if run[0] == model]
        curves[model] = ([counts[i].mean() for i in rows], [errors[i] for i in rows], [counts[i].std() for i in rows])

    reference = (512, overall_coefficient_error(PERTURBATION_EVALUATIONS[0.55]), "AMT (learned, perturbed)")
    plot_error_vs_cost(curves, ylabel="Relative error of coefficients (%)", xticks=[128, 256, 512, 1024, 2048],
                       reference=reference, save_path="outputs/plots/performance_coefficients_vs_tokens.png")


def plot_model_floor_comparison():
    """Relative L2 error of every surface field of the 512-token runs of PERFORMANCE_RUNS against their reconstruction floor."""
    # The configs of these runs were not kept, so the learned mesh names its scorer as in compare_models.py
    meshes = {"AMT (uniform)": ("configs/deterministic_transformer/deterministic_UNIFORM_8x8.yaml", {}),
              "AMT (learned)": ("configs/learned_transformer/tokens=512/learned_n=512_affine=1-2.yaml",
                                {"checkpoint_file": "outputs/checkpoints/2026-10-13_scorer_n=512_square_mesh.pt", "max_patch_size": 32})}
    # The ViT head predicts every pixel of its patch, so no mesh bounds its error
    floors = {"ViT": np.zeros(3)}
    floors.update({model: mesh_l2_floors(config, orders=(2,), overrides=overrides)[2] for model, (config, overrides) in meshes.items()})
    errors = {model: np.array([row[-2] for row in read_evaluation(path)["l2"][:3]]) for model, n, path, _, _ in PERFORMANCE_RUNS if n == 512}
    plot_model_floor(errors, floors, save_path="outputs/plots/model_floor_n=512.png")


def plot_cell_shape_floor_comparison():
    """Reconstruction floor of the uniform 2:1, 1:1 and 1:2 meshes against their token count."""
    # The same 300 wings as the threshold mesh comparison of scripts/mesh_floor.py
    plot_cell_shape_floors(cell_shape_floors(sample_rows(300)), save_path="outputs/plots/cell_shape_floors.png")


if __name__ == "__main__":
    # plot_perturbation()
    # plot_perturbation_coefficients()
    # plot_perturbation_robustness_curves()
    # plot_cell_shape_floor_comparison()
    # plot_performance()
    plot_performance_coefficients()
    # plot_model_floor_comparison()
    # plot_polynomial_orders()
    # plot_polynomial_coefficients()
    # plot_polynomial_floor_comparison()
