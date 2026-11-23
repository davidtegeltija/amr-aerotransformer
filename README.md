# Learned Adaptive Mesh Refinement for Token-Efficient High-Resolution Aerodynamic Prediction
This is an adaptive mesh refinement approach to a transformer-based aerodynamic profile prediction. It is a project I did for my Master Thesis as a part of my Computational Science and Engineering degree at Technical University of Munich (TUM).

## Overview
A flow field on a regular grid is split into a quadtree mesh, each leaf cell becomes one token, and a transformer (Adaptive Mesh Transformer - AMT) predicts the flow per cell as a polynomial that is painted back onto the grid. The mesh comes from one of two sources:

- **Deterministic**: hand-written refinement criteria (`src/amr/refinement_criteria.py`), including uniform meshes.
- **Learned**: a CNN scorer (`src/models/refinement_net.py`) predicts a per-pixel target depth, trained against an oracle mesh calibrated to a token budget `n_target`.

A dense ViT serves as the baseline. Supported datasets are SuperWing (`wing`, steady surface flow plus aerodynamic coefficients) and the CFDBench next-step problems (`cavity`, `cylinder`, `dam`, `tube`).

```
configs/         model configs (per model type) and data configs (configs/data/)
scripts/         evaluation, comparison and visualization scripts
src/
  main.py        training entry point
  amr/           quadtree, deterministic and learned mesh builders, oracle depth
  data/          datasets, train/val/test split, collate functions (mesh + tokens)
  models/        AMRTransformer, RefinementNet, ViT, per-cell reconstruction
  training/      training loops, losses, LR schedulers
  inference/     single-sample prediction
  evaluation/    prediction errors and aerodynamic coefficients
  utils/         config loading, checkpoints, plotting
```

## Setup
The project runs on Python with PyTorch. I suggest installing the dependencies into a virtual environment, created with Python's built-in `venv` module:

```console
python -m venv venv

venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux

pip install -r requirements.txt
```

## How to Train
A run is a **model** config (`configs/<model type>/*.yaml`) paired with a **data** config (`configs/data/*.yaml`), merged at load time. Run from the project root:

```console
python -m src.main --config configs/learned_transformer/tokens=512/learned_n=512_affine=1-2.yaml --data configs/data/wing.yaml
```

The optional flag `--name <run>` names the log directory and checkpoint file, which default to the two config names joined and prefixed with the date. You can change the values without editing the config files by using `--override epochs=5 batch_size=4`. To smoke-test the pipeline without real data, use `--data configs/data/synthetic.yaml`.

To train a model on a learned mesh you need two sequential runs:
1. train a scorer (`configs/scorer/`)
2. train the AMT (`configs/learned_transformer/`) whose `checkpoint_file` points at that scorer checkpoint and whose `min_patch_size` / `max_patch_size` match the scorer's.

The config is a flat YAML mapping. `configs/baseline.yaml` lists every available key with its explanation. The main ones are:

| Key | Description |
|---|---|
| `model_trained` | `deterministic_transformer`, `learned_transformer`, `scorer` or `vit` |
| `min_patch_size`, `max_patch_size` | Finest and coarsest allowed cell size in pixels |
| `affine_input`, `affine_output` | Polynomial order of the input token features (0-1) and of the output prediction (0-2) |
| `refinement_criteria` | Preset from `CRITERIA_REGISTRY` (deterministic mesh) |
| `checkpoint_file`, `offset` | Scorer checkpoint and mesh budget offset (learned mesh) |
| `mesh_perturbation` | Train-time mesh noise (deterministic and learned mesh) |
| `n_target`, `split_rule`, `patch_sides_ratio` | Token budget, oracle split rule and cell shape (scorer) |
| `d_model`, `n_layers`, `n_heads`, `d_ff`, `dropout` | Transformer size (AMT and ViT) |
| `epochs`, `batch_size`, `lr`, `weight_decay`, `warmup_steps`, `schedule` | Training |
| `dataset`, `input_file`, `target_file`, `index_file`, `val_split`, `seed` | Data config |

Checkpoints are saved to `outputs/checkpoints/`, logs and TensorBoard runs to `outputs/logs/`, loss curves to `outputs/loss/`.

## How to Evaluate
The evaluation scripts are configured by the variables at the top of their `__main__` block and run from the project root. To evaluate a single model set `model_config`, `data_config` and `checkpoint_file` in `scripts/evaluate_model.py`, then run

```console
python scripts/evaluate_model.py
```

It saves the plots you select to `outputs/plots/` and writes the test metrics to `outputs/evaluation/<checkpoint>.txt`. A checkpoint that already has an evaluation file is not re-evaluated. Set `test_perturbation` to evaluate an AMT on a perturbed learned mesh.

To compare several models on the same test samples, list them in `MODELS` in `scripts/compare_models.py`. The remaining scripts in `scripts/` produce the thesis figures.

## Monitoring with TensorBoard
Every run logs its losses, learning rate, mean token count and resolved config under `outputs/logs/`:

```console
tensorboard --logdir outputs/logs
```

then open <http://localhost:6006>. The `setuptools<81` pin in `requirements.txt` is needed for TensorBoard to launch.
