# Transformer Based Adaptive Mesh Refinement for Aerodynamic Prediction 
This is an adaptive mesh refinement approach to a Transformer-based aerodynamic profile prediction. It is a project I did for my Master Thesis as a part of my Computational Science and Engineering degree at Technical University of Munich (TUM).

## How to Run
I suggest using a virtual environment. You can easily create one using a python preinstalled packaged venv by running

```console
python -m venv <venv_name>
```

This creates a folder called venv in your project directory. Don't worry, the folder is already added to .gitignore. After creating the environment you have to active it by running

```console
Windows
<venv_name>\Scripts\activate

Linux
source <venv_name>/bin/activate
```

After activating the virtual environment you can install the necessary dependencies with

```console
pip install -r requirements.txt
```
This installs the minimal dependencies. If you want to speed up training you can also install FlashAttention (optional but recommended) with `pip install flash-attn --no-build-isolation`. Unfortunately this only works for Linux OS.

Training and evaluation are driven by a YAML config file. Invoke `main.py` and point it at a config:

```console
python main.py --config configs/baseline.yaml
```

If `--config` is omitted, `configs/baseline.yaml` is used by default. Individual values can be overridden at runtime without editing the file:

```console
python main.py --config configs/baseline.yaml --override epochs=5 batch_size=4
```

To smoke-test the pipeline against synthetic data, leave `input_file` and `target_file` unset (or `null`) in the config — `main.py` will fall back to `SyntheticDataset`.

### Config fields

The config is a flat YAML mapping. See `configs/baseline.yaml` for the canonical defaults.

| Key | Description |
|---|---|
| `input_file` | `.npz`/`.npy` of shape `[N, H, W, C]` (geometry). `null` -> synthetic dataset |
| `target_file` | `.npz`/`.npy` of shape `[N, H, W, C]` (steady flow state) |
| `val_split` | Fraction of samples held out for validation |
| `d_model`, `n_layers`, `n_heads`, `d_ff` | Transformer hyperparameters |
| `min_depth`, `max_depth` | Quadtree minimum/maximum subdivision depth |
| `refinement_mode` | `deterministic` or `learned` |
| `refinement_criteria` | Preset name from `CRITERIA_REGISTRY` (used only when `refinement_mode: deterministic`) |
| `epochs`, `batch_size`, `warmup_steps`, `num_workers` | Standard training knobs |
| `training_phase` | `1` (scorer only, transformer frozen), `2` (scorer + transformer joint), or `null` for standard deterministic training |
| `checkpoint_file` | Path to checkpoint to load before training. For phase 1 use a pretrained-transformer checkpoint; for phase 2 use a phase-1 scorer checkpoint |
| `lambda_budget`, `lambda_smooth` | Regularizer weights for the RefinementNet loss |
| `tau_start_phase1`, `tau_end_phase1`, `tau_start_phase2`, `tau_end_phase2` | Gumbel-softmax temperature schedules per phase |
| `scorer_lr`, `transformer_lr` | Per-module learning rates |
| `n_max` | Max token count used to normalize the budget loss (default `8192` for a 256x128 grid) |

Checkpoints are written under `outputs/checkpoints/` (`phase1_scorer.pt`, `phase2_joint.pt`).

## References