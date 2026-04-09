# Transformer Based Adaptive Mesh Reduction for Aerodynamic Prediction 
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

To run the actual training/eval algorithm you can execute the main.py function or call the script directly in the bash. It takes the following parameters

| Parameter     | Description |
|---------------|-------------|
| `input_file`  | Single `.npz` or `.npy` file with shape `[N, H, W, C]` |
| `target_file` | Single `.npz` or `.npy` file with shape `[N, H, W, C]` |
| `val_split`   | Fraction of samples held out for validation (default = 0.1) |
| `d_model`     |  |
| `n_layers`    |  |
| `n_heads`     |  |
| `min_depth`   | Quadtree minimum subdivision depth (default = 2) |
| `max_depth`   | Quadtree maximum subdivision depth (default = 6) |
| `epochs`      | (default = 100)
| `batch_size`  | (default=8)
| `warmup_steps`| (default=4000)
| `num_workers` | (default=0)
| `seed`        | (default=42)

Example run with synthethic data to verify the pipeline

```console
python train.py --epochs 5 --n_samples 64 --batch_size 4
```

## References