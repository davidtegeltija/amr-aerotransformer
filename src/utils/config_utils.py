"""
========================================================================
Config loading — the one place a run's YAML files become an args dict.
========================================================================

Contents
--------
load_config          - merge a model config with a data config and validate the result
resolve_depth_bounds - turn the configured patch sizes into quadtree depth bounds

A run is described by two YAML files: a model config (``configs/*.yaml``,
what to train and with which hyperparameters) and a data config
(``configs/data/*.yaml``, which dataset to train it on). Splitting them lets
any model config pair with any dataset, but it also means neither half alone
is a runnable description of a run — the merge here is what produces one.

Every entry point (``main.py`` and the scripts in ``scripts/``) goes through
this function, so a config that is wrong is rejected once, up front, with the
offending file named, rather than failing deep inside a DataLoader worker.
"""

from pathlib import Path
from typing import Dict
import yaml

from src.utils.geometry_utils import patch_sizes_to_depth_bounds


# ---------------------------------------------------------------------------
# Config loading — model config + data config -> one validated args dict
# ---------------------------------------------------------------------------

def load_config(path: str, data_path: str) -> Dict:
    """
    Load a YAML model config and merge a YAML data config over it.

    Args:
        path      : path to the model config (``configs/*.yaml``)
        data_path : path to the data config (``configs/data/*.yaml``), whose
                    keys are merged over the model config

    Returns:
        A flat dict of the merged keys, mimicking an argparse namespace.

    Raises:
        SystemExit: if 'model_trained' or 'dataset' names an unknown option, or
            a data file the chosen dataset requires is missing from disk. These
            are unrecoverable startup errors for the entry points that call
            this, so they exit with a message rather than a traceback.
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Add data config to cfg
    with open(data_path, "r") as f:
        cfg.update(yaml.safe_load(f))

    MODEL_TRAINED_OPTIONS = (
        "deterministic_transformer",  # AMR transformer on a criteria-driven mesh
        "learned_transformer",        # AMR transformer on a frozen-scorer mesh
        "scorer",                     # RefinementNet trained against oracle depths
        "vit",                        # dense ViT baseline, no quadtree
    )

    DATASET_OPTIONS = ("wing_dataset", "cavity_dataset", "synthetic_dataset")

    model_trained = cfg.get("model_trained")
    if model_trained not in MODEL_TRAINED_OPTIONS:
        valid = ", ".join(MODEL_TRAINED_OPTIONS)
        raise SystemExit(f"Invalid model_trained {model_trained!r} in {path}.\nValid options are: {valid}")

    dataset_type = cfg.get("dataset")
    if dataset_type not in DATASET_OPTIONS:
        raise SystemExit(f"Invalid dataset {dataset_type!r} in {data_path}.\nValid options are: {', '.join(DATASET_OPTIONS)}")

    # Null input_file selects the synthetic dataset; wing needs three arrays, cavity one root.
    if cfg.get("input_file") is not None:
        path_keys = ("input_file", "target_file", "index_file") if dataset_type == "wing_dataset" else ("input_file",)
        for key in path_keys:
            value = cfg.get(key)
            if value is None or not Path(value).exists():
                raise SystemExit(f"dataset {dataset_type!r} requires {key}, got {value!r} which does not exist")

    print(cfg)  # Print out the whole yaml file so it can be logged
    return cfg


# ---------------------------------------------------------------------------
# Patch sizes -> quadtree depth bounds, the one conversion every run shares
# ---------------------------------------------------------------------------

def resolve_depth_bounds(args: Dict, dataset) -> Dict:
    """Derive quadtree depth bounds from the configured patch sizes and store them in ``args``.

    Configs express mesh bounds as pixel patch sizes; the builders, oracle and loss
    all work in integer depths. This is the single conversion point, so every entry
    point gets identical bounds for a given config + grid.

    Args:
        args: Merged config dict; gains ``min_depth`` and ``max_depth``.
        dataset: Dataset supplying the grid dimensions ``H``, ``W``.

    Returns:
        The updated ``args``
    """
    H, W = dataset.H, dataset.W
    min_depth, max_depth = patch_sizes_to_depth_bounds(H, W, args.get("min_patch_size"), args.get("max_patch_size"))
    args["min_depth"] = min_depth
    args["max_depth"] = max_depth
    return args
