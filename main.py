import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.utils.tensorboard import SummaryWriter

from src.amr.refinement_criteria import CRITERIA_REGISTRY
from src.amr.oracle_depth import calibrate_global_tolerance
from src.data.collate_fn import DeterministicCollateFn, LearnedCollateFn, ScorerCollateFn, VitCollateFn
from src.data.dataset import AeroDataset
from src.data.cavity_dataset import CavityDataset
from src.data.synthetic_dataset import SyntheticDataset
from src.model.amr_model import AdaptiveMeshAeroModel
from src.model.refinement_net import RefinementNet
from src.model.vit import ViT
from src.train import train_transformer, train_scorer_supervised, train_vit
from src.utils.data_utils import geometry_disjoint_split
from src.utils.train_utils import plot_loss_curves
from src.utils.geometry_utils import patch_sizes_to_depth_bounds


def load_config(path: str, data_path: str) -> Dict:
    """Load a YAML model config and merge a YAML data config over it. Return a flat namespace mimicking the argparse args."""
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


def load_state_dict_partial(module, path, device, strip_prefix=None):
    """Load a checkpoint into ``module`` tolerantly (strict=False).

    Drops shape-mismatched params (so e.g. a constant-head checkpoint can
    warm-start an affine_output model) and optionally strips a leading key
    prefix. Scorer checkpoints are saved namespaced as ``scorer.*`` (see
    ``save_checkpoint(..., prefix="scorer")``); pass ``strip_prefix="scorer."``
    to load them into a standalone ``RefinementNet``.
    """
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if strip_prefix:
        state = {(k[len(strip_prefix):] if k.startswith(strip_prefix) else k): v
                 for k, v in state.items()}
    module_sd = module.state_dict()
    dropped = [k for k, v in state.items() if k in module_sd and v.shape != module_sd[k].shape]
    if dropped:
        state = {k: v for k, v in state.items() if k not in dropped}
        print(f"  re-initialising {len(dropped)} shape-mismatched param(s): {dropped}")
    missing, unexpected = module.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint {path}  (missing: {len(missing)}, unexpected: {len(unexpected)})")


def split_by_group_id(dataset: Dataset, group_ids, val_split: float, seed: int, group_name: str):
    """Split a dataset into disjoint train/val/test subsets by group id (geometry or case)."""
    try:
        train_idx, val_idx, test_idx = geometry_disjoint_split(group_ids, val_split, seed)
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)
        print(f"{group_name}-disjoint split (seed={seed}): "
              f"{len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")
        return train_dataset, val_dataset, test_dataset
    except ValueError as e:
        print(f"WARNING: {e}")
        print("Falling back to validating on the full training set. This is an "
              "overfit sanity check only — val_loss is NOT a generalization metric "
              f"when train and val share a {group_name.lower()}.")
        return dataset, dataset, dataset


def build_collate_fn(args: Dict, train_dataset: Dataset, input_channels: int, device: torch.device):
    """Build the collate that supplies each training loop with its batch layout."""
    model_trained = args["model_trained"]

    # ViT baseline
    if model_trained == "vit":
        return VitCollateFn()

    min_depth = args["min_depth"]
    max_depth = args["max_depth"]

    # Deterministic mesh: physics-based AMR criterion.
    if model_trained == "deterministic_transformer":
        refinement_criteria = args["refinement_criteria"]
        if refinement_criteria not in CRITERIA_REGISTRY:
            valid = ", ".join(sorted(CRITERIA_REGISTRY))
            raise SystemExit(f"Unknown refinement_criteria {refinement_criteria!r}.\nAvailable options are: {valid}")

        return DeterministicCollateFn(
            refinement_criteria=CRITERIA_REGISTRY[refinement_criteria],
            min_depth=min_depth,
            max_depth=max_depth,
        )

    # Learned-scorer training: oracle depth targets from a calibrated tolerance.
    if model_trained == "scorer":
        n_target = args["n_target"]
        n_calib = min(args["calib_samples"], len(train_dataset))
        calib_targets = [np.asarray(train_dataset[i]["target"], dtype=np.float32) for i in range(n_calib)]
        tol = calibrate_global_tolerance(calib_targets, n_target=n_target, min_depth=min_depth, max_depth=max_depth)
        # max_depth is already the reachable depth (derived from min_patch_size by
        # patch_sizes_to_depth_bounds), so it is the reachable cap directly.
        print(f"Oracle target: global tol={tol:.4g} (n_target={n_target}, "
              f"calib n={n_calib}, reachable_depth={max_depth})")
        return ScorerCollateFn(tol=tol, min_depth=min_depth, max_depth=max_depth)

    # Learned-mesh transformer training: a frozen, pretrained scorer defines the mesh.
    if model_trained == "learned_transformer":
        checkpoint_file = args.get("checkpoint_file")
        if checkpoint_file is None:
            raise SystemExit("model_trained 'learned_transformer' requires checkpoint_file pointing to a trained scorer")
        if not Path(checkpoint_file).is_file():
            raise SystemExit(f"checkpoint_file {checkpoint_file!r} does not exist. It must point to a trained scorer checkpoint.")

        scorer = RefinementNet(input_channels=input_channels)
        load_state_dict_partial(scorer, checkpoint_file, device, strip_prefix="scorer.")
        return LearnedCollateFn(scorer, min_depth=min_depth, max_depth=max_depth, offset=args["offset"])

    raise SystemExit(f"No collate defined for model_trained {model_trained!r}")


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, collate_fn, device: torch.device) -> DataLoader:
    """Build a DataLoader with the batch/worker settings shared by all training loops."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                      num_workers=num_workers, collate_fn=collate_fn, 
                      pin_memory=device.type == "cuda")


def train_model(args, model, train_loader, val_loader, device, writer):
    """Dispatch to the training loop matching model_trained."""
    model_trained = args["model_trained"]

    # ViT baseline
    if model_trained == "vit":
        return train_vit(
            model, train_loader, val_loader, device,
            epochs=args["epochs"],
            writer=writer,
        )

    # Transformer training — identical loop on a deterministic or a learned mesh;
    # only the collate that produced the tokens differs.
    if model_trained in ("deterministic_transformer", "learned_transformer"):
        return train_transformer(
            model, train_loader, val_loader, device,
            epochs=args["epochs"],
            d_model=args["d_model"],
            warmup_steps=args["warmup_steps"],
            writer=writer,
        )

    # Learned-scorer training
    if model_trained == "scorer":
        return train_scorer_supervised(
            model, train_loader, val_loader, device,
            epochs=args["epochs"],
            min_depth=args["min_depth"],
            max_depth=args["max_depth"],
            tv_weight=args["tv_weight"],
            decision_weight=args["decision_weight"],
            decision_margin=args["decision_margin"],
            decision_temp=args["decision_temp"],
            writer=writer,
        )

    raise SystemExit(f"No training loop defined for model_trained {model_trained!r}")


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Train AdaptiveMeshAeroModel",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=
        """
        Examples
        --------
        # Pair a model config with a data config:
        python main.py --config configs/baseline.yaml --data configs/data/wing.yaml

        # Override a single value at runtime:
        python main.py --config configs/baseline.yaml --data configs/data/wing.yaml --override epochs=5
        """,
    )

    parser.add_argument("--config", type=str, required=True, help="Path to a YAML model config file (configs/*.yaml)")
    parser.add_argument("--data", type=str, required=True, help="Path to a YAML data config, merged over --config (configs/data/*.yaml)")
    parser.add_argument("--override", nargs="*", metavar="KEY=VALUE", help="Override specific config values at runtime, e.g. --override epochs=5 batch_size=16")

    cli = parser.parse_args(args)
    args = load_config(cli.config, cli.data)

    # Apply any runtime overrides, casting to the type of the existing value
    if cli.override:
        for item in cli.override:
            key, value = item.split("=", 1)
            existing = args.get(key)
            if existing is not None:
                value = type(existing)(value)  # preserve int/float/str type
            args[key] = value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ----------------------------------------------------------------
    # TensorBoard writer (one run dir per config + timestamp)
    # ----------------------------------------------------------------
    # Both stems, since one model config now runs against several data configs and
    # the model stem alone would collide across datasets.
    config_name = f"{Path(cli.config).stem}_{Path(cli.data).stem}"
    run_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{config_name}"
    log_dir = Path("outputs/logs") / run_name
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_text("config", f"```yaml\n{yaml.safe_dump(args, sort_keys=False)}```", 0)
    print(f"TensorBoard logging to {log_dir}  (view: tensorboard --logdir outputs/logs)")

    # ----------------------------------------------------------------
    # Build Dataset
    # ----------------------------------------------------------------
    print("\n======== Building Dataset ========")
    seed = args["seed"]
    dataset_type = args["dataset"]

    if dataset_type == "wing_dataset" and args["input_file"] is not None:
        print(f"Using data from {args['input_file']}")
        dataset = AeroDataset(input_path=args["input_file"], target_path=args["target_file"], index_path=args["index_file"])
        # Split by geometry, not by row. Each geometry spans many rows (one per
        # operating condition); a row-level split leaks the same wing into both
        # train and val, so val_loss measures interpolation, not generalization.
        train_dataset, val_dataset, _ = split_by_group_id(dataset, dataset.geometry_ids(), args["val_split"], seed, "Geometry")

    elif dataset_type == "cavity_dataset" and args["input_file"] is not None:
        print(f"Using cavity next-step data from {args['input_file']}")
        dataset = CavityDataset(input_path=args["input_file"])
        # Split by case, not by pair. Consecutive frames of one simulation are
        # highly correlated; a pair-level split leaks a case into both train and
        # val, so val_loss would measure interpolation, not generalization.
        train_dataset, val_dataset, _ = split_by_group_id(dataset, dataset.case_ids(), args["val_split"], seed, "Case")

    else:
        print("No input and target data provided -> using synthetic dataset.")
        dataset = SyntheticDataset(n_samples=64, seed=seed)
        n_val = max(1, int(args["val_split"] * len(dataset)))
        train_dataset, val_dataset = random_split(
            dataset, [len(dataset) - n_val, n_val], generator=torch.Generator().manual_seed(seed))
        
    input_channels = dataset.input_channels
    output_channels = dataset.output_channels

    # ----------------------------------------------------------------
    # AMR Model
    # ----------------------------------------------------------------
    model_trained = args["model_trained"]

    if model_trained != "vit":
        print("\n======== Model (AMR) ========")

        # Patch-size bounds -> Quadtree depth bounds
        H, W = dataset.H, dataset.W
        min_patch_size = args["min_patch_size"]
        max_patch_size = args["max_patch_size"]
        min_depth, max_depth = patch_sizes_to_depth_bounds(H, W, min_patch_size, max_patch_size)
        args["min_depth"] = min_depth
        args["max_depth"] = max_depth
        print(f"Patch-size bounds -> depth bounds:\nmin_patch_size={min_patch_size}, "
            f"max_patch_size={max_patch_size} -> min_depth={min_depth}, max_depth={max_depth} "
            f"(grid {H}x{W})")
        
        # Model
        if model_trained == "scorer":
            model = RefinementNet(input_channels=input_channels).to(device)
            print(f"Scorer parameters: {sum(p.numel() for p in model.parameters()):,}")
        else:
            model = AdaptiveMeshAeroModel(
                input_channels=input_channels,
                output_channels=output_channels,
                d_model=args["d_model"],
                n_layers=args["n_layers"],
                n_heads=args["n_heads"],
                d_ff=args["d_ff"],
                dropout=args["dropout"],
                affine_output=args["affine_output"],
            )
            print(f"Model parameters: {model.count_parameters():,}")

        collate_fn = build_collate_fn(args, train_dataset, input_channels, device)

        # Train
        print("\n======== Training ========")
        train_loader = make_loader(train_dataset, args["batch_size"], True, args["num_workers"], collate_fn, device)
        val_loader = make_loader(val_dataset, args["batch_size"], False, args["num_workers"], collate_fn, device)

        train_loss_history, val_loss_history = train_model(args, model, train_loader, val_loader, device, writer)

    # ----------------------------------------------------------------
    # ViT baseline branch (standalone, no AMR / scorer / quadtree)
    # ----------------------------------------------------------------
    else:
        H, W = dataset.H, dataset.W
        patch_size = args["min_patch_size"]
        n_hidden = args["n_hidden"]
        n_head = args["n_head"]
        pos_embedding = args["pos_embedding"]

        if H % patch_size != 0 or W % patch_size != 0:
            raise SystemExit(f"image_size {(H, W)} must be divisible by patch_size {patch_size}")

        if n_hidden % n_head != 0:
            raise SystemExit(f"n_hidden {n_hidden} must be divisible by n_head {n_head}")

        if pos_embedding == "sincos" and n_hidden % 4 != 0:
            raise SystemExit(f"n_hidden {n_hidden} must be divisible by 4 when pos_embedding='sincos'")

        print("\n======== Model (ViT baseline) ========")
        model = ViT(
            image_size=(H, W),
            patch_size=patch_size,
            fun_dim=input_channels,
            out_dim=output_channels,
            n_layers=args["n_layers"],
            n_hidden=n_hidden,
            n_head=n_head,
            mlp_ratio=args["mlp_ratio"],
            dropout=args["dropout"],
            pos_embedding=pos_embedding)
        print(f"ViT parameters: {sum(p.numel() for p in model.parameters()):,}")

        collate_fn = build_collate_fn(args, train_dataset, input_channels, device)

        print("\n======== Training ========")
        train_loader = make_loader(train_dataset, args["batch_size"], True, args["num_workers"], collate_fn, device)
        val_loader = make_loader(val_dataset, args["batch_size"], False, args["num_workers"], collate_fn, device)

        train_loss_history, val_loss_history = train_model(args, model, train_loader, val_loader, device, writer)

    plot_loss_curves(train_loss_history, val_loss_history, args["epochs"], save_path=f"outputs/loss/{config_name}_config_loss.png")
    writer.close()

if __name__ == "__main__":
    # When calling the function from bash
    if len(sys.argv) > 1:
        print(sys.argv)
        main()

    # When calling the function from IDE
    else:
        args = ["--config", "configs/deterministic_transformer.yaml", "--data", "configs/data/overfit.yaml"]
        print(args)
        main(args)
