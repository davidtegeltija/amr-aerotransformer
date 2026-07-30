import argparse
from datetime import datetime
import os
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
from src.model.amr_model import AMRTransformer
from src.model.refinement_net import RefinementNet
from src.model.vit_model import ViT
from src.train import train_transformer, train_scorer_supervised, train_vit
from src.utils.config_utils import load_config
from src.utils.data_utils import build_dataset, geometry_disjoint_split
from src.utils.model_utils import load_checkpoint
from src.utils.train_utils import plot_loss_curves
from src.utils.geometry_utils import patch_sizes_to_depth_bounds


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
        load_checkpoint(scorer, checkpoint_file, device)
        return LearnedCollateFn(scorer, min_depth=min_depth, max_depth=max_depth, offset=args["offset"])

    raise SystemExit(f"No collate defined for model_trained {model_trained!r}")


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, collate_fn, device: torch.device) -> DataLoader:
    """Build a DataLoader with the batch/worker settings shared by all training loops."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                      num_workers=num_workers, collate_fn=collate_fn, 
                      pin_memory=device.type == "cuda")


def train_model(args, model, train_loader, val_loader, device, writer, save_path):
    """Dispatch to the training loop matching model_trained."""
    model_trained = args["model_trained"]

    # ViT baseline
    if model_trained == "vit":
        return train_vit(
            model, train_loader, val_loader, device,
            epochs=args["epochs"],
            save_path=save_path,
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
            save_path=save_path,
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
            save_path=save_path,
            writer=writer,
        )

    raise SystemExit(f"No training loop defined for model_trained {model_trained!r}")


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Train AMRTransformer",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=
        """
        Examples
        --------
        # Pair a model config with a data config:
        python main.py --config configs/baseline.yaml --data configs/data/wing.yaml

        # Override a single value at runtime:
        python main.py --config configs/baseline.yaml --data configs/data/wing.yaml --override epochs=5

        # Name the run, so its logs and checkpoint do not collide with a concurrent one:
        python main.py --config configs/baseline.yaml --data configs/data/wing.yaml --name baseline_lr1e4

        # Detach and keep running after the terminal/SSH session closes:
        nohup python -u -m main --config configs/baseline.yaml --data configs/data/wing.yaml --name baseline_lr1e4 &
        """,
    )

    parser.add_argument("--config", type=str, required=True, help="Path to a YAML model config file (configs/*.yaml)")
    parser.add_argument("--data", type=str, required=True, help="Path to a YAML data config, merged over --config (configs/data/*.yaml)")
    parser.add_argument("--name", type=str, default=None, help="Identifier for this run. Names the log directory, the loss plot and the checkpoint (default: <model-config>_<data-config>)")
    parser.add_argument("--override", nargs="*", metavar="KEY=VALUE", help="Override specific config values at runtime, e.g. --override epochs=5 batch_size=16")

    cli = parser.parse_args(args)

    # ----------------------------------------------------------------
    # Run Setup (log file, checkpoint dir, TensorBoard) and Load Arguments
    # ----------------------------------------------------------------
    # Make log file
    run_name = cli.name or f"{Path(cli.config).stem}_{Path(cli.data).stem}"
    log_dir = Path("outputs/logs") / f"{datetime.now().strftime("%Y-%m-%d_%H-%M")}_{run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"

    # buffering=1: line-buffered, so the log can be tailed live (like -u)
    # Redirect the OS-level fds too, replicating shell's "> .log 2>&1"
    log_file = open(log_path, "w", buffering=1)
    os.dup2(log_file.fileno(), sys.stdout.fileno())
    os.dup2(log_file.fileno(), sys.stderr.fileno())
    sys.stdout = log_file
    sys.stderr = log_file

    # First thing in the log: how this run was invoked. sys.argv is just the
    # script path when main() is called with an explicit list (the IDE branch),
    # so the parsed namespace is what is authoritative in both branches.
    print(f"Command: {sys.argv}")
    print(f"CLI args: {vars(cli)}")

    # Make checkpoints dir
    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / f"{datetime.now().strftime("%Y-%m-%d")}_{run_name}.pt"

    args = load_config(cli.config, cli.data)

    # Apply any runtime overrides, casting to the type of the existing value
    if cli.override:
        for item in cli.override:
            key, value = item.split("=", 1)
            existing = args.get(key)
            if existing is not None:
                value = type(existing)(value)  # preserve int/float/str type
            args[key] = value

    # TensorBoard writer (one run dir per config + timestamp)
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_text("config", f"```yaml\n{yaml.safe_dump(args, sort_keys=False)}```", 0)
    print(f"TensorBoard logging to {log_dir}  (view: tensorboard --logdir outputs/logs)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ----------------------------------------------------------------
    # Build Dataset
    # ----------------------------------------------------------------
    print("\n======== Building Dataset ========")
    seed = args["seed"]
    dataset, dataset_type = build_dataset(args)

    if dataset_type == "wing_dataset":
        # Split by geometry, not by row. Each geometry spans many rows (one per
        # operating condition); a row-level split leaks the same wing into both
        # train and val, so val_loss measures interpolation, not generalization.
        train_dataset, val_dataset, _ = split_by_group_id(dataset, dataset.geometry_ids(), args["val_split"], seed, "Geometry")

    elif dataset_type == "cavity_dataset":
        # Split by case, not by pair. Consecutive frames of one simulation are
        # highly correlated; a pair-level split leaks a case into both train and
        # val, so val_loss would measure interpolation, not generalization.
        train_dataset, val_dataset, _ = split_by_group_id(dataset, dataset.case_ids(), args["val_split"], seed, "Case")

    else:
        # Generated samples share no geometry or case, so there is nothing to
        # keep disjoint — a plain random split is the honest one here.
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
            model = AMRTransformer(
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

        train_loss_history, val_loss_history = train_model(args, model, train_loader, val_loader, device, writer, save_path)

    # ----------------------------------------------------------------
    # ViT baseline branch (standalone, no AMR / scorer / quadtree)
    # ----------------------------------------------------------------
    else:
        H, W = dataset.H, dataset.W
        patch_size = args["min_patch_size"]
        d_model = args["d_model"]
        n_heads = args["n_heads"]
        pos_embedding = args["pos_embedding"]

        if H % patch_size != 0 or W % patch_size != 0:
            raise SystemExit(f"image_size {(H, W)} must be divisible by patch_size {patch_size}")

        if d_model % n_heads != 0:
            raise SystemExit(f"d_model {d_model} must be divisible by n_heads {n_heads}")

        if pos_embedding == "sincos" and d_model % 4 != 0:
            raise SystemExit(f"d_model {d_model} must be divisible by 4 when pos_embedding='sincos'")

        print("\n======== Model (ViT baseline) ========")
        model = ViT(
            image_size=(H, W),
            patch_size=patch_size,
            input_channels=input_channels,
            output_channels=output_channels,
            d_model=d_model,
            n_layers=args["n_layers"],
            n_heads=n_heads,
            d_ff=args["d_ff"],
            dropout=args["dropout"],
            pos_embedding=pos_embedding)
        print(f"ViT parameters: {sum(p.numel() for p in model.parameters()):,}")

        collate_fn = build_collate_fn(args, train_dataset, input_channels, device)

        print("\n======== Training ========")
        train_loader = make_loader(train_dataset, args["batch_size"], True, args["num_workers"], collate_fn, device)
        val_loader = make_loader(val_dataset, args["batch_size"], False, args["num_workers"], collate_fn, device)

        train_loss_history, val_loss_history = train_model(args, model, train_loader, val_loader, device, writer, save_path)

    plot_loss_curves(train_loss_history, val_loss_history, args["epochs"], save_path=f"outputs/loss/{run_name}_loss.png")
    writer.close()


if __name__ == "__main__":
    # When calling the function from bash
    if len(sys.argv) > 1:
        print(sys.argv)
        main()

    # When calling the function from IDE
    else:
        print("No CLI args given -> using the hardcoded IDE configuration. Pass --config and --data to run something else.")        
        model_config = "configs/deterministic_transformer.yaml"
        data_config = "configs/data/overfit.yaml"
        name = "deterministic_transformer_overfit_400_tokens"

        args = ["--config", model_config, "--data", data_config, "--name", name]

        print(args)
        main(args)
