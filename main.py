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
from src.amr.quadtree_tokenizer import QuadtreeTokenizer
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


def load_config(path: str) -> Dict:
    """Load a YAML config and return a flat namespace mimicking the argparse args."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

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
    """Split a dataset into disjoint train/val subsets by group id (geometry or case)."""
    try:
        train_idx, val_idx = geometry_disjoint_split(group_ids, val_split, seed)
        train_dataset, val_dataset = Subset(dataset, train_idx), Subset(dataset, val_idx)
        print(f"{group_name}-disjoint split (seed={seed}): "
              f"{len(train_idx)} train / {len(val_idx)} val")
        return train_dataset, val_dataset
    except ValueError as e:
        print(f"WARNING: {e}")
        print("Falling back to validating on the full training set. This is an "
              "overfit sanity check only — val_loss is NOT a generalization metric "
              f"when train and val share a {group_name.lower()}.")
        return dataset, dataset


def build_collate_fn(args: Dict, train_dataset: Dataset, input_channels: int, device: torch.device):
    """Build the collate that supplies each training loop with its batch layout."""
    # ViT baseline
    if args.get("model_type") == "vit":
        return VitCollateFn()

    min_depth = args.get("min_depth")
    max_depth = args.get("max_depth")

    # Deterministic mesh: physics-based AMR criterion.
    if args.get("refinement_mode") == "deterministic":
        refinement_criteria = args.get("refinement_criteria")
        if refinement_criteria not in CRITERIA_REGISTRY:
            valid = ", ".join(sorted(CRITERIA_REGISTRY))
            raise KeyError(f"Unknown refinement_criteria {refinement_criteria!r}.\nAvailable options are: {valid}")
        
        tokenizer = QuadtreeTokenizer(
            min_depth=min_depth,
            max_depth=max_depth,
            refinement_criteria=CRITERIA_REGISTRY[refinement_criteria],
        )
        return DeterministicCollateFn(tokenizer)

    # Learned-scorer training: oracle depth targets from a calibrated tolerance.
    if args.get("refinement_mode") == "learned" and args.get("learned_training_mode") == "scorer":
        n_target = args.get("n_target", 256)
        n_calib = min(args.get("calib_samples", 64), len(train_dataset))
        calib_targets = [np.asarray(train_dataset[i]["target"], dtype=np.float32) for i in range(n_calib)]
        tol = calibrate_global_tolerance(calib_targets, n_target=n_target, min_depth=min_depth, max_depth=max_depth)
        # max_depth is already the reachable depth (derived from min_patch_size by
        # patch_sizes_to_depth_bounds), so it is the reachable cap directly.
        print(f"Oracle target: global tol={tol:.4g} (n_target={n_target}, "
              f"calib n={n_calib}, reachable_depth={max_depth})")
        return ScorerCollateFn(tol=tol, min_depth=min_depth, max_depth=max_depth)

    # Learned-mesh transformer training: a frozen, pretrained scorer defines the mesh.
    elif args.get("refinement_mode") == "learned" and args.get("learned_training_mode") == "transformer":
        if args.get("checkpoint_file") is None:
            raise SystemExit("refinement_mode 'learned' with learned_training_mode 'transformer' requires checkpoint_file pointing to a trained scorer")

        scorer = RefinementNet(input_channels=input_channels)
        load_state_dict_partial(scorer, args.get("checkpoint_file"), device, strip_prefix="scorer.")
        return LearnedCollateFn(scorer, min_depth=min_depth, max_depth=max_depth, offset=args.get("offset", 0.0))


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, collate_fn, device: torch.device) -> DataLoader:
    """Build a DataLoader with the batch/worker settings shared by all training loops."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                      num_workers=num_workers, collate_fn=collate_fn, 
                      pin_memory=device.type == "cuda")


def train_model(args, model, train_loader, val_loader, device, writer):
    """Dispatch to the training loop matching model_type / learned_training_mode."""
    # ViT baseline
    if args.get("model_type") == "vit":
        return train_vit(
            model, train_loader, val_loader, device,
            epochs=args.get("epochs"),
            writer=writer,
        )

    # Deterministic-mesh transformer training
    if args.get("refinement_mode") == "deterministic":
        return train_transformer(
            model, train_loader, val_loader, device,
            epochs=args.get("epochs"), 
            d_model=args.get("d_model"),
            warmup_steps=args.get("warmup_steps"), 
            writer=writer,
        )

    # Learned-scorer training
    if args.get("refinement_mode") == "learned" and args.get("learned_training_mode") == "scorer":
        return train_scorer_supervised(
            model, train_loader, val_loader, device,
            epochs=args.get("epochs"),
            min_depth=args.get("min_depth"),
            max_depth=args.get("max_depth"),
            tv_weight=args.get("tv_weight", 0.0),
            decision_weight=args.get("decision_weight", 0.0),
            decision_margin=args.get("decision_margin", 0.0),
            decision_temp=args.get("decision_temp", None),
            writer=writer,
        )
    
    # Learned-mesh transformer training
    elif args.get("refinement_mode") == "learned" and args.get("learned_training_mode") == "transformer":
        return train_transformer(
            model, train_loader, val_loader, device,
            epochs=args.get("epochs"), 
            d_model=args.get("d_model"),
            warmup_steps=args.get("warmup_steps"), 
            writer=writer,
        )


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Train AdaptiveMeshAeroModel",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=
        """
        Examples
        --------
        # Run with a selected config:
        python main.py --config configs/baseline.yaml

        # Override a single value at runtime:
        python main.py --config configs/baseline.yaml --override epochs=5
        """,
    )

    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to a YAML config file (default: configs/baseline.yaml)")
    parser.add_argument("--override", nargs="*", metavar="KEY=VALUE", help="Override specific config values at runtime, e.g. --override epochs=5 batch_size=16")

    cli = parser.parse_args(args)
    args = load_config(cli.config)

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
    config_name = Path(cli.config).stem
    run_name = f"{config_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_dir = Path("runs") / run_name
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_text("config", f"```yaml\n{yaml.safe_dump(args, sort_keys=False)}```", 0)
    print(f"TensorBoard logging to {log_dir}  (view: tensorboard --logdir runs)")

    # ----------------------------------------------------------------
    # Build Dataset
    # ----------------------------------------------------------------
    print("\n======== Building Dataset ========")
    seed = args.get("seed", 42)
    dataset_type = args.get("dataset")

    if dataset_type not in ("aero_dataset", "cavity_dataset"):
            raise SystemExit("Only 'aero_dataset' or 'cavity_dataset' are acceptable dataset types")

    if dataset_type == "aero_dataset" and args.get("input_file") is not None:
        print(f"Using data from {args.get('input_file')}")
        dataset = AeroDataset(input_path=args.get("input_file"), target_path=args.get("target_file"), index_path=args.get("index_file"))
        # Split by geometry, not by row. Each geometry spans many rows (one per
        # operating condition); a row-level split leaks the same wing into both
        # train and val, so val_loss measures interpolation, not generalization.
        train_dataset, val_dataset = split_by_group_id(dataset, dataset.geometry_ids(), args.get("val_split"), seed, "Geometry")

    elif dataset_type == "cavity_dataset" and args.get("input_file") is not None:
        print(f"Using cavity next-step data from {args.get('input_file')}")
        dataset = CavityDataset(input_path=args.get("input_file"))
        # Split by case, not by pair. Consecutive frames of one simulation are
        # highly correlated; a pair-level split leaks a case into both train and
        # val, so val_loss would measure interpolation, not generalization.
        train_dataset, val_dataset = split_by_group_id(dataset, dataset.case_ids(), args.get("val_split"), seed, "Case")

    else:
        print("No input and target data provided -> using synthetic dataset.")
        seed = np.random.randint(0, 1e6)
        dataset = SyntheticDataset(n_samples=64, seed=seed)
        n_val = max(1, int(args.get("val_split") * len(dataset)))
        train_dataset, val_dataset = random_split(
            dataset, [len(dataset) - n_val, n_val], generator=torch.Generator().manual_seed(seed))
        
    input_channels = dataset.input_channels
    output_channels = dataset.output_channels

    # ----------------------------------------------------------------
    # AMR Model
    # ----------------------------------------------------------------
    model_type = args.get("model_type", "amr")

    if model_type == "amr":
        print("\n======== Model (AMR) ========")
        refinement_mode = args.get("refinement_mode")
        learned_training_mode = args.get("learned_training_mode")

        if refinement_mode not in ("deterministic", "learned"):
            raise SystemExit("Only 'deterministic' or 'learned' are acceptable refinement modes")
        
        if refinement_mode == "deterministic" and learned_training_mode:
            raise SystemExit("Deterministic mesh is incompatible with --learned_training_mode (no scorer to train)")
        
        if learned_training_mode and learned_training_mode not in ("scorer", "transformer"):
            raise SystemExit("Only 'scorer' or 'transformer' are acceptable learned training modes")

        # Patch-size bounds -> Quadtree depth bounds
        H, W = dataset.H, dataset.W
        min_patch_size = args.get("min_patch_size")
        max_patch_size = args.get("max_patch_size")
        min_depth, max_depth = patch_sizes_to_depth_bounds(H, W, min_patch_size, max_patch_size)
        args["min_depth"] = min_depth
        args["max_depth"] = max_depth
        print(f"Patch-size bounds -> depth bounds:\nmin_patch_size={min_patch_size}, "
            f"max_patch_size={max_patch_size} -> min_depth={min_depth}, max_depth={max_depth} "
            f"(grid {H}x{W})")
        
        # Model
        if args.get("learned_training_mode") == "scorer":
            model = RefinementNet(input_channels=input_channels).to(device)
            print(f"Scorer parameters: {sum(p.numel() for p in model.parameters()):,}")
        else:
            model = AdaptiveMeshAeroModel(
                input_channels=input_channels,
                output_channels=output_channels,
                d_model=args.get("d_model"),
                n_layers=args.get("n_layers"),
                n_heads=args.get("n_heads"),
                d_ff=args.get("d_ff"),
                dropout=args.get("dropout"),
                affine_output=args.get("affine_output", False),
            )
            print(f"Model parameters: {model.count_parameters():,}")

        collate_fn = build_collate_fn(args, train_dataset, input_channels, device)

        # Train
        print("\n======== Training ========")
        train_loader = make_loader(train_dataset, args.get("batch_size"), True, args.get("num_workers"), collate_fn, device)
        val_loader = make_loader(val_dataset, args.get("batch_size"), False, args.get("num_workers"), collate_fn, device)

        train_loss_history, val_loss_history = train_model(args, model, train_loader, val_loader, device, writer)

    # ----------------------------------------------------------------
    # ViT baseline branch (standalone, no AMR / scorer / quadtree)
    # ----------------------------------------------------------------
    elif model_type == "vit":
        H, W = dataset.H, dataset.W
        patch_size = args.get("min_patch_size")
        n_hidden = args.get("n_hidden")
        n_head = args.get("n_head")
        pos_embedding = args.get("pos_embedding", "sincos")

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
            n_layers=args.get("n_layers"),
            n_hidden=n_hidden,
            n_head=n_head,
            mlp_ratio=args.get("mlp_ratio"),
            dropout=args.get("dropout"),
            pos_embedding=pos_embedding)
        print(f"ViT parameters: {sum(p.numel() for p in model.parameters()):,}")

        collate_fn = build_collate_fn(args, train_dataset, input_channels, device)

        print("\n======== Training ========")
        train_loader = make_loader(train_dataset, args.get("batch_size"), True, args.get("num_workers"), collate_fn, device)
        val_loader = make_loader(val_dataset, args.get("batch_size"), False, args.get("num_workers"), collate_fn, device)

        train_loss_history, val_loss_history = train_model(args, model, train_loader, val_loader, device, writer)

    else:
        raise SystemExit("Only 'amr' or 'vit' are acceptable model types")

    plot_loss_curves(train_loss_history, val_loss_history, args.get("epochs"), save_path=f"outputs/loss/{config_name}_config_loss.png")
    writer.close()

if __name__ == "__main__":
    # When calling the function from bash
    if len(sys.argv) > 1:
        print(sys.argv)
        main()

    # When calling the function from IDE
    else:
        config = "configs/overfit.yaml"
        args = ["--config", config]
        print(args)
        main(args)
