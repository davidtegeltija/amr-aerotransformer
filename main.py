import argparse
from pathlib import Path
import sys
from typing import Dict
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

from src.amr.refinement_criteria import CRITERIA_REGISTRY
from src.amr.quadtree_tokenizer import QuadtreeTokenizer
from src.amr.oracle_depth import calibrate_global_tolerance, max_reachable_depth
from src.data.collate_fn import DeterministicCollateFn, LearnedCollateFn, ScorerCollateFn
from src.data.dataset import AeroDataset
from src.data.synthetic_dataset import SyntheticDataset
from src.model.amr_model import AdaptiveMeshAeroModel
from src.model.vit import ViT
from src.train import train_deterministic_mesh, train_scorer_supervised, train_learned_mesh_p2, train_vit
from src.utils.data_utils import geometry_disjoint_split
from src.utils.train_utils import plot_loss_curves


def load_config(path: str) -> Dict:
    """Load a YAML config and return a flat namespace mimicking the argparse args."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return cfg


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
    # Build Dataset
    # ----------------------------------------------------------------
    print("\n======== Building Dataset ========")
    if args.get("input_file") is not None: 
        print(f"Using data from {args.get('input_file')}")
        dataset = AeroDataset(input_path=args.get("input_file"), target_path=args.get("target_file"), index_path=args.get("index_file"))
        input_channels = dataset.input_channels
        output_channels = dataset.output_channels

        # Split by geometry, not by row. Each geometry spans many rows (one per
        # operating condition); a row-level split leaks the same wing into both
        # train and val, so val_loss measures interpolation, not generalization.
        seed = args.get("seed", 42)
        try:
            train_idx, val_idx = geometry_disjoint_split(
                dataset.geometry_ids(), args.get("val_split"), seed
            )
            train_dataset, val_dataset = Subset(dataset, train_idx), Subset(dataset, val_idx)
            print(f"Geometry-disjoint split (seed={seed}): "
                  f"{len(train_idx)} train rows / {len(val_idx)} val rows")
        except ValueError as e:
            print(f"WARNING: {e}")
            print("Falling back to validating on the full training set. "
                  "This is an overfit sanity check only — val_loss is NOT a "
                  "generalization metric when train and val share geometries.")
            train_dataset = val_dataset = dataset
    else:
        print("No input and target data provided -> using synthetic dataset.")
        seed = np.random.randint(0, 1e6)
        dataset = SyntheticDataset(n_samples=64, seed=seed)
        input_channels = dataset.input_channels
        output_channels = dataset.output_channels
        n_val = max(1, int(args.get("val_split") * len(dataset)))
        train_dataset, val_dataset = random_split(dataset, [len(dataset) - n_val, n_val], generator=torch.Generator().manual_seed(seed),)

    # ----------------------------------------------------------------
    # ViT baseline branch (standalone, no AMR / scorer / quadtree)
    # ----------------------------------------------------------------
    if args.get("model_type") == "vit":
        print("\n======== Model (ViT baseline) ========")
        H, W = args.get("image_size")
        patch_size = args.get("patch_size")
        n_hidden = args.get("n_hidden")
        n_head = args.get("n_head")
        pos_embedding = args.get("pos_embedding", "sincos")
        assert H % patch_size == 0 and W % patch_size == 0, \
            f"image_size {(H, W)} must be divisible by patch_size {patch_size}"
        assert n_hidden % n_head == 0, \
            f"n_hidden {n_hidden} must be divisible by n_head {n_head}"
        assert pos_embedding != "sincos" or n_hidden % 4 == 0, \
            f"n_hidden {n_hidden} must be divisible by 4 when pos_embedding='sincos'"

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
            pos_embedding=pos_embedding,
        ).to(device)
        print(f"ViT parameters: {sum(p.numel() for p in model.parameters()):,}")

        collate_fn = LearnedCollateFn()
        train_loader = DataLoader(train_dataset, batch_size=args.get("batch_size"),
                                  num_workers=args.get("num_workers"), shuffle=True,
                                  collate_fn=collate_fn,
                                  pin_memory=device.type == "cuda")
        val_loader = DataLoader(val_dataset, batch_size=args.get("batch_size"),
                                num_workers=args.get("num_workers"), shuffle=False,
                                collate_fn=collate_fn,
                                pin_memory=device.type == "cuda")

        ckpt_path = args.get("checkpoint_path", "outputs/vit")
        train_loss_history, val_loss_history = train_vit(
            model, train_loader, val_loader, device,
            epochs=args.get("epochs"),
            lr=args.get("lr"),
            weight_decay=args.get("weight_decay"),
            grad_clip=args.get("grad_clip"),
            save_path=f"{ckpt_path}/best.pt",
        )

        config_name = Path(cli.config).stem
        plot_loss_curves(train_loss_history, val_loss_history, args.get("epochs"),
                         save_path=f"outputs/loss/{config_name}_config_loss.png")
        return

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    print("\n======== Model ========")
    if args.get("refinement_mode") not in ["deterministic", "learned"]:
        raise SystemExit("Only 'deterministic' or 'learned' are acceptable refinement modes")
    
    if args.get("refinement_mode") == "deterministic" and args.get("learned_training_mode"):
        raise SystemExit("Deterministic mesh is incompatible with --learned_training_mode (no scorer to train)")
    
    if args.get("learned_training_mode") and args.get("learned_training_mode") not in ["scorer", "fine-tune"]:
        raise SystemExit("Only 'learned_training' or 'fine-tune' are acceptable learned training modes")
    
    if args.get("refinement_mode") == "deterministic":
        if args.get("refinement_criteria") not in CRITERIA_REGISTRY:
            valid = ", ".join(sorted(CRITERIA_REGISTRY))
            raise KeyError(f"Unknown --refinement_criteria {args.get('refinement_criteria')!r}.\nAvailable options are: {valid}")

        criteria = CRITERIA_REGISTRY[args.get("refinement_criteria")]
        
        tokenizer = QuadtreeTokenizer(
            min_depth=args.get("min_depth"),
            max_depth=args.get("max_depth"),
            refinement_criteria=criteria,
        )
        collate_fn = DeterministicCollateFn(tokenizer)
    else:
        criteria = None

        if args.get("learned_training_mode") == "scorer":
            # Supervised scorer training: calibrate ONE global tolerance on a subset
            # of the train split so the mean oracle leaf count lands near n_target.
            # A single global tol keeps the error scale fixed and lets per-sample
            # counts vary with geometry complexity.
            min_depth = args.get("min_depth")
            max_depth = args.get("max_depth")
            min_cell_size = args.get("min_cell_size", 4)
            n_target = args.get("n_target", 256)
            n_calib = min(args.get("calib_samples", 64), len(train_dataset))
            calib_targets = [np.asarray(train_dataset[i]["target"], dtype=np.float32)
                            for i in range(n_calib)]
            tol = calibrate_global_tolerance(
                calib_targets, n_target=n_target,
                min_depth=min_depth, max_depth=max_depth, min_cell_size=min_cell_size,
            )
            reachable = max_reachable_depth(
                calib_targets[0].shape[0], calib_targets[0].shape[1],
                min_cell_size, max_depth,
            )
            print(f"Oracle target: global tol={tol:.4g} (n_target={n_target}, "
                f"calib n={n_calib}, reachable_depth={reachable})")
            collate_fn = ScorerCollateFn(
                tol=tol, min_depth=min_depth, max_depth=max_depth,
                min_cell_size=min_cell_size,
            )
        elif args.get("learned_training_mode") == "fine-tune":
            collate_fn = LearnedCollateFn() # Collate (tokenization now happens inside the model)

    model = AdaptiveMeshAeroModel(
        input_channels=input_channels,
        output_channels=output_channels,
        d_model=args.get("d_model"),
        n_layers=args.get("n_layers"),
        n_heads=args.get("n_heads"),
        d_ff=args.get("d_ff"),
        dropout=args.get("dropout"),
        min_depth=args.get("min_depth"),
        max_depth=args.get("max_depth"),
        min_cell_size=args.get("min_cell_size", 4),
        refinement_mode=args.get("refinement_mode"),
        refinement_criteria=criteria,
    )

    print(f"Model parameters: {model.count_parameters():,}")

    # ----------------------------------------------------------------
    # Optional checkpoint load
    # ----------------------------------------------------------------
    if args.get("checkpoint_file") is not None:
        ckpt = torch.load(args.get("checkpoint_file"), map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        # Allow partial loads: a transformer-only checkpoint can warm-start the
        # frozen transformer used by the optional end-to-end sanity metric.
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint {args.get('checkpoint_file')}")
        print(f"  missing keys:    {len(missing)} (expected: scorer.* before scorer training)")
        print(f"  unexpected keys: {len(unexpected)} (should be 0 or near 0)")
    elif args.get("learned_training_mode") == "fine-tune":
        raise SystemExit("--learned_training_mode fine-tune requires --checkpoint pointing to a scorer checkpoint")

    # ----------------------------------------------------------------
    # Train
    # ----------------------------------------------------------------
    print("\n======== Training ========")
    
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.get("batch_size"), shuffle=True,
                              num_workers=args.get("num_workers"), collate_fn=collate_fn,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_dataset, batch_size=args.get("batch_size"), shuffle=False,
                              num_workers=args.get("num_workers"), collate_fn=collate_fn,
                              pin_memory=device.type == "cuda")

    if args.get("refinement_mode") == "deterministic":
        train_loss_history, val_loss_history = train_deterministic_mesh(
            model, train_loader, val_loader, device,
            epochs=args.get("epochs"),
            d_model=args.get("d_model"),
            warmup_steps=args.get("warmup_steps"),
            save_path="outputs/checkpoints",
        )
    else:
        if args.get("learned_training_mode") == "scorer":
            train_loss_history, val_loss_history = train_scorer_supervised(
                model, train_loader, val_loader, device,
                epochs=args.get("epochs"),
                tv_weight=args.get("tv_weight", 0.0),
                decision_weight=args.get("decision_weight", 0.0),
                decision_margin=args.get("decision_margin", 0.0),
                decision_temp=args.get("decision_temp", None),
                end_to_end_every=args.get("end_to_end_every", 0),
                save_path="outputs/checkpoints",
            )
        elif args.get("learned_training_mode") == "fine-tune":
            train_loss_history, val_loss_history = train_learned_mesh_p2(
                model, train_loader, val_loader, device,
                epochs=args.get("epochs"),
                lambda_budget=args.get("lambda_budget"),
                lambda_smooth=args.get("lambda_smooth"),
                tau_start=args.get("tau_start_phase2"),
                tau_end=args.get("tau_end_phase2"),
                scorer_lr=args.get("scorer_lr"),
                transformer_lr=args.get("transformer_lr"),
                n_max=args.get("n_max"),
                save_path="outputs/checkpoints/phase2_joint.pt",
            )

    config_name = Path(cli.config).stem
    plot_loss_curves(train_loss_history, val_loss_history, args.get("epochs"), save_path=f"outputs/loss/{config_name}_config_loss.png")


if __name__ == "__main__":
    # When calling the function from bash
    if len(sys.argv) > 1:
        print(sys.argv)
        main()

    # When calling the function from IDE
    else:            
        config = "configs/baseline.yaml"
        args = ["--config", config]
        print(args)
        main(args)