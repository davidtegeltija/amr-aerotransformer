import argparse
import os
import sys
from typing import Dict
import yaml

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.amr.configs import CRITERIA_REGISTRY
from src.amr.quadtree_tokenizer import QuadtreeTokenizer
from src.data.collate_fn import DeterministicCollateFn, LearnedCollateFn
from src.data.dataset import AeroDataset
from src.data.synthetic_dataset import SyntheticDataset
from src.model.amr_model import AdaptiveMeshAeroModel
from src.train import train_deterministic_mesh, train_learned_mesh_p1, train_learned_mesh_p2


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

    os.makedirs("outputs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ----------------------------------------------------------------
    # Build Dataset
    # ----------------------------------------------------------------
    print("\n======== Building Dataset ========")
    if args.get("input_file") is not None: 
        print(f"Using data from {args.get("input_file")}")
        dataset = AeroDataset(input_path=args.get("input_file"), target_path=args.get("target_file"))
        input_channels = dataset.input_channels
        output_dim     = dataset.output_dim
        n_val = max(1, int(args.get("val_split") * len(dataset)))
        train_dataset, val_dataset = random_split(dataset, [len(dataset) - n_val, n_val])
    else:
        print("No input and target data provided -> using synthetic dataset.")
        seed = np.random.randint(0, 2**32)
        dataset = SyntheticDataset(n_samples=64, seed=seed)
        input_channels = dataset.input_channels
        output_dim     = dataset.output_dim
        n_val   = max(1, int(args.get("val_split") * len(dataset)))
        train_dataset, val_dataset = random_split(dataset, [len(dataset) - n_val, n_val], generator=torch.Generator().manual_seed(seed),)

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    print("\n======== Model ========")
    if args.get("refinement_mode") not in ["deterministic", "learned"]:
        raise SystemExit("Only 'deterministic' or 'learned' are acceptable refinement modes")
    
    if args.get("refinement_mode") == "deterministic":
        if args.get("refinement_criteria") not in CRITERIA_REGISTRY:
            valid = ", ".join(sorted(CRITERIA_REGISTRY))
            raise KeyError(f"Unknown --refinement_criteria {args.get("refinement_criteria")!r}.\nAvailable options are: {valid}")

        criteria = CRITERIA_REGISTRY[args.get("refinement_criteria")]
        
        tokenizer = QuadtreeTokenizer(
            min_depth=args.get("min_depth"),
            max_depth=args.get("max_depth"),
            refinement_criteria=criteria,
        )
        collate_fn = DeterministicCollateFn(tokenizer)
    else:
        criteria = None
        collate_fn = LearnedCollateFn() # Collate (tokenization now happens inside the model)

    model = AdaptiveMeshAeroModel(
        input_channels=input_channels,
        output_dim=output_dim,
        d_model=args.get("d_model"),
        n_layers=args.get("n_layers"),
        n_heads=args.get("n_heads"),
        d_ff=args.get("d_ff"),
        min_depth=args.get("min_depth"),
        max_depth=args.get("max_depth"),
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

        # Allow partial loads: phase 2 starts from a transformer-only checkpoint
        # (no scorer weights yet), so strict=False is the right default here.
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint {args.get("checkpoint_file")}")
        print(f"  missing keys:    {len(missing)} (expected: scorer.* on first phase-2 run)")
        print(f"  unexpected keys: {len(unexpected)} (should be 0 or near 0)")
    elif args.get("training_phase") == 1:
        print("WARNING: --phase 2 invoked without --checkpoint. "
              "Starting from a RANDOMLY-INITIALIZED transformer - useful for "
              "smoke tests only, not a real phase-2 run.")
    elif args.get("training_phase") == 2:
        raise SystemExit(
            "--phase 3 requires --checkpoint pointing to a phase-2 scorer "
            "checkpoint (e.g. outputs/phase2_scorer.pt)."
        )

    # ----------------------------------------------------------------
    # Train
    # ----------------------------------------------------------------
    print("\n======== Training ========")
    if args.get("training_phase") and args.get("training_phase") not in [1, 2]:
        raise SystemExit("Only 1 or 2 are acceptable training phases.")
    
    if args.get("refinement_mode") == "deterministic" and args.get("training_phase") in [1, 2]:
        raise SystemExit("Deterministic mesh is incompatible with --training_phase (no scorer to train)")
    
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=args.get("batch_size"), shuffle=True,
                              num_workers=args.get("num_workers"), collate_fn=collate_fn,
                              pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_dataset,   batch_size=args.get("batch_size"), shuffle=False,
                              num_workers=args.get("num_workers"), collate_fn=collate_fn,
                              pin_memory=device.type == "cuda")

    if args.get("training_phase") == 1:
        train_learned_mesh_p1(
            model, train_loader, val_loader, device,
            phase2_epochs=args.get("epochs"),
            lambda_budget=args.get("lambda_budget"),
            lambda_smooth=args.get("lambda_smooth"),
            tau_start=args.get("tau_start_phase1"),
            tau_end=args.get("tau_end_phase1"),
            scorer_lr=args.get("scorer_lr"),
            n_max=args.get("n_max"),
            save_path="outputs/checkpoints/phase1_scorer.pt",
        )
    elif args.get("training_phase") == 2:
        train_learned_mesh_p2(
            model, train_loader, val_loader, device,
            phase3_epochs=args.get("epochs"),
            lambda_budget=args.get("lambda_budget"),
            lambda_smooth=args.get("lambda_smooth"),
            tau_start=args.get("tau_start_phase2"),
            tau_end=args.get("tau_end_phase2"),
            scorer_lr=args.get("scorer_lr"),
            transformer_lr=args.get("transformer_lr"),
            n_max=args.get("n_max"),
            save_path="outputs/checkpoints/phase2_joint.pt",
        )
    else:
        train_deterministic_mesh(
            model, train_loader, val_loader, device,
            epochs=args.get("epochs"),
            d_model=args.get("d_model"),
            warmup_steps=args.get("warmup_steps"),
            checkpoint_path="outputs/checkpoints",
        )


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