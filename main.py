import argparse
import os
import sys

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

def main(args=None):
    parser = argparse.ArgumentParser(
        description="Train AdaptiveMeshAeroModel",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
        Examples
        --------
        # No data provided -> synthetic field (smoke-test):
        python train.py --epochs 5

        # Single .npz or .npy file whose first axis is the sample dimension:
        python train.py --input_file /data/airfoil_dataset.npz --target_file /data/ground_truth.npy --epochs 200 --val_split 0.15
        """,
    )

    # ---- Data ----
    parser.add_argument("--input_file", type=str, default=None,
                        help="Single .npz or .npy file with shape [N, H, W, C] representing the geometry of the problem.")
    parser.add_argument("--target_file", type=str, default=None,
                        help="Single .npz or .npy file with shape [M, H, W, C] representing the steady flow state.")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of samples held out for validation (default: 0.1)")

    # ---- Model ----
    parser.add_argument("--d_model",   type=int, default=256)
    parser.add_argument("--n_layers",  type=int, default=6)
    parser.add_argument("--n_heads",   type=int, default=4)
    parser.add_argument("--d_ff",      type=int, default=1024)
    parser.add_argument("--min_depth", type=int, default=2,
                        help="Quadtree minimum subdivision depth (default: 2)")
    parser.add_argument("--max_depth", type=int, default=6,
                        help="Quadtree maximum subdivision depth (default: 6)")
    parser.add_argument("--refinement_mode", choices=["learned", "deterministic"], default="deterministic",
                        help="Mesh-refinement policy. 'learned' uses the RefinementNet "
                            "scorer + build_learned_adaptive_mesh. 'deterministic' (default) uses physics/"
                            "geometry thresholds via build_adaptive_mesh and --refinement_criteria.")
    parser.add_argument("--refinement_criteria", type=str, default="AERODYNAMIC_CRITERIA_2",
                        help="Name of preset in src/amr/configs.py (see CRITERIA_REGISTRY). "
                            "Used only when --refinement_mode deterministic.")

    # ---- Training ----
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--warmup_steps", type=int,   default=4000)
    parser.add_argument("--num_workers",  type=int,   default=0)
    parser.add_argument("--seed",         type=int,   default=42)

    # ---- Phase 1 / Phase 2 learned-scorer training ----
    parser.add_argument("--phase", type=int, choices=[1, 2], default=None,
                        help="Training phase 1: scorer only (transformer frozen), "
                            "Training phase 2: scorer + transformer")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to a checkpoint to load BEFORE training. For --phase 1 this "
                        "should be a pretrained-transformer checkpoint. For --phase 2 this "
                        "should be a trained-scorer checkpoint from phase 2.")
    parser.add_argument("--phase1_epochs", type=int, default=50)
    parser.add_argument("--phase2_epochs", type=int, default=30)
    parser.add_argument("--lambda_budget", type=float, default=0.01)
    parser.add_argument("--lambda_smooth", type=float, default=0.001)
    parser.add_argument("--tau_start_phase2", type=float, default=5.0)
    parser.add_argument("--tau_end_phase2",   type=float, default=0.5)
    parser.add_argument("--tau_start_phase3", type=float, default=0.5)
    parser.add_argument("--tau_end_phase3",   type=float, default=0.1)
    parser.add_argument("--scorer_lr",      type=float, default=1e-3)
    parser.add_argument("--transformer_lr", type=float, default=1e-4)
    parser.add_argument("--n_max", type=int, default=8192,
                        help="Max possible token count used to normalize L_budget. Default 8192 for the 256x128 grid.")

    args = parser.parse_args(args)

    os.makedirs("outputs", exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ----------------------------------------------------------------
    # Build datasets
    # ----------------------------------------------------------------
    print("\n====== Building Dataset ======")
    if args.input_file is not None: 
        print(f"Using data from {args.input_file}")
        dataset = AeroDataset(input_path=args.input_file, target_path=args.target_file)
        input_channels = dataset.input_channels
        output_dim     = dataset.output_dim
        first_sample   = dataset[0]
        n_val = max(1, int(args.val_split * len(dataset)))
        train_dataset, val_dataset = random_split(dataset, [len(dataset) - n_val, n_val])
    else:
        print("No input and target data provided -> using synthetic dataset.")
        dataset = SyntheticDataset(n_samples=64, seed=args.seed)
        input_channels = dataset.input_channels
        output_dim     = dataset.output_dim
        first_sample   = dataset[0]

        n_val   = max(1, int(args.val_split * len(dataset)))
        train_dataset, val_dataset = random_split(dataset, [len(dataset) - n_val, n_val], generator=torch.Generator().manual_seed(args.seed),)
    # ----------------------------------------------------------------
    # Collate (tokenization now happens inside the model)
    # ----------------------------------------------------------------
    if  args.refinement_mode == "deterministic":
        tokenizer = QuadtreeTokenizer(
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            refinement_criteria=CRITERIA_REGISTRY[args.refinement_criteria],
        )
        collate_fn = DeterministicCollateFn(tokenizer)
    else:
        collate_fn = LearnedCollateFn()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=device.type == "cuda")

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    print("\n====== Model ======")
    if args.refinement_mode == "deterministic":
        try:
            criteria = CRITERIA_REGISTRY[args.refinement_criteria]
        except KeyError:
            valid = ", ".join(sorted(CRITERIA_REGISTRY))
            raise SystemExit(
                f"Unknown --refinement_criteria {args.refinement_criteria!r}. "
                f"Valid presets: {valid}"
            )
    else:
        criteria = None

    model = AdaptiveMeshAeroModel(
        input_channels=input_channels,
        output_dim=output_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        refinement_mode=args.refinement_mode,
        refinement_criteria=criteria,
    )
    print(f"Model parameters: {model.count_parameters():,}")

    # ----------------------------------------------------------------
    # Optional checkpoint load
    # ----------------------------------------------------------------
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

        # Allow partial loads: phase 2 starts from a transformer-only checkpoint
        # (no scorer weights yet), so strict=False is the right default here.
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded checkpoint {args.checkpoint}")
        print(f"  missing keys:    {len(missing)} (expected: scorer.* on first phase-2 run)")
        print(f"  unexpected keys: {len(unexpected)} (should be 0 or near 0)")
    elif args.phase == 2:
        print("WARNING: --phase 2 invoked without --checkpoint. "
              "Starting from a RANDOMLY-INITIALIZED transformer - useful for "
              "smoke tests only, not a real phase-2 run.")
    elif args.phase == 3:
        raise SystemExit(
            "--phase 3 requires --checkpoint pointing to a phase-2 scorer "
            "checkpoint (e.g. outputs/phase2_scorer.pt)."
        )

    # ----------------------------------------------------------------
    # Train
    # ----------------------------------------------------------------
    print("\n====== Training ======")
    if args.refinement_mode == "deterministic" and args.phase in {2, 3}:
        raise SystemExit(
            "Deterministic mesh is incompatible with --phase 2/3 "
            "(no scorer to train). Use the legacy train() path by omitting --phase."
        )

    model = model.to(device)

    if args.phase == 1:
        train_learned_mesh_p1(
            model, train_loader, val_loader, device,
            phase2_epochs=args.phase2_epochs,
            lambda_budget=args.lambda_budget,
            lambda_smooth=args.lambda_smooth,
            tau_start=args.tau_start_phase2,
            tau_end=args.tau_end_phase2,
            scorer_lr=args.scorer_lr,
            n_max=args.n_max,
            save_path="outputs/phase2_scorer.pt",
        )
    elif args.phase == 2:
        train_learned_mesh_p2(
            model, train_loader, val_loader, device,
            phase3_epochs=args.phase3_epochs,
            lambda_budget=args.lambda_budget,
            lambda_smooth=args.lambda_smooth,
            tau_start=args.tau_start_phase3,
            tau_end=args.tau_end_phase3,
            scorer_lr=args.scorer_lr,
            transformer_lr=args.transformer_lr,
            n_max=args.n_max,
            save_path="outputs/phase3_joint.pt",
        )
    else:
        train_deterministic_mesh(
            model, train_loader, val_loader,
            epochs=args.epochs,
            device=device,
            d_model=args.d_model,
            warmup_steps=args.warmup_steps,
            checkpoint_path="outputs/checkpoints",
        )


if __name__ == "__main__":
    # When calling the function from bash
    if len(sys.argv) > 1:
        print(sys.argv)
        main()

    # When calling the function from IDE
    else:            
        input_file = "data/crmmgeom.npy"
        target_file = "data/crmmdata.npy"
        epochs = "1"
        args = ["--input_file", input_file,
                "--target_file", target_file,
                "--epochs", epochs]
        print(args)
        main(args)