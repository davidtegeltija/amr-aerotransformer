import argparse
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.amr.configs import AERODYNAMIC_CRITERIA
from src.amr.quadtree_tokenizer import QuadtreeTokenizer
from src.data.collate_fn import CollateFn
from src.data.dataset import CFDDataset
from src.data.synthetic_dataset import SyntheticCFDDataset
from src.model.amr_model import AdaptiveMeshAeroModel
from src.train import train
from src.utils.flow_visualization import plot_flow_comparison
from src.utils.mesh_visualization import visualize_quadtree_mesh

def main(args=None):
    parser = argparse.ArgumentParser(
        description="Train AdaptiveMeshAeroModel",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
        Examples
        --------
        # No data provided → synthetic field (smoke-test):
        python train.py --epochs 5

        # Directory of per-sample .npz files (auto 90/10 split):
        python train.py --data_dir /data/airfoil/ --epochs 200

        # Single .npz or .npy file whose first axis is the sample dimension:
        python train.py --data_file /data/airfoil_dataset.npz --epochs 200

        # Custom validation fraction:
        python train.py --data_dir /data/airfoil/ --val_split 0.15
        """,
    )

    # ---- Data ----
    parser.add_argument("--data_dir",  type=str, default=None,
                        help="Directory of per-sample .npz files  (keys: 'input', 'target')")
    parser.add_argument("--data_file", type=str, default=None,
                        help="Single .npz or .npy file with shape [N, H, W, C]")
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

    # ---- Training ----
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--warmup_steps", type=int,   default=4000)
    parser.add_argument("--num_workers",  type=int,   default=0)
    parser.add_argument("--seed",         type=int,   default=42)

    args = parser.parse_args(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ----------------------------------------------------------------
    # Build datasets
    # ----------------------------------------------------------------
    print("\n====== Building Dataset ======")
    if args.data_dir is not None or args.data_file is not None:
        source = {"data_dir": args.data_dir} if args.data_dir else {"file_path": args.data_file}
        train_ds = CFDDataset(**source, val_split=args.val_split, split="train", seed=args.seed)
        val_ds   = CFDDataset(**source, val_split=args.val_split, split="val",   seed=args.seed)
        input_channels = train_ds.input_channels
        output_dim     = train_ds.output_dim
        first_sample   = train_ds[0]
        print(f"Using data from {source}")
    else:
        print("No --data_dir or --data_file provided -> using synthetic dataset.")
        full_ds = SyntheticCFDDataset(n_samples=64, seed=args.seed)
        n_val   = max(1, int(args.val_split * len(full_ds)))
        train_ds, val_ds = torch.utils.data.random_split(
            full_ds, [len(full_ds) - n_val, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )
        input_channels = full_ds.input_channels
        output_dim     = full_ds.output_dim
        first_sample   = full_ds[0]

    # ----------------------------------------------------------------
    # Tokenizer & collate
    # ----------------------------------------------------------------
    tokenizer = QuadtreeTokenizer(
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        refinement_criteria=AERODYNAMIC_CRITERIA,
    )
    collate_fn = CollateFn(tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn,
                              pin_memory=device.type == "cuda")

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    print("\n====== Model ======")
    model = AdaptiveMeshAeroModel(
        input_channels=input_channels,
        output_dim=output_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
    )
    print(f"Model parameters: {model.count_parameters():,}")

    # ----------------------------------------------------------------
    # Visualise mesh on first sample before training
    # ----------------------------------------------------------------
    inp0 = first_sample["input"]
    tok_arr, tok_list = tokenizer.tokenize(inp0)
    print(f"Tokens for first sample: {len(tok_list)}")
    visualize_quadtree_mesh(inp0, tok_list, save_path="mesh_visualisation.png")
    print("Saved mesh_visualisation.png")

    # ----------------------------------------------------------------
    # Train
    # ----------------------------------------------------------------
    print("\n====== Training ======")
    train(
        model, train_loader, val_loader,
        epochs=args.epochs,
        device=device,
        d_model=args.d_model,
        warmup_steps=args.warmup_steps,
        checkpoint_dir="checkpoints",
    )

    # ----------------------------------------------------------------
    # Visualise predictions on first sample after training
    # ----------------------------------------------------------------
    inp0_t = torch.from_numpy(inp0).to(device)
    result = model.forward_single(inp0_t)
    pred0  = result["prediction"].cpu().numpy()
    gt0    = first_sample["target"]
    plot_flow_comparison(gt0, pred0, save_path="prediction_comparison.png")
    print("Saved prediction_comparison.png")


if __name__ == "__main__":
    # When calling the function from bash
    if len(sys.argv) > 1:
        print(sys.argv)
        main()

    # When calling the function from IDE
    else:            
        data_file = "data/crmmdata.npy"
        epochs = "1"
        args = ["--data_file", data_file,
                "--epochs", epochs]
        print(args)
        main(args)