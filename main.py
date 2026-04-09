import argparse
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.amr.configs import AERODYNAMIC_CRITERIA_2
from src.amr.quadtree_tokenizer import QuadtreeTokenizer
from src.data.collate_fn import CollateFn
from src.data.dataset import AeroDataset
from src.data.synthetic_dataset import SyntheticDataset
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
    # Tokenizer & collate
    # ----------------------------------------------------------------
    tokenizer = QuadtreeTokenizer(
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        refinement_criteria=AERODYNAMIC_CRITERIA_2,
    )
    
    collate_fn = CollateFn(tokenizer)

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
    visualize_quadtree_mesh(inp0, tok_list, save_path="outputs/plots")
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
        checkpoint_dir="outputs/checkpoints",
    )

    # ----------------------------------------------------------------
    # Visualise predictions on first sample after training
    # ----------------------------------------------------------------
    inp0_t = torch.from_numpy(inp0).to(device)
    result = model.forward_single(inp0_t)
    pred0  = result["prediction"].cpu().numpy()
    gt0    = first_sample["target"]
    plot_flow_comparison(gt0, pred0, save_path="outputs/plots")
    print("Saved prediction_comparison.png")


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