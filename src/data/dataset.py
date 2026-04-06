"""
========================================================================
Real-data CFD dataset for the Adaptive Mesh CFD pipeline.
========================================================================

Contents
--------
CFDDataset      - loads real aerodynamic data from .npz or .npy files
save_sample_npz - helper to convert arbitrary arrays into the expected .npz format

All dataset classes return samples as:
    {"input": np.ndarray [H, W, C], "target": np.ndarray [H, W, output_dim]}

For synthetic data, see src.data.synthetic_dataset.
For the DataLoader collate function, see src.data.collate_fn.
"""


from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class CFDDataset(Dataset):
    """
    Generic CFD dataset for real aerodynamic data.

    Supported file layouts
    ----------------------
    Each sample is a single .npz file containing two arrays:

        'input'  : [H, W, C]          - input channels (position, AoA, Mach, …)
        'target' : [H, W, output_dim] - ground-truth CFD quantities (u, v, p, …)

    The key names are configurable via `input_key` / `target_key` so you can
    adapt to whatever naming your CFD solver uses.


    Data format helper
    ------------------
    If your data is not yet in .npz format, see `save_sample_npz()` at the
    bottom of this file for a one-liner conversion from numpy arrays.

    Parameters
    ----------
    data_dir    : path to a directory containing .npz files (mutually exclusive with `file_list`)
    file_list   : explicit list of Path / str file paths (mutually exclusive with `data_dir`)
    val_split   : fraction of files to reserve for validation when no
                  pre-split directories exist. Only used when building the
                  dataset from a flat directory. Pass None to disable.
    split       : 'train' or 'val' - which portion to return when val_split
                  is active. Ignored when val_split is None.
    seed        : random seed for the train/val split shuffle
    normalise   : if True, z-score normalise inputs and targets using
                  statistics computed over the entire dataset (once, at init).
                  Useful if your channels have very different scales.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        file_path: Optional[str] = None,
        val_split: float = 0.1,
        split: str = "train",
        seed: int = 42,
    ):
        """
        Parameters
        ----------
        data_dir  : directory of .npz files, one per sample. The dataset is
                    split into train/val according to val_split and split.
        file_path : path to a single .npy or .npz file whose first axis is
                    the sample dimension - the file is split the same way.
        val_split : fraction of samples held out for validation (default 0.1)
        split     : 'train' or 'val'
        seed      : random seed for the split shuffle

        Each .npz file must contain two arrays named 'input' and 'target'.
        For a single .npy file the array is treated as inputs; targets must
        be provided separately (see save_sample_npz).
        """
        assert (data_dir is None) != (file_path is None), \
            "Provide exactly one of data_dir or file_path."

        rng = np.random.default_rng(seed)

        if file_path is not None:
            # ---- single file: first axis = samples ----
            p = Path(file_path)
            assert p.exists(), f"file_path does not exist: {file_path}"
            if p.suffix == ".npz":
                archive = np.load(p)
                all_inputs  = archive["input"].astype(np.float32)
                all_targets = archive["target"].astype(np.float32)
            elif p.suffix == ".npy":
                arr = np.load(p).astype(np.float32)
                # Treat the array as inputs; build dummy targets of zeros
                # (user should provide a proper paired file)
                arr = arr.transpose(0, 3, 2, 1) # This is needed if input shape is not (B, H, W, C)
                all_inputs  = arr
                all_targets = np.zeros_like(arr[..., :1])
                print(f"Warning: .npy file loaded as inputs only. Targets are set to zero. Use a paired .npz for real training.")
            else:
                raise ValueError(f"Unsupported file extension: {p.suffix}. Use .npz or .npy")

            assert all_inputs.ndim == 4, \
                f"Single-file input must be [N, H, W, C], got {all_inputs.shape}"
            assert all_targets.ndim == 4, \
                f"Single-file target must be [N, H, W, output_dim], got {all_targets.shape}"

            N = len(all_inputs)
            indices = rng.permutation(N).tolist()
            n_val   = max(1, int(val_split * N))
            chosen  = indices[:n_val] if split == "val" else indices[n_val:]
            self._inputs  = all_inputs[chosen]
            self._targets = all_targets[chosen]
            self._mode    = "array"

        else:
            # ---- directory of per-sample .npz files ----
            data_path = Path(data_dir)
            assert data_path.exists(), f"data_dir does not exist: {data_dir}"
            all_files = sorted(data_path.glob("*.npz"))
            assert len(all_files) > 0, (
                f"No .npz files found in {data_dir}.\n"
                f"Each sample must be a .npz file with keys 'input' and 'target'.\n"
                f"See save_sample_npz() in train.py for a one-liner converter."
            )
            indices = rng.permutation(len(all_files)).tolist()
            n_val   = max(1, int(val_split * len(all_files)))
            chosen  = indices[:n_val] if split == "val" else indices[n_val:]
            self.files = [all_files[i] for i in chosen]
            self._mode = "files"

        assert (self._mode == "array" and len(self._inputs) > 0) or \
               (self._mode == "files" and len(self.files) > 0), \
            f"Split '{split}' resulted in 0 samples. Check val_split."

        # Infer dimensions from the first sample
        s0 = self[0]
        inp0, tgt0 = s0["input"], s0["target"]
        assert inp0.ndim == 3, f"Expected [H, W, C], got {inp0.shape}"
        assert tgt0.ndim == 3, f"Expected [H, W, output_dim], got {tgt0.shape}"
        self.H, self.W, self.input_channels = inp0.shape
        self.output_dim = tgt0.shape[-1]

        print(
            f"CFDDataset ({split}): {len(self)} samples  |  "
            f"grid {self.H}×{self.W}  |  "
            f"input_channels={self.input_channels}  output_dim={self.output_dim}"
        )

    def __len__(self) -> int:
        if self._mode == "array":
            return len(self._inputs)
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        if self._mode == "array":
            return {
                "input":  self._inputs[idx],
                "target": self._targets[idx],
            }
        data = np.load(self.files[idx])
        return {
            "input":  data["input"].astype(np.float32),
            "target": data["target"].astype(np.float32),
        }
    


# ---------------------------------------------------------------------------
# Data format helper
# ---------------------------------------------------------------------------

def save_sample_npz(
    input_array: np.ndarray,
    target_array: np.ndarray,
    path: str,
) -> None:
    """
    Save one CFD sample as a .npz file compatible with CFDDataset.

    Parameters
    ----------
    input_array  : [H, W, C]            numpy float32 array
    target_array : [H, W, output_dim]   numpy float32 array
    path         : output file path     ("data/train/sample_0000.npz")

    Example
    -------
        # Convert from your own format:
        inp = np.stack([x_grid, y_grid, aoa_grid, mach_grid], axis=-1)  # [H, W, 4]
        tgt = np.stack([u, v, p], axis=-1)                               # [H, W, 3]
        save_sample_npz(inp, tgt, "data/train/sample_0000.npz")
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, input=input_array, target=target_array)
    print(f"Saved {path}  (input={input_array.shape}, target={target_array.shape})")



