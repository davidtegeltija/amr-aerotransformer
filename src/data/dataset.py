"""
========================================================================
Real-data CFD dataset for the Adaptive Mesh CFD pipeline.
========================================================================

Contents
--------
AeroDataset      - loads real aerodynamic data from .npz or .npy files
save_sample_npz - helper to convert arbitrary arrays into the expected .npz format

All dataset classes return samples as:
    {"input": np.ndarray [H, W, C], "target": np.ndarray [H, W, output_dim]}

For synthetic data, see src.data.synthetic_dataset.
For the DataLoader collate function, see src.data.collate_fn.
"""


from pathlib import Path
from typing import Dict, Optional

import numpy as np
from torch.utils.data import Dataset


class AeroDataset(Dataset):
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
    input_path  : path to a single .npz file (contains both arrays) or a .npy
                  file (requires target_path).
    target_path : path to a single .npy file with targets. Required when
                  input_path points to a .npy file (ignored for .npz)

    Each .npz file must contain two arrays named 'input' and 'target'.
    For a single .npy file the array is treated as inputs; targets must
    be provided separately (see save_sample_npz).
    """

    def __init__(
        self,
        input_path: str,
        target_path: Optional[str] = None,
    ):

        path_i = Path(input_path)
        if not path_i.exists():
            raise FileNotFoundError(f"input_path does not exist: {input_path}")

        # Load arrays
        if path_i.suffix == ".npz":
            if target_path is not None:
                raise ValueError("target_path should not be provided when input_path is a .npz file; the .npz is expected to contain both 'input' and 'target' arrays."
                                 )
            inputs_npz = np.load(path_i)
            if "input" not in inputs_npz or "target" not in inputs_npz:
                raise KeyError(f"'{path_i}' must contain arrays named 'input' and 'target'. Found: {list(inputs_npz.files)}")
                     
            self._inputs  = inputs_npz["input"].astype(np.float32)
            self._targets = inputs_npz["target"].astype(np.float32)

        elif path_i.suffix == ".npy":
            if target_path is None:
                raise ValueError("target_path is required when input_path is a .npy file.")
            
            path_t = Path(target_path)
            if not path_t.exists():
                raise FileNotFoundError(f"target_path does not exist: {target_path}")            

            self._inputs = np.load(path_i).astype(np.float32)
            self._targets = np.load(path_t).astype(np.float32)

            # Detect channel-first (N, C, H, W) and transpose to (N, H, W, C)
            if self._inputs.shape[1] < self._inputs.shape[2] and self._inputs.shape[1] < self._inputs.shape[3]:
                self._inputs = self._inputs.transpose(0, 3, 2, 1)
            if self._targets.shape[1] < self._targets.shape[2] and self._targets.shape[1] < self._targets.shape[3]:
                self._targets = self._targets.transpose(0, 3, 2, 1)

        else:
            raise ValueError(f"Unsupported file extension: {path_i.suffix}. Use .npz or .npy")

        # Validate shapes — expect [N, H, W, C] (channels last)
        if self._inputs.ndim != 4:
            raise ValueError(f"Inputs must be 4-D [N, H, W, C], got shape {self._inputs.shape}.")

        if self._targets.ndim != 4:
            raise ValueError(f"Targets must be 4-D [N, H, W, output_dim], got shape {self._targets.shape}.")
        
        if self._inputs.shape[0] != self._targets.shape[0]:
            raise ValueError(f"Inputs and targets must have the same number of samples, got {self._inputs.shape[0]} vs {self._targets.shape[0]}.")

        # Expose dataset metadata
        self.H, self.W      = self._inputs.shape[1], self._inputs.shape[2]
        self.input_channels = self._inputs.shape[3]
        self.output_dim     = self._targets.shape[3]

        print(
            f"AeroDataset: {len(self)} samples  |  "
            f"grid {self.H}x{self.W}  |  "
            f"input_channels={self.input_channels}  output_dim={self.output_dim}"
        )

    def __len__(self) -> int:
        return len(self._inputs)

    def __getitem__(self, index: int) -> Dict:
        return {
            "input":  self._inputs[index],
            "target": self._targets[index],
        }



