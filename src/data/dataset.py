"""
========================================================================
Real-data CFD dataset for the Adaptive Mesh CFD pipeline.
========================================================================

Loads real aerodynamic data from .npz or .npy files
All dataset classes return samples as:
    {
        "input": np.ndarray [H, W, C], 
        "target": np.ndarray [H, W, output_channels]
    }

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

        'input'  : [H, W, C] input channels (position, AoA, Mach, …)
        'target' : [H, W, output_channels] ground-truth CFD quantities (u, v, p, …)

    The key names are configurable via `input_key` / `target_key` so you can
    adapt to whatever naming your CFD solver uses.


    Data format helper
    ------------------
    Each .npz file must contain two arrays named 'input' and 'target'.
    If your data is not yet in .npz format, see 'save_sample_npz()'.
    For a single .npy file the array is treated as inputs; targets must
    be provided separately (see save_sample_npz).

    Args:
        input_path  : path to a single .npz file (contains both arrays) or a .npy
                    file (requires target_path).
        target_path : path to a single .npy file with targets. Required when
                    input_path points to a .npy file (ignored for .npz)
        index_path  : path to a single .npy file which defines the operating 
                    conditions
    """

    def __init__(
        self,
        index_path: str,
        input_path: str,
        target_path: Optional[str] = None,
    ):
        path_index = Path(index_path)
        path_input = Path(input_path)

        if not path_index.exists():
            raise FileNotFoundError(f"index_path does not exist: {index_path}")
        
        if not path_input.exists():
            raise FileNotFoundError(f"input_path does not exist: {input_path}")

        # Load arrays
        if path_input.suffix == ".npz":
            if target_path is not None:
                raise ValueError("target_path should not be provided when input_path is a .npz file; the .npz is expected to contain both 'input' and 'target' arrays."
                                 )
            inputs_npz = np.load(path_input)
            if "input" not in inputs_npz or "target" not in inputs_npz:
                raise KeyError(f"'{path_input}' must contain arrays named 'input' and 'target'. Found: {list(inputs_npz.files)}")

            self._inputs = inputs_npz["input"].astype(np.float32)
            self._targets = inputs_npz["target"].astype(np.float32)

        elif path_input.suffix == ".npy":
            if target_path is None:
                raise ValueError("target_path is required when input_path is a .npy file.")
            
            path_target = Path(target_path)
            if not path_target.exists():
                raise FileNotFoundError(f"target_path does not exist: {target_path}")            

            self._inputs = np.load(path_input).astype(np.float32)
            self._targets = np.load(path_target).astype(np.float32)

            # Detect channel-first (N, C, W, H) and transpose to (N, H, W, C)
            if self._inputs.shape[1] < self._inputs.shape[2] and self._inputs.shape[1] < self._inputs.shape[3]:
                self._inputs = self._inputs.transpose(0, 3, 2, 1)
            if self._targets.shape[1] < self._targets.shape[2] and self._targets.shape[1] < self._targets.shape[3]:
                self._targets = self._targets.transpose(0, 3, 2, 1)

        else:
            raise ValueError(f"Unsupported file extension: {path_input.suffix}. Use .npz or .npy")

        # Validate shapes — expect [N, H, W, C] (channels last)
        if self._inputs.ndim != 4:
            raise ValueError(f"Inputs must be 4-D [N, H, W, C], got shape {self._inputs.shape}.")

        if self._targets.ndim != 4:
            raise ValueError(f"Targets must be 4-D [N, H, W, output_channels], got shape {self._targets.shape}.")
        
        if self._inputs.shape[0] != self._targets.shape[0]:
            raise ValueError(f"Inputs and targets must have the same number of samples, got {self._inputs.shape[0]} vs {self._targets.shape[0]}.")
        
        
        N, H, W, _ = self._inputs.shape
        # Add operating conditions (angle of attack, Mach number) as inputs channels
        # The columns of the index file are defined in https://huggingface.co/datasets/yunplus/SuperWing
        self._index = np.load(path_index)
        angle_of_attack = self._index[:, 2]
        mach_number = self._index[:, 3]
        aoa_channel  = angle_of_attack.reshape(N, 1, 1, 1) * np.ones((N, H, W, 1), dtype=np.float32)
        mach_channel = mach_number.reshape(N, 1, 1, 1) * np.ones((N, H, W, 1), dtype=np.float32)
        self._inputs = np.concatenate([self._inputs, aoa_channel, mach_channel], axis=-1)    # (N, H, W, C+2)

        # Expose dataset metadata
        self.H, self.W = H, W
        self.input_channels = self._inputs.shape[3]
        self.output_channels = self._targets.shape[3]

        print(
            f"AeroDataset: {len(self)} samples  |  "
            f"grid {self.H}x{self.W}  |  "
            f"input_channels={self.input_channels}  output_channels={self.output_channels}"
        )

    def __len__(self) -> int:
        return len(self._inputs)

    def __getitem__(self, index: int) -> Dict:
        return {
            "input":  self._inputs[index],
            "target": self._targets[index],
        }



