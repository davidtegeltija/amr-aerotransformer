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


class WingDataset(Dataset):
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


    Geometry row layouts
    --------------------
    Every geometry is simulated at several operating conditions, so the raw
    SuperWing arrays hold one geometry row per *wing* and one target row per
    *simulation*. Both layouts of the input array are accepted:

        one row per geometry   : the raw 'geom0.npy'; row i of the dataset reads
                                 geometry ``index[i, 0]``. Nothing is duplicated,
                                 so the whole dataset fits in memory.
        one row per simulation : the aligned arrays written by
                                 ``create_data_subset``, where the geometry rows
                                 are already repeated to match the targets 1:1.

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

            self._inputs = np.load(path_input, mmap_mode="r").astype(np.float32)
            self._targets = np.load(path_target, mmap_mode="r").astype(np.float32)

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
        
        self._index = np.load(path_index)
        if self._index.shape[0] != self._targets.shape[0]:
            raise ValueError(f"Index and targets must have the same number of rows, got {self._index.shape[0]} vs {self._targets.shape[0]}. The index file describes one simulation per target row, so the two are written together.")

        # Column 0 of the index is the geometry each simulation was run on (see
        # https://huggingface.co/datasets/yunplus/SuperWing). It is a row index
        # into the raw geometry array
        geometry_rows = self._index[:, 0].astype(int)
        n_inputs, H, W, _ = self._inputs.shape

        # One input row per target, so the geometry rows were already
        # repeated to match and every row maps to itself
        if n_inputs == self._targets.shape[0]:
            self._geometry_rows = np.arange(n_inputs)
        # Fewer input rows than targets, so each one is a geometry
        # shared by several simulations
        elif np.all(geometry_rows < n_inputs):
            self._geometry_rows = geometry_rows
        else:
            raise ValueError(f"Inputs hold neither one row per target ({self._targets.shape[0]}) nor one row per geometry: got {n_inputs} rows, but the index refers to geometry {geometry_rows.max()}. Inputs, targets and index must come from the same dataset.")

        # Add operating conditions (angle of attack, Mach number) as inputs channels
        # The columns of the index file are defined in https://huggingface.co/datasets/yunplus/SuperWing
        angle_of_attack = self._index[:, 2]
        mach_number = self._index[:, 3]
        self._conditions = np.column_stack((angle_of_attack, mach_number)).astype(np.float32)

        # Expose dataset metadata. Two input channels are appended per sample in
        # __getitem__, so they are counted here but never materialised for the
        # whole dataset (that alone would be ~19 GB on the full SuperWing set).
        self.H = H
        self.W = W
        self.input_channels = self._inputs.shape[3] + self._conditions.shape[1]
        self.output_channels = self._targets.shape[3]

        print(
            f"WingDataset: {len(self)} samples  |  "
            f"grid {H}x{W}  |  "
            f"input_channels={self.input_channels}  output_channels={self.output_channels}"
        )

    def __len__(self) -> int:
        return len(self._targets)

    def __getitem__(self, index: int) -> Dict:
        geometry = self._inputs[self._geometry_rows[index]]                     # [H, W, C]

        # Angle of attack and Mach number ride along as two constant channels.
        conditions = np.broadcast_to(self._conditions[index], (self.H, self.W, 2))

        return {
            "index":  index,
            "input":  np.concatenate([geometry, conditions], axis=-1),          # [H, W, C+2]
            "target": self._targets[index],
        }

    def geometry_ids(self) -> np.ndarray:
        """Return the geometry id of every dataset row.

        Each geometry (a single wing) is simulated at many operating
        conditions, so it spans several dataset rows that share the same
        geometry id (column 0 of the index file). Use this to build a
        geometry-disjoint train/val split — splitting rows directly leaks a
        geometry across the boundary.

        Returns:
            Integer array of shape [N] with the geometry id of each row.
        """
        return self._index[:, 0].astype(int)



