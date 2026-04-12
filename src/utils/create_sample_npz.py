from pathlib import Path
from typing import Optional
import numpy as np


def create_sample_npz(
    input_array: np.ndarray,
    target_array: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Save one CFD sample as a .npz file compatible with AeroDataset.

    Parameters
    ----------
    input_array  : [H, W, C]            numpy float32 array
    target_array : [H, W, output_dim]   numpy float32 array
    save_path    : output file path     if given, save the data to this path

    Example
    -------
    inputs = np.stack([x_grid, y_grid, aoa_grid, mach_grid], axis=-1)  # [H, W, 4]
    target = np.stack([u, v, p], axis=-1)                               # [H, W, 3]
    save_sample_npz(inputs, target, "data/train/sample_0000.npz")
    """
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, input=input_array, target=target_array)
        print(f"New .npz sample created at {save_path}.\nInput: {input_array.shape}, Target: {target_array.shape}\n")
    else:
        print("If you want the new .npz sample to be saved add a save_path argument to the function call")


if __name__ == "__main__":
    input_array = np.load("data/crmmgeom.npy")
    target_array = np.load("data/crmmdata.npy")
    save_path = "/data"

    create_sample_npz(input_array, target_array, save_path)