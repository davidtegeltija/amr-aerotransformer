from pathlib import Path
from typing import Optional
import numpy as np


def create_data_subset(
    input_array: np.ndarray, 
    target_array: np.ndarray, 
    index_array: np.ndarray, 
    n_samples: int,
    save_path: Optional[str] = None,
) -> None:
    """
    Create a subset of the SuperWing dataset by selecting n_samples geometries.

    Args:
        input_array  : [N_geom, 3, H, W] geometry array (geom0.npy)
        target_array : [N_samples, 3, H, W] simulation results (data.npy)
        index_array  : [N_samples, ...] mapping array where column 0 gives the geometry index for each target row (index.npy)
        n_samples    : number of geometries to select from N_geom
        save_path    : if given, save the resulting subset to this path as 'geom_subset-{n_samples}.npy' and 'data_subset-{n_samples}.npy'
    """
    if target_array.shape[0] != index_array.shape[0]:
        raise ValueError(f"Target and index array must have the same first dimension shape, got {target_array.shape[0]} vs {index_array.shape[0]}.")

    # Select n_samples random geometries from the 4239 available
    selected_inputs = np.random.choice(input_array.shape[0], size=n_samples, replace=False)
    selected_inputs.sort()

    # Use index_array's first column to find which target rows map to each geometry
    shape_indices = index_array[:, 0].astype(int)

    input_subsets = []
    target_subsets = []

    for input_index in selected_inputs:
        # Find all target samples that correspond to this geometry
        mask = shape_indices == input_index
        target_samples = target_array[mask]
        n_corresponding = target_samples.shape[0]

        # Repeat the input geometry to match the number of target samples
        input_repeated = np.repeat(input_array[input_index:input_index+1], n_corresponding, axis=0)

        input_subsets.append(input_repeated)
        target_subsets.append(target_samples)

    input_subset = np.concatenate(input_subsets, axis=0)
    target_subset = np.concatenate(target_subsets, axis=0)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(f"{save_path}/geom_subset-{n_samples}.npy", input_subset)
        np.save(f"{save_path}/data_subset-{n_samples}.npy", target_subset)
        print(f"New dataset created at {save_path}.\nInput subset: {input_subset.shape}, Target subset: {target_subset.shape}\n")
    else:
        print("If you want the new dataset to be saved add a save_path argument to the function call")


def create_sample_npz(
    input_array: np.ndarray,
    target_array: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """
    Save one CFD sample as a .npz file compatible with AeroDataset.
    
    Args:
        input_array  : [H, W, C] float32 input field
        target_array : [H, W, output_dim] float32 target field
        save_path    : if given, write a compressed .npz to this path with keys 'input' and 'target'`.'
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
    index_array = np.load("data/index.npy")
    n_samples = 100  # number of geometries to select
    save_path = "/data"

    create_data_subset(input_array, target_array, index_array, n_samples, save_path)
    # create_sample_npz(input_array, target_array, save_path)



