from pathlib import Path
from typing import List, Optional, Tuple
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

    # Select n_samples random geometries from the N_geom available
    selected_inputs = np.random.choice(input_array.shape[0], size=n_samples, replace=False)
    selected_inputs.sort()

    # Use index_array's first column to find which target rows map to each geometry
    shape_indices = index_array[:, 0].astype(int)

    input_subsets = []
    target_subsets = []
    index_subsets = []

    for input_index in selected_inputs:
        # Find all target samples that correspond to this geometry
        mask = shape_indices == input_index
        target_samples = target_array[mask]
        index_samples = index_array[mask]
        n_corresponding = target_samples.shape[0]

        # Repeat the input geometry to match the number of target samples
        input_repeated = np.repeat(input_array[input_index:input_index+1], n_corresponding, axis=0)

        input_subsets.append(input_repeated)
        target_subsets.append(target_samples)
        index_subsets.append(index_samples)

    input_subset = np.concatenate(input_subsets, axis=0)
    target_subset = np.concatenate(target_subsets, axis=0)
    index_subset = np.concatenate(index_subsets, axis=0)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(f"{save_path}/geom_subset-{n_samples}.npy", input_subset)
        np.save(f"{save_path}/data_subset-{n_samples}.npy", target_subset)
        np.save(f"{save_path}/index_subset-{n_samples}.npy", index_subset)
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
        target_array : [H, W, output_channels] float32 target field
        save_path    : if given, write a compressed .npz to this path with keys 'input' and 'target'`.'
    """
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, input=input_array, target=target_array)
        print(f"New .npz sample created at {save_path}.\nInput: {input_array.shape}, Target: {target_array.shape}\n")
    else:
        print("If you want the new .npz sample to be saved add a save_path argument to the function call")


def geometry_disjoint_split(
    geometry_ids: np.ndarray,
    val_split: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split row indices into train/val/test so no geometry appears in two splits.

    The test geometries are drawn first and held out. Validation geometries are
    then drawn from the geometries that remain, and every geometry left over
    forms the training split.

    Args:
        geometry_ids : [N] geometry id of every dataset row
                       (see ``AeroDataset.geometry_ids``)
        val_split    : fraction of *geometries* (not rows) used for validation
        seed         : seed for the geometry choice, so the split is
                       reproducible across runs

    Returns:
        ``(train_idx, val_idx, test_idx)``: three lists of row indices,
        disjoint by geometry.
    """
    # Fraction of geometries held out for testing. Fixed, not a config knob: the
    # test split must stay the same across every run and config for the numbers
    # reported on it to be comparable.
    test_split = 0.1

    geometry_ids = np.asarray(geometry_ids)
    unique = np.unique(geometry_ids)
    n_test_geom = int(test_split * len(unique))
    n_val_geom = int(val_split * len(unique))

    # Check if there are too few geometries to form a non-empty train, val and test split
    if n_test_geom < 1 or n_val_geom < 1 or n_test_geom + n_val_geom >= len(unique):
        raise ValueError(
            f"Cannot build a geometry-disjoint split. {len(unique)} unique "
            f"geometries with val_split={val_split} and test_split={test_split} yield "
            f"{n_val_geom} validation and {n_test_geom} test geometries. "
            f"Need at least 1 geometry in each of the three splits."
        )

    # One shuffle of the geometries: the first slice is test, the next is val,
    # and the rest is train, so the three splits cannot share a geometry.
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    test_geoms = set(shuffled[:n_test_geom].tolist())
    val_geoms = set(shuffled[n_test_geom:n_test_geom + n_val_geom].tolist())

    train_idx, val_idx, test_idx = [], [], []
    for i, g in enumerate(geometry_ids):
        if g in test_geoms:
            test_idx.append(i)
        elif g in val_geoms:
            val_idx.append(i)
        else:
            train_idx.append(i)
    return train_idx, val_idx, test_idx


def test_row_indices(dataset, dataset_type: str, val_split: float, seed: int) -> List[int]:
    """
    Recover the held-out test rows of a dataset built from a training config.

    The split is a pure function of the group ids, ``val_split`` and ``seed``, so
    replaying it here reproduces exactly the rows the training run held out. That
    keeps the test set off disk: prediction scripts recover it from the same config
    instead of a second copy of the data that could drift out of sync.

    Args:
        dataset      : the full dataset, before any splitting
        dataset_type : the config's 'dataset' key ('wing_dataset' or 'cavity_dataset'),
                       which decides whether rows group by geometry or by case
        val_split    : the config's val_split, so the split matches the training run
        seed         : the config's seed, likewise

    Returns:
        Row indices of the test split, i.e. rows the model has never seen.
    """
    group_ids = dataset.case_ids() if dataset_type == "cavity_dataset" else dataset.geometry_ids()
    _, _, test_idx = geometry_disjoint_split(group_ids, val_split, seed)
    return test_idx


if __name__ == "__main__":
    input_array = np.load("data/crmmgeom.npy")
    target_array = np.load("data/crmmdata.npy")
    index_array = np.load("data/crmmindex.npy")
    n_samples = 100  # number of geometries to select
    save_path = "/data"

    create_data_subset(input_array, target_array, index_array, n_samples, save_path)
    # create_sample_npz(input_array, target_array, save_path)



