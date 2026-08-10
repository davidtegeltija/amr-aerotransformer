from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset, Subset


def split_by_group_id(dataset: Dataset, group_ids, val_split: float, seed: int, group_name: str):
    """Split a dataset into disjoint train/val/test subsets by group id (geometry or case)."""
    try:
        train_idx, val_idx, test_idx = geometry_disjoint_split(group_ids, val_split, seed)
        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)
        test_dataset = Subset(dataset, test_idx)
        print(f"{group_name}-disjoint split (seed={seed}): "
              f"{len(train_idx)} train / {len(val_idx)} val / {len(test_idx)} test")
        return train_dataset, val_dataset, test_dataset
    except ValueError as e:
        print(f"WARNING: {e}")
        print("Falling back to validating on the full training set. This is an "
              "overfit sanity check only — val_loss is NOT a generalization metric "
              f"when train and val share a {group_name.lower()}.")
        return dataset, dataset, dataset

    
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
                       (see ``WingDataset.geometry_ids``)
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
