"""
========================================================================
Offline data prep: carve a smaller SuperWing dataset out of the full one.
========================================================================

Optional: only needed to train on *fewer* geometries than the dataset holds.
``WingDataset`` reads the raw arrays directly (it resolves each simulation row to
its geometry through column 0 of the index), so training on everything needs no
preparation at all — point ``configs/data/wing.yaml`` at geom0/data/index.

Selects ``n_samples`` geometries at random and keeps every simulation row
belonging to them, so the subset stays geometry-complete and the
geometry-disjoint split still works on it. The geometry rows are repeated to
match the target rows, which is why a subset costs ~6.8x its geometry data on
disk — the reason the full dataset is better read raw.

The heavy lifting lives in ``src/utils/data_utils.py``; this file only supplies
the paths and sizes for one run.
"""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.data_utils import create_data_subset, create_sample_npz


if __name__ == "__main__":
    input_array = np.load("data/crmmgeom.npy")
    target_array = np.load("data/crmmdata.npy")
    index_array = np.load("data/crmmindex.npy")
    n_samples = 100  # number of geometries to select
    save_path = "/data"

    create_data_subset(input_array, target_array, index_array, n_samples, save_path)
    # create_sample_npz(input_array, target_array, save_path)
