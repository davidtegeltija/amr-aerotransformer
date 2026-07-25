"""
========================================================================
Offline data prep: carve a smaller SuperWing dataset out of the full one.
========================================================================

Run once, before training, to produce the ``*_subset-{n}.npy`` arrays that the
data configs point at (see ``configs/data/wing.yaml``). Selects ``n_samples``
geometries at random and keeps every simulation row belonging to them, so the
subset stays geometry-complete and the geometry-disjoint split still works on it.

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
