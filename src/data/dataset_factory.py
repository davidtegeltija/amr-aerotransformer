from typing import Dict, Tuple
from torch.utils.data import Dataset

from src.data.wing_dataset import WingDataset
from src.data.cavity_dataset import CavityDataset
from src.data.synthetic_dataset import SyntheticDataset


def build_dataset(args: Dict) -> Tuple[Dataset, str]:
    """
    Construct the dataset selected by a merged model+data config.

    That fallback is why the resolved type is returned alongside the dataset:
    it can differ from ``args['dataset']``, and callers that branch on the kind
    of dataset they got (to pick a split, or a group id to split on) must follow
    the choice actually made here rather than re-deriving it from the config.

    Args:
        args : merged config, i.e. a model config with a data config
               (``configs/data/*.yaml``) merged over it. Every key read here
               lives in that data half, so a config missing 'dataset' was
               never merged and would otherwise fail much later with a
               confusing error.
    """
    if "dataset" not in args:
        raise KeyError(
            "Config has no 'dataset' key. The data keys live in configs/data/*.yaml "
            "and must be merged over the model config before building a dataset."
        )

    dataset_type = args["dataset"]
    input_file = args.get("input_file")

    if dataset_type == "wing_dataset" and input_file is not None:
        print(f"Using wing data from {input_file}")
        return WingDataset(input_path=input_file, target_path=args["target_file"], index_path=args["index_file"]), dataset_type

    if dataset_type == "cavity_dataset" and input_file is not None:
        print(f"Using cavity next-step data from {input_file}")
        return CavityDataset(input_path=input_file), dataset_type

    if dataset_type == "synthetic_dataset" or input_file is None:
        print("No input and target data provided -> using synthetic dataset.")
        return SyntheticDataset(n_samples=64, seed=args["seed"]), dataset_type

    raise ValueError(
        f"Unknown dataset {dataset_type!r}. Valid options are: "
        "wing_dataset, cavity_dataset, synthetic_dataset."
    )
