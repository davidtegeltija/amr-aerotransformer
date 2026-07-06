"""
========================================================================
Cavity-flow next-step dataset for the Adaptive Mesh CFD pipeline.
========================================================================

Loads lid-driven cavity simulations laid out as per-case folders::

    data/cavity/<group>/<caseXXXX>/
        u.npy       # [T, H, W]  x-velocity, one channel per timestep
        v.npy       # [T, H, W]  y-velocity, one channel per timestep
        case.json   # physical parameters (vel_top, viscosity, density, ...)

The learning task is next-step prediction: given the flow (u, v) at time t,
predict the flow at time t+1. A case with T timesteps yields T-1 pairs.

Samples follow the same contract as :class:`~src.data.dataset.AeroDataset`::

    {
        "input":  np.ndarray [H, W, C],   # (u_t, v_t, + broadcast physics)
        "target": np.ndarray [H, W, 2],   # (u_{t+1}, v_{t+1}) or the residual
    }

so the existing collate function, model and training loop consume it unchanged.
The model needs no architecture change: construct it with
``input_channels = 2 + len(param_keys)`` and ``output_channels = 2``.

For the SuperWing steady-state data, see :class:`~src.data.dataset.AeroDataset`.
For the DataLoader collate function, see src.data.collate_fn.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset

# Physics fields (from case.json) appended as constant input channels by default.
# vel_top and viscosity set the Reynolds number and thus the dynamics, so a single
# frame is otherwise ambiguous across cases; density/height/width disambiguate the
# geometry and fluid. 'rotated' is boolean metadata and deliberately excluded.
DEFAULT_PARAM_KEYS: Tuple[str, ...] = ("vel_top", "viscosity", "density", "height", "width")


class CavityDataset(Dataset):
    """Next-step (t -> t+1) cavity-flow dataset built from per-case folders.

    Each case folder contributes ``T - 1`` consecutive-frame pairs. Physical
    parameters read from ``case.json`` are broadcast to constant channels and
    appended to the input, mirroring how :class:`AeroDataset` appends the
    angle-of-attack / Mach channels.

    Args:
        input_path: Path to the cavity dataset root (the folder containing the
            group sub-folders, e.g. ``data/cavity``).
        groups: Group sub-folders to include (e.g. ``("bc", "geo", "prop")``).
            ``None`` includes every sub-folder found under ``input_path``.
        param_keys: ``case.json`` keys to append as constant input channels.
            Pass ``()`` to disable physics channels (input becomes just u, v).
        predict_residual: If True, the target is the per-pixel change
            ``frame[t+1] - frame[t]`` instead of the absolute next frame.
        normalize: If True (default), standardise the u/v channels and each
            physics channel to zero mean / unit variance using statistics
            computed over the dataset. Stats are exposed for inversion via
            :meth:`denormalize_target`.
        max_frames_per_case: Cap on the number of timesteps read from any single
            case, to stop a very long simulation from dominating the pair count
            (one cavity case has ~1000 steps). Defaults to 200; pass ``None`` to
            read every frame.

    Attributes:
        H, W: Spatial grid size.
        input_channels: Number of input channels (``2 + len(param_keys)``).
        output_channels: Number of target channels (always 2: u, v).
    """

    def __init__(
        self,
        input_path: str,
        groups: Optional[Sequence[str]] = None,
        param_keys: Sequence[str] = DEFAULT_PARAM_KEYS,
        predict_residual: bool = False,
        normalize: bool = True,
        max_frames_per_case: Optional[int] = 200,
    ):
        path_input = Path(input_path)
        if not path_input.exists():
            raise FileNotFoundError(f"input_path does not exist: {input_path}")

        self.param_keys = tuple(param_keys)
        self.predict_residual = predict_residual
        self.normalize = normalize

        # Discover case folders (any dir holding both u.npy and v.npy).
        group_dirs = (
            [path_input / g for g in groups] if groups is not None
            else sorted(p for p in path_input.iterdir() if p.is_dir())
        )
        case_dirs: List[Path] = []
        for gdir in group_dirs:
            if not gdir.exists():
                raise FileNotFoundError(f"group folder does not exist: {gdir}")
            for cdir in sorted(gdir.iterdir()):
                if (cdir / "u.npy").exists() and (cdir / "v.npy").exists():
                    case_dirs.append(cdir)
        if not case_dirs:
            raise ValueError(f"No cases with u.npy and v.npy found under {input_path}")

        # Load every case once; keep frames per case and index the (t, t+1) pairs
        # so overlapping frames are stored a single time.
        self._frames: List[np.ndarray] = []   # each [T, H, W, 2]
        self._params: List[np.ndarray] = []   # each [len(param_keys)]
        self._pairs: List[Tuple[int, int]] = []  # (case_index, t)
        case_ids: List[int] = []              # integer id per row, for disjoint split
        H: Optional[int] = None
        W: Optional[int] = None

        for case_id, cdir in enumerate(case_dirs):
            u = np.load(cdir / "u.npy").astype(np.float32)   # [T, H, W]
            v = np.load(cdir / "v.npy").astype(np.float32)
            if u.shape != v.shape or u.ndim != 3:
                raise ValueError(f"{cdir}: expected matching [T, H, W] u/v, got {u.shape} and {v.shape}")
            if max_frames_per_case is not None:
                u, v = u[:max_frames_per_case], v[:max_frames_per_case]

            frames = np.stack([u, v], axis=-1)   # [T, H, W, 2]
            T, h, w, _ = frames.shape
            if H is None:
                H, W = h, w
            elif (h, w) != (H, W):
                raise ValueError(f"{cdir}: grid {h}x{w} differs from {H}x{W}; all cases must share a grid.")
            if T < 2:
                continue  # need at least one consecutive pair

            params = self._read_params(cdir)
            ci = len(self._frames)
            self._frames.append(frames)
            self._params.append(params)
            case_ids.extend([case_id] * (T - 1))
            self._pairs.extend((ci, t) for t in range(T - 1))

        if not self._pairs:
            raise ValueError("No consecutive-frame pairs could be built (every case had < 2 timesteps).")

        self._case_ids: np.ndarray = np.asarray(case_ids, dtype=int)

        # Expose dataset metadata
        self.H, self.W = int(self._frames[0].shape[1]), int(self._frames[0].shape[2])
        self.input_channels = 2 + len(self.param_keys)
        self.output_channels = 2

        self._compute_stats()

        print(
            f"CavityDataset: {len(self)} pairs from {len(self._frames)} cases  |  "
            f"grid {H}x{W}  |  input_channels={self.input_channels}  "
            f"output_channels={self.output_channels}  |  "
            f"residual={predict_residual}  normalize={normalize}"
        )

    def _read_params(self, cdir: Path) -> np.ndarray:
        """Read the requested physics fields from a case's case.json.

        Args:
            cdir: Case directory containing ``case.json``.

        Returns:
            Float array of shape ``[len(param_keys)]`` in ``param_keys`` order.
        """
        if not self.param_keys:
            return np.zeros((0,), dtype=np.float32)
        meta = json.loads((cdir / "case.json").read_text())
        try:
            return np.array([float(meta[k]) for k in self.param_keys], dtype=np.float32)
        except KeyError as e:
            raise KeyError(f"{cdir / 'case.json'} is missing param key {e}; found {list(meta)}") from e

    def _compute_stats(self) -> None:
        """Compute channel statistics used for optional standardisation.

        Populates per-channel mean/std for the u/v flow channels, the target
        (absolute next frame or residual), and the physics channels. When
        ``normalize`` is False these are set to identity (mean 0, std 1) so the
        same code path applies with no effect.
        """
        eps = 1e-8
        if not self.normalize:
            self.uv_mean = np.zeros(2, np.float32)
            self.uv_std = np.ones(2, np.float32)
            self.target_mean = np.zeros(2, np.float32)
            self.target_std = np.ones(2, np.float32)
            self.param_mean = np.zeros(len(self.param_keys), np.float32)
            self.param_std = np.ones(len(self.param_keys), np.float32)
            return

        # u/v stats over every frame (streamed to avoid a giant concatenation).
        n = 0
        s = np.zeros(2, np.float64)
        ss = np.zeros(2, np.float64)
        for frames in self._frames:
            flat = frames.reshape(-1, 2)
            n += flat.shape[0]
            s += flat.sum(0)
            ss += (flat.astype(np.float64) ** 2).sum(0)
        self.uv_mean = (s / n).astype(np.float32)
        self.uv_std = np.sqrt(np.maximum(ss / n - (s / n) ** 2, 0)).astype(np.float32) + eps

        # Target stats: identical to u/v for absolute targets; computed from the
        # per-pair differences for residual targets (mean ~0, smaller scale).
        if self.predict_residual:
            n = 0
            s = np.zeros(2, np.float64)
            ss = np.zeros(2, np.float64)
            for ci, t in self._pairs:
                d = (self._frames[ci][t + 1] - self._frames[ci][t]).reshape(-1, 2)
                n += d.shape[0]
                s += d.sum(0)
                ss += (d.astype(np.float64) ** 2).sum(0)
            self.target_mean = (s / n).astype(np.float32)
            self.target_std = np.sqrt(np.maximum(ss / n - (s / n) ** 2, 0)).astype(np.float32) + eps
        else:
            self.target_mean = self.uv_mean.copy()
            self.target_std = self.uv_std.copy()

        # Physics stats over cases (unweighted by pair count).
        params = np.stack(self._params, axis=0) if self.param_keys else np.zeros((len(self._frames), 0), np.float32)
        self.param_mean = params.mean(0).astype(np.float32)
        self.param_std = params.std(0).astype(np.float32) + eps

    def denormalize_target(self, target: np.ndarray) -> np.ndarray:
        """Map a normalised target/prediction back to physical units.

        Inverse of the standardisation applied to targets. For residual mode the
        result is the physical *change* to add to the input frame; for absolute
        mode it is the physical next frame directly.

        Args:
            target: Array whose last dimension is the 2 target channels.

        Returns:
            Array of the same shape in physical units.
        """
        return target * self.target_std + self.target_mean

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> Dict:
        ci, t = self._pairs[index]
        frames = self._frames[ci]
        x = frames[t]                                   # [H, W, 2]
        y = frames[t + 1] - x if self.predict_residual else frames[t + 1]

        x = (x - self.uv_mean) / self.uv_std
        y = (y - self.target_mean) / self.target_std

        if self.param_keys:
            params = (self._params[ci] - self.param_mean) / self.param_std
            param_channels = np.broadcast_to(params, (self.H, self.W, params.shape[0]))
            x = np.concatenate([x, param_channels], axis=-1)

        return {
            "index": index,
            "input": np.ascontiguousarray(x, dtype=np.float32),
            "target": np.ascontiguousarray(y, dtype=np.float32),
        }

    def case_ids(self) -> np.ndarray:
        """Return the case id of every dataset row.

        Consecutive frames of one simulation are highly correlated, so a random
        pair-level split leaks a case across train/val. Feed this to
        ``geometry_disjoint_split`` (see src.utils.data_utils) to build a
        case-disjoint split instead — the direct analogue of
        :meth:`AeroDataset.geometry_ids`.

        Returns:
            Integer array of shape ``[N]`` with the case id of each row.
        """
        return self._case_ids
