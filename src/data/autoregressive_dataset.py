"""
========================================================================
Next-step CFDBench datasets for the Adaptive Mesh CFD pipeline.
========================================================================

Contents
--------
AutoregressiveDataset - shared loading, pairing, normalisation and sampling
CavityDataset         - lid-driven cavity flow
DamDataset            - flow over a dam barrier
CylinderDataset       - flow around a cylinder
TubeDataset           - flow through a tube

Every problem is laid out as per-case folders::

    data/<problem>/<group>/<caseXXXX>/
        u.npy       # [T, H, W]  x-velocity, one channel per timestep
        v.npy       # [T, H, W]  y-velocity, one channel per timestep
        case.json   # physical parameters of the case

The learning task is next-step prediction: given the flow (u, v) at frame t,
predict the flow at frame t+k. A case with T timesteps yields T-k pairs. Physical
parameters are broadcast to constant channels and appended to the input.

Compared with the AMR-Transformer code (``src/dataset/`` there), three of its
steps are left out for every problem: the inlet column and wall rows padded onto
the stored grid, the convergence cutoff that stops a case once consecutive frames
barely change, and the fixed parameter scaling, which is replaced by a z-score
over the dataset's own cases.

Samples follow the same contract as :class:`~src.data.wing_dataset.WingDataset`::

    {
        "input":  np.ndarray [H, W, C],   # (u_t, v_t, + broadcast physics)
        "target": np.ndarray [H, W, 2],   # (u_{t+k}, v_{t+k}) or the residual
    }

so the existing collate function, model and training loop consume them unchanged.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Shared next-step dataset
# ---------------------------------------------------------------------------

class AutoregressiveDataset(Dataset):
    """Next-step (t -> t+k) dataset built from per-case folders.

    Each case folder contributes ``T - k`` frame pairs. Every frame is held in RAM
    once as float32; the pairs index into them. A subclass sets the defaults below
    and, if a parameter is derived from case.json rather than read from it,
    overrides :meth:`_read_meta`.

    Args:
        input_path: Path to the dataset root (the folder containing the group
            sub-folders, e.g. ``data/cavity``).
        groups: Group sub-folders to include (e.g. ``("bc", "geo", "prop")``).
            ``None`` includes every sub-folder found under ``input_path``.
        param_keys: case.json keys to append as constant input channels.
            ``None`` uses the class default; ``()`` disables physics channels.
        time_step: Frames between input and target (k). ``None`` uses the class default.
        predict_residual: If True, the target is the per-pixel change
            ``frame[t+k] - frame[t]`` instead of the absolute next frame.
        normalize: If True (default), standardise the u/v channels and each
            physics channel to zero mean / unit variance using statistics
            computed over the dataset. Stats are exposed for inversion via
            :meth:`denormalize_target`.

    Attributes:
        H, W: Spatial grid size.
        input_channels: Number of input channels (``2 + len(param_keys)``).
        output_channels: Number of target channels (always 2: u, v).
    """

    DEFAULT_PARAM_KEYS: Tuple[str, ...] = ()
    DEFAULT_TIME_STEP: int = 1

    def __init__(
        self,
        input_path: str,
        groups: Optional[Sequence[str]] = None,
        param_keys: Optional[Sequence[str]] = None,
        time_step: Optional[int] = None,
        predict_residual: bool = False,
        normalize: bool = True,
    ):
        path_input = Path(input_path)
        if not path_input.exists():
            raise FileNotFoundError(f"input_path does not exist: {input_path}")

        self.param_keys = tuple(self.DEFAULT_PARAM_KEYS if param_keys is None else param_keys)
        self.time_step = self.DEFAULT_TIME_STEP if time_step is None else time_step
        if self.time_step < 1:
            raise ValueError(f"time_step must be at least 1, got {self.time_step}")
        self.predict_residual = predict_residual
        self.normalize = normalize
        k = self.time_step

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

        # Load every case once; keep frames per case and index the (t, t+k) pairs
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

            frames = np.stack([u, v], axis=-1)   # [T, H, W, 2]
            T, h, w, _ = frames.shape
            if H is None:
                H, W = h, w
            elif (h, w) != (H, W):
                raise ValueError(f"{cdir}: grid {h}x{w} differs from {H}x{W}; all cases must share a grid.")
            if T <= k:
                continue  # need at least one (t, t+k) pair

            ci = len(self._frames)
            self._frames.append(frames)
            self._params.append(self._read_params(cdir))
            case_ids.extend([case_id] * (T - k))
            self._pairs.extend((ci, t) for t in range(T - k))

        if not self._pairs:
            raise ValueError(f"No consecutive-frame pairs could be built (every case had <= {k} timesteps).")

        self._case_ids: np.ndarray = np.asarray(case_ids, dtype=int)

        # Expose dataset metadata
        self.H, self.W = int(self._frames[0].shape[1]), int(self._frames[0].shape[2])
        self.input_channels = 2 + len(self.param_keys)
        self.output_channels = 2

        self._compute_stats()

        print(
            f"{type(self).__name__}: {len(self)} pairs from {len(self._frames)} cases  |  "
            f"grid {H}x{W}  |  input_channels={self.input_channels}  "
            f"output_channels={self.output_channels}  |  "
            f"time_step={k}  residual={predict_residual}  normalize={normalize}"
        )

    def _read_meta(self, cdir: Path) -> Dict:
        """Read a case's case.json. Override to derive further parameters from it.

        Args:
            cdir: Case directory containing ``case.json``.

        Returns:
            The case's parameters by name.
        """
        return json.loads((cdir / "case.json").read_text())

    def _read_params(self, cdir: Path) -> np.ndarray:
        """Read the requested physics fields of a case.

        Args:
            cdir: Case directory containing ``case.json``.

        Returns:
            Float array of shape ``[len(param_keys)]`` in ``param_keys`` order.
        """
        if not self.param_keys:
            return np.zeros((0,), dtype=np.float32)
        meta = self._read_meta(cdir)
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

        # One pass over the cases: u/v stats over every frame and residual stats
        # over every (t, t+k) pair difference. Sums accumulate in float64: a
        # float32 sum over a 2000-frame case is off by ~1% in the mean.
        k = self.time_step
        n = 0
        s = np.zeros(2, np.float64)
        ss = np.zeros(2, np.float64)
        n_diff = 0
        s_diff = np.zeros(2, np.float64)
        ss_diff = np.zeros(2, np.float64)
        for frames in self._frames:
            flat = frames.reshape(-1, 2)
            n += flat.shape[0]
            s += flat.sum(0, dtype=np.float64)
            ss += np.square(flat, dtype=np.float64).sum(0)
            if self.predict_residual:
                d = (frames[k:] - frames[:-k]).reshape(-1, 2)
                n_diff += d.shape[0]
                s_diff += d.sum(0, dtype=np.float64)
                ss_diff += np.square(d, dtype=np.float64).sum(0)
        self.uv_mean = (s / n).astype(np.float32)
        self.uv_std = np.sqrt(np.maximum(ss / n - (s / n) ** 2, 0)).astype(np.float32) + eps

        # Target stats: identical to u/v for absolute targets; from the per-pair
        # differences for residual targets (mean ~0, smaller scale).
        if self.predict_residual:
            self.target_mean = (s_diff / n_diff).astype(np.float32)
            self.target_std = np.sqrt(np.maximum(ss_diff / n_diff - (s_diff / n_diff) ** 2, 0)).astype(np.float32) + eps
        else:
            self.target_mean = self.uv_mean.copy()
            self.target_std = self.uv_std.copy()

        # Physics stats over cases (unweighted by pair count), in float64 so a
        # parameter that is constant over the dataset z-scores to exactly 0.
        params = np.stack(self._params, axis=0).astype(np.float64) if self.param_keys else np.zeros((len(self._frames), 0))
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
        y_next = frames[t + self.time_step]
        y = y_next - x if self.predict_residual else y_next

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
        ``geometry_disjoint_split`` (see src.data.split) to build a case-disjoint
        split instead -- the direct analogue of :meth:`WingDataset.geometry_ids`.

        Returns:
            Integer array of shape ``[N]`` with the case id of each row.
        """
        return self._case_ids


# ---------------------------------------------------------------------------
# CFDBench problems
# ---------------------------------------------------------------------------

class CavityDataset(AutoregressiveDataset):
    """Lid-driven cavity flow, t -> t+1 (frames 0.1 s apart)."""

    # vel_top and viscosity set the Reynolds number and thus the dynamics, so a
    # single frame is otherwise ambiguous across cases; density/height/width
    # disambiguate the geometry and fluid. 'rotated' is boolean metadata and
    # deliberately excluded.
    DEFAULT_PARAM_KEYS = ("vel_top", "viscosity", "density", "height", "width")


class DamDataset(AutoregressiveDataset):
    """Flow over a dam barrier, t -> t+1 (frames 0.1 s apart).

    The parameters are theirs (``param_keys`` in their ``load_case_data``), which
    leave the barrier size out. Their mask channel is left out as well: its barrier
    line slices the time axis and masks nothing, and their collate drops it.
    """

    DEFAULT_PARAM_KEYS = ("velocity", "density", "viscosity", "height", "width")


class CylinderDataset(AutoregressiveDataset):
    """Flow around a cylinder, t -> t+100 (frames 0.001 s apart).

    k = 100 (0.1 s) is the AMR-Transformer code's spacing (``delta_time=0.1`` over
    0.001 s frames). case.json gives the domain bounds around a cylinder centred
    at the origin; as in their ``load_case_data`` these become the cylinder centre
    and the domain size.
    """

    # The flow conditions plus the cylinder size and position and the domain
    # size, which together fix the geometry of a case.
    DEFAULT_PARAM_KEYS = ("vel_in", "density", "viscosity", "radius", "center_x", "center_y", "height", "width")
    DEFAULT_TIME_STEP = 100

    def _read_meta(self, cdir: Path) -> Dict:
        meta = super()._read_meta(cdir)
        # Domain bounds -> cylinder centre (distance from the left / bottom
        # boundary) and domain size.
        meta["center_x"] = -meta["x_min"]
        meta["center_y"] = -meta["y_min"]
        meta["height"] = meta["y_max"] - meta["y_min"]
        meta["width"] = meta["x_max"] - meta["x_min"]
        return meta


class TubeDataset(AutoregressiveDataset):
    """Flow through a tube, t -> t+1, the spacing of the AMR-Transformer code."""

    # Their ``case_params_keys``, i.e. every field of case.json.
    DEFAULT_PARAM_KEYS = ("vel_in", "density", "viscosity", "height", "width")
