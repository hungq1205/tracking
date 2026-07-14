"""
Shared GPU-accelerated grid-grouping utilities.

Used by ScanSession.process_frames_batch (bookkeeping for per-point
confidence/frame-id/pixel through the existing Open3D voxel_down_sample +
statistical-outlier-removal pipeline) and by voxelize_cloud_with_confidence
(the confidence-aware occupancy voxelization consumed by
surface_refinement.refine_voxel_map). See CLAUDE.md's 3D Scanning Pipeline
section for how these fit together.

Device selection mirrors da3_wrapper.py's existing pattern: SCAN_DEVICE env
var (default "cuda"), falling back to CPU when CUDA isn't actually
available. Every GPU entry point here retries once on CPU on any
RuntimeError (covers CUDA OOM and CUDA-context errors) rather than crashing
a whole scan.
"""

import os
from typing import Dict, Tuple, Union

import numpy as np
import torch

_ReduceSpec = Union[str, Tuple[str, str]]  # "mean" | ("best", driver_key)


def resolve_device(device: "str | torch.device | None" = None) -> torch.device:
    """SCAN_DEVICE env var (default "cuda"), falling back to CPU when CUDA
    isn't actually available on this machine — same idiom as
    da3_wrapper.py's DA3Estimator.__init__."""
    if isinstance(device, torch.device):
        return device
    name = device or os.getenv("SCAN_DEVICE", "cuda")
    if name == "cuda" and not torch.cuda.is_available():
        name = "cpu"
    return torch.device(name)


def voxel_index(
    points: np.ndarray, voxel_size: float, origin: np.ndarray, device: torch.device
) -> torch.Tensor:
    """Integer grid index per point: floor((points - origin) / voxel_size).
    Returns an (N,3) int64 tensor on `device`."""
    pts_t = torch.as_tensor(np.asarray(points, dtype=np.float32), device=device)
    origin_t = torch.as_tensor(np.asarray(origin, dtype=np.float32), device=device)
    return torch.floor((pts_t - origin_t) / voxel_size).long()


def pack_ijk(ijk: torch.Tensor) -> torch.Tensor:
    """Pack an (N,3) int64 grid-index tensor into a single (N,) int64 key,
    suitable for torch.unique / dict lookups. Offset by a large constant so
    negative indices (points below `origin` on some axis, which can't
    happen given origin=points.min(axis=0) but is possible when reusing a
    caller-supplied origin) don't collide or go negative in the packed key.
    """
    OFFSET = 1 << 20  # generous headroom: |grid index| well under 1M for any real scan
    STRIDE = 2 * OFFSET
    shifted = ijk + OFFSET
    return (
        shifted[:, 0] * STRIDE * STRIDE
        + shifted[:, 1] * STRIDE
        + shifted[:, 2]
    )


def aggregate_by_key(
    keys: torch.Tensor,
    value_arrays: Dict[str, torch.Tensor],
    reduce_spec: Dict[str, _ReduceSpec],
    device: torch.device,
) -> dict:
    """
    Group rows of `value_arrays` by `keys` (an (N,) int64 tensor) and reduce
    each field per `reduce_spec`:
      - "mean"              → per-group mean (scatter_reduce)
      - ("best", driver)     → the row whose `value_arrays[driver]` is
                                highest within the group (scatter-argmax,
                                then gather)

    Returns {"unique_keys": (M,) int64 tensor, **{name: (M, ...) tensor}}.
    Empty input (N==0) returns empty tensors for every field.
    """
    n = keys.shape[0]
    if n == 0:
        out = {"unique_keys": torch.zeros((0,), dtype=torch.long, device=device)}
        for name, arr in value_arrays.items():
            shape = (0,) + tuple(arr.shape[1:])
            out[name] = torch.zeros(shape, dtype=arr.dtype, device=device)
        return out

    unique_keys, inverse = torch.unique(keys, return_inverse=True)
    m = unique_keys.shape[0]
    out = {"unique_keys": unique_keys}

    for name, spec in reduce_spec.items():
        arr = value_arrays[name]
        if spec == "mean":
            flat = arr.reshape(n, -1).float()
            sums = torch.zeros((m, flat.shape[1]), dtype=torch.float32, device=device)
            sums.scatter_reduce_(
                0, inverse.unsqueeze(1).expand(-1, flat.shape[1]), flat, reduce="sum"
            )
            counts = torch.zeros((m,), dtype=torch.float32, device=device)
            counts.scatter_reduce_(0, inverse, torch.ones(n, device=device), reduce="sum")
            means = sums / counts.clamp_min(1.0).unsqueeze(1)
            out[name] = means.reshape((m,) + tuple(arr.shape[1:])).to(arr.dtype)
        elif isinstance(spec, tuple) and spec[0] == "best":
            driver = value_arrays[spec[1]].float()
            best_val = torch.full((m,), float("-inf"), device=device)
            best_val.scatter_reduce_(0, inverse, driver, reduce="amax", include_self=True)
            # Row is the group's argmax if its driver value equals the group's max
            # (ties broken arbitrarily but deterministically by scan order via
            # scatter_reduce's "amax" + a stable first-match pass below).
            is_best = driver >= best_val[inverse] - 1e-12
            # Sentinel = n (one past the last valid row index) so "amin" always
            # shrinks toward the true smallest candidate row per group, giving
            # a deterministic tie-break (lowest row index wins) instead of
            # scatter's otherwise-unspecified order for duplicate confidences.
            best_row_for_group = torch.full((m,), n, dtype=torch.long, device=device)
            row_idx = torch.arange(n, device=device)
            cand = row_idx[is_best]
            cand_groups = inverse[is_best]
            best_row_for_group.scatter_reduce_(0, cand_groups, cand, reduce="amin", include_self=True)
            out[name] = value_arrays[name][best_row_for_group]
        else:
            raise ValueError(f"Unknown reduce spec for '{name}': {spec!r}")

    return out
