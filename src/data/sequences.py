"""Sliding-window sequence creation (never crosses split or vehicle boundaries)."""
import numpy as np


def make_windows(F: np.ndarray, T: np.ndarray, length: int, horizon: int):
    """F: (n, n_features) inputs, T: (n, n_targets).
    Returns X (m, length, n_features), y (m, n_targets), target_row_index (m,)."""
    n = len(F)
    m = n - length - horizon + 1
    if m <= 0:
        return (np.empty((0, length, F.shape[1]), np.float32),
                np.empty((0, T.shape[1]), np.float32), np.empty(0, int))
    start = np.arange(m)
    idx = start[:, None] + np.arange(length)[None, :]
    tgt = start + length - 1 + horizon
    return F[idx].astype(np.float32), T[tgt].astype(np.float32), tgt
