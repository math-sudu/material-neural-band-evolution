"""Rigid-motion matrix for the published observation operator."""
import numpy as np


def rigid_matrix(points):
    """u = translation + infinitesimal rotation cross position, point-major."""
    r = points - points.mean(axis=0)
    out = np.zeros((len(r), 3, 6))
    out[:, :, :3] = np.eye(3)
    out[:, 0, 4], out[:, 0, 5] = r[:, 2], -r[:, 1]
    out[:, 1, 3], out[:, 1, 5] = -r[:, 2], r[:, 0]
    out[:, 2, 3], out[:, 2, 4] = r[:, 1], -r[:, 0]
    return out.reshape(-1, 6)
