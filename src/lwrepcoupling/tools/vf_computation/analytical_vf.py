"""Module calculating exact closed-form analytical view factors for benchmark geometries."""

import numpy as np


def view_factor_parallel_plates(w: float, h: float, d: float) -> float:
    """Computes the analytical view factor between two identical parallel opposing rectangles.

    Args:
        w: Width of the plates (m).
        h: Height of the plates (m).
        d: Perpendicular separation distance (m).

    Returns:
        float: The geometric view factor bounded between 0.0 and 1.0.
    """
    if d <= 0.0 or w <= 0.0 or h <= 0.0:
        return 0.0

    X = w / d
    Y = h / d

    term1 = 0.5 * np.log(((1.0 + X**2) * (1.0 + Y**2)) / (1.0 + X**2 + Y**2))
    term2 = X * np.sqrt(1.0 + Y**2) * np.arctan(X / np.sqrt(1.0 + Y**2))
    term3 = Y * np.sqrt(1.0 + X**2) * np.arctan(Y / np.sqrt(1.0 + X**2))
    term4 = X * np.arctan(X)
    term5 = Y * np.arctan(Y)

    vf = (2.0 / (np.pi * X * Y)) * (term1 + term2 + term3 - term4 - term5)
    return float(np.clip(vf, 0.0, 1.0))


def view_factor_perpendicular_plates(w1: float, w2: float, h: float) -> float:
    """Computes the analytical view factor between two perpendicular rectangles sharing a common
        edge 'h'.

    Args:
        w1: Width of plate 1, the emitting surface (m).
        w2: Width of plate 2, the receiving surface (m).
        h: Height of the shared common junction edge (m).

    Returns:
        float: The geometric view factor bounded between 0.0 and 1.0.
    """
    if h <= 0.0 or w1 <= 0.0 or w2 <= 0.0:
        return 0.0

    # Scale dimensions relative to the shared edge height H
    W = w1 / h
    L = w2 / h

    W2 = W**2
    L2 = L**2

    term1 = W * np.arctan(1.0 / W)
    term2 = L * np.arctan(1.0 / L)
    term3 = np.sqrt(W2 + L2) * np.arctan(1.0 / np.sqrt(W2 + L2))

    # Logarithmic components group safely
    num_part1 = (1.0 + W2) * (1.0 + L2) / (1.0 + W2 + L2)
    num_part2 = (W2 * (1.0 + W2 + L2) / ((1.0 + W2) * (W2 + L2))) ** W2
    num_part3 = (L2 * (1.0 + W2 + L2) / ((1.0 + L2) * (W2 + L2))) ** L2

    term4 = 0.25 * np.log(num_part1 * num_part2 * num_part3)

    vf = (1.0 / (np.pi * W)) * (term1 + term2 - term3 + term4)
    return float(np.clip(vf, 0.0, 1.0))
