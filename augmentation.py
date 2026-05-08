"""
Data augmentation functions for mosquito 3D flight trajectory.

Each function takes a numpy array of shape (T, 3) representing (x, y, z)
coordinates over T timesteps, and returns an augmented array of the same shape.

Usage example:
    import numpy as np
    import pandas as pd
    from augmentation import translate_last_to_origin

    df = pd.read_csv("data/train/TRAIN_00001.csv")
    coords = df[["x", "y", "z"]].values  # shape (T, 3)

    augmented = translate_last_to_origin(coords)
"""

import numpy as np


def translate_last_to_origin(coords: np.ndarray) -> np.ndarray:
    """Translate trajectory so that the last coordinate becomes (0, 0, 0).

    Args:
        coords: Array of shape (T, 3) with columns [x, y, z].

    Returns:
        Translated array of the same shape.
    """
    offset = coords[-1].copy()
    return coords - offset
