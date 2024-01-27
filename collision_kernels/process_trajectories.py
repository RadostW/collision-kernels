import numpy as np


def effective_radius(trajectory_array):
    """
    Compute effective area of the particle from collision data

    Parameters
    ----------
    trajectory_array : np.array
        2 by N array of tuples initial radius and outcome (1 - hit, 0 - miss)

    Returns
    -------
    float
        Effective radius of the particle

    """

    # Compute outcome as E[2\pi r hit(r)]
    effecive_area = np.mean(2 * np.pi * trajectory_array[:, 0] * trajectory_array[:, 1])
    effective_radius = (effecive_area / np.pi)**0.5

    return effective_radius

def sherwood(trajectory_array):
    """
    Compute Sherwood number from collision data

    Parameters
    ----------
    trajectory_array : np.array
        2 by N array of tuples initial radius and outcome (1 - hit, 0 - miss)

    Returns
    -------
    float
        Sherwood number

    """
    raise NotImplementedError