import numpy as np


def effective_radius(trajectory_array):
    """
    Compute effective area of the particle from collision data
    ASSUMES UNIFORM SAMPLING!

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
    radial_distance = trajectory_array[:, 0]
    is_hit = trajectory_array[:, 1]

    effective_area = np.max(radial_distance) * np.mean(
        2 * np.pi * radial_distance * is_hit
    )

    effective_radius = (effective_area / np.pi) ** 0.5

    return effective_radius


def sherwood(trajectory_array, peclet, small_r=0.05):
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
    r_eff = effective_radius(trajectory_array)

    return (peclet / 4) * ((r_eff / (small_r + 1)) ** 2)
