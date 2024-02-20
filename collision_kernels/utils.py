import numpy as np


def sherwood_from_area(peclet, effective_area, small_r):
    r_eff = (effective_area / np.pi) ** 0.5
    return (peclet / 4) * ((r_eff / (small_r + 1)) ** 2)
