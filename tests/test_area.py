import collision_kernels.process_trajectories
import numpy as np

import pytest


@pytest.mark.parametrize(
    "target_r",
    [
        0.1,
        0.2,
        0.5,
        1.0,
        2.0,
        5.0,
    ],
)
def test_area_ballistic(target_r):
    r = np.linspace(0, 10, 1000)
    hit = np.where(r < target_r, 1, 0)
    mock_data = np.vstack((r, hit)).T
    np.random.shuffle(mock_data)

    result = collision_kernels.process_trajectories.effective_radius(mock_data)

    assert np.allclose(
        result,
        target_r,
        atol=1e-02,
    )


if __name__ == "__main__":
    test_area_ballistic(1.0)
