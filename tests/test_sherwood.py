import collision_kernels.process_trajectories
import numpy as np

import pytest


@pytest.mark.parametrize(
    "small_r",
    [
        0.05,
        0.10,
        0.50,
        1.0,
    ],
)
def test_area_ballistic(small_r):
    # small_r = 0.05
    target_r = 1.0 + small_r
    r = np.linspace(0, 10, 1000)
    hit = np.where(r < target_r, 1, 0)
    mock_data = np.vstack((r, hit)).T
    np.random.shuffle(mock_data)

    sh = collision_kernels.process_trajectories.sherwood(
        mock_data, peclet=1, small_r=small_r
    )

    assert np.allclose(
        sh,
        0.25,
        atol=1e-02,
    )


if __name__ == "__main__":
    test_area_ballistic()
