import collision_kernels
import numpy as np


def test_area_ballistic():
    target_r = 1.0
    r = np.linspace(0, 10, 1000)
    hit = np.where(r < target_r, 1, 0)
    mock_data = np.shuffle(np.vstack((r, hit)))

    assert np.allclose(
        collision_kernels.process_trajectories.effective_radius(mock_data),
        target_r,
        atol=1e-03,
    )
