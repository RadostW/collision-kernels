import collision_kernels.process_trajectories
import numpy as np


def test_area_ballistic():
    target_r = 1.0
    r = np.linspace(0, 10, 1000)
    hit = np.where(r < target_r, 1, 0)
    mock_data = np.vstack((r, hit))
    np.random.shuffle(mock_data)

    assert np.allclose(
        collision_kernels.process_trajectories.effective_radius(mock_data),
        target_r,
        atol=1e-03,
    )
