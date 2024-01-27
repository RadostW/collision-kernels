import collision_kernels.process_trajectories
import numpy as np


def test_area_ballistic():
    target_r = 1.05
    r = np.linspace(0, 10, 5000)
    hit = np.where(r < target_r, 1, 0)
    mock_data = np.vstack((r, hit))
    np.random.shuffle(mock_data)

    sh = collision_kernels.process_trajectories.sherwood(mock_data)

    assert np.allclose(
        sh,
        1.0/4,
        atol=1e-03,
    )


if __name__ == "__main__":
    test_area_ballistic()