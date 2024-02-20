import collision_kernels.integrate_with_weight
import collision_kernels.generate_trajectories
import collision_kernels.clift_predictions
import collision_kernels.utils
import numpy as np
import numpy.random

import pytest


@pytest.mark.parametrize(
    "peclet",
    [
        #    1,
        #    10,
        #    100,
        1000,
        10_000,
        #    100_000,
        #    1_000_000
    ],
)
def test_literature_vs_simpson(peclet):

    true_sherwood = collision_kernels.clift_predictions.sherwood(peclet=peclet)
    true_area = collision_kernels.clift_predictions.effective_area(peclet=peclet)

    small_r = 0.01
    batch_size = 10000
    absolute_tolerance = true_area * 0.01
    max_step_size = 0.5
    t_max = 100

    print(f"{absolute_tolerance=}")

    def weight(x):
        return 2 * np.pi * x

    def sampler(x, n):
        """
        Location x
        Number of samples n
        """

        # print(f"Sampler {x=} {n=}")
        done = 0
        results = list()
        while done < n:
            estimated_probability, good_runs = (
                collision_kernels.generate_trajectories.calculate_probability(
                    x_position=x,
                    trials=batch_size,
                    peclet=peclet,
                    t_max=t_max,
                    small_r=small_r
                )
            )
            results.append(estimated_probability)
            done = done + good_runs

        ret = np.array(results)
        return ret

    effective_area = collision_kernels.integrate_with_weight.integrate_with_weight(
        weight=weight,
        sampler=sampler,
        absolute_tolerance=absolute_tolerance,
        max_step_size=max_step_size,
    )

    sherwood = collision_kernels.utils.sherwood_from_area(
        peclet=peclet, effective_area=effective_area, small_r=small_r
    )

    print(f"{sherwood=} {true_sherwood=}")
    print(f"{effective_area=} {true_area=}")

    assert sherwood == pytest.approx(true_sherwood, rel=0.1)


if __name__ == "__main__":
    test_literature_vs_simpson(1000)
