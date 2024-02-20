import collision_kernels.integrate_with_weight
import collision_kernels.generate_trajectories
import numpy as np
import numpy.random

import pytest


@pytest.mark.parametrize(
    "peclet",
    [                
        1,
        10,
        100,
        1000,
        10_000,
        100_000,
        1_000_000
    ],
)
def test_acrivos_vs_simpson(peclet):

    def acrivos(peclet):
        return (1/2)*(1+(1+2*peclet)**(1/3))

    def weight(x):
        return 2 * np.pi * x

    def sampler(x, n):
        """
        Location x
        Number of samples n
        """

        batch_size = 10000
        done = 0
        results = list()
        while done < n:
            estimated_probability, good_runs = collision_kernels.generate_trajectories.calculate_probability(x,batch_size)
            results.append(
                estimated_probability
            )
            done = done + good_runs

        ret = np.array(results)
        return ret

    absolute_tolerance = acrivos(peclet) * 0.01
    max_step_size = 0.1

    effe = collision_kernels.integrate_with_weight.integrate_with_weight(
        weight=weight,
        sampler=sampler,
        absolute_tolerance=absolute_tolerance,
        max_step_size=0.1,
    )

    assert effe == pytest.approx(acrivos(peclet), rel=0.1)


if __name__ == "__main__":
    test_acrivos_vs_simpson(steepness=5, location=1, true_result=5.52)
