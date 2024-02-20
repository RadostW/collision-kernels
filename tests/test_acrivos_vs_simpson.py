import collision_kernels.integrate_with_weight
import collision_kernels.generate_trajectories
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
def test_acrivos_vs_simpson(peclet):

    def acrivos(peclet):
        return (1/2)*(1+(1+2*peclet)**(1/3))
    
    true_sherwood = acrivos(peclet)

    batch_size = 1000
    absolute_tolerance = acrivos(peclet) * 0.1
    max_step_size = 0.5

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
            estimated_probability, good_runs = collision_kernels.generate_trajectories.calculate_probability(x,batch_size,peclet)            
            results.append(
                estimated_probability
            )
            done = done + good_runs

        ret = np.array(results)
        return ret    

    effective_area = collision_kernels.integrate_with_weight.integrate_with_weight(
        weight=weight,
        sampler=sampler,
        absolute_tolerance=absolute_tolerance,
        max_step_size=max_step_size,
    )

    r_eff = (effective_area/np.pi)**0.5
    small_r = 0.05
    sherwood = (peclet / 4) * ((r_eff / (small_r + 1)) ** 2)

    print(f"{sherwood=} {true_sherwood=}")

    assert sherwood == pytest.approx(true_sherwood, rel=0.1)


if __name__ == "__main__":
    test_acrivos_vs_simpson(1000)
