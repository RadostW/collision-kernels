import collision_kernels.integrate_with_weight
import numpy as np
import numpy.random

import pytest


@pytest.mark.parametrize(
    "steepness,location,true_result",
    [
        # (0.01, 1, 52114.2), # very slow
        # (0.05, 1, 2155.77), # very slow
        (0.1, 1, 561.92),
        (0.5, 1, 31.0821),
        (1, 1, 11.3492),
        (2, 1, 5.51966),
        (5, 1, 3.55332),
        (10, 1, 3.24494),
        # (0.01, 0.1, 51720.7), # very slow
        # (0.05, 0.1, 2075.81), # very slow
        (0.1, 0.1, 521.142),
        (0.5, 0.1, 21.5577),
        (1, 0.1, 5.6192),
        (2, 0.1, 1.52592),
        (5, 0.1, 0.310821),
        (10, 0.1, 0.113492),
    ],
)
def test_integrate_with_weight(steepness, location, true_result):

    def weight(x):
        return 2 * np.pi * x

    def sampler(x, n):
        """
        Location x
        Number of samples n
        """

        def p(x):
            return 1 / (1 + np.exp(steepness * (-location + x)))

        batch_size = 10000
        done = 0
        results = list()
        while done < n:
            batchshes = int(n / batch_size)
            results.append(np.mean(np.random.binomial(1, p(x), batch_size)))
            done = done + batch_size
        ret = np.array(results)
        # print(f"Sampler: {x:5.2f} {np.mean(ret):5.2f}")
        return ret

    absolute_tolerance = true_result * 0.01
    max_step_size = 0.1

    result = collision_kernels.integrate_with_weight.integrate_with_weight(
        weight=weight,
        sampler=sampler,
        absolute_tolerance=absolute_tolerance,
        max_step_size=0.1,
    )
    # print(f"Result: {result:5.2f}")
    # print(f"Error: {result-true_result:e}")
    # print(f"Ratio: {(result-true_result)/absolute_tolerance:e}")

    assert result == pytest.approx(true_result, rel=0.1)


if __name__ == "__main__":
    test_integrate_with_weight(steepness=5, location=1, true_result=5.52)
