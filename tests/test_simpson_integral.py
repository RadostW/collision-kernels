import collision_kernels.integrate_with_weight
import numpy as np
import numpy.random

import pytest



@pytest.mark.parametrize(
    "steepness,true_result",
    [
        (0.01,8294.24),
        (0.05,343.1),        
        (0.1,89.43),
        (0.5,4.946),
        (1,1.806),
    ],
)
def test_integrate_with_weight(steepness,true_result):

    def weight(x):
        return 2 * np.pi * x

    def sampler(x,n):
        """
        Location x
        Number of samples n
        """
        def p(x):
            return 1/(1+np.exp(steepness*(-1+x)))
        
        return np.random.binomial(1, p(x), n)

    result = collision_kernels.integrate_with_weight(weight=weight,sampler=sampler)
    
    assert np.allclose(
        result,
        true_result,
        rtol=0.1,
    )

if __name__ == "__main__":
    test_integrate_with_weight()
