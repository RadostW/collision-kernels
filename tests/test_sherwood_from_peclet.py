import collision_kernels.collision_kernels
import numpy as np
import pytest


@pytest.mark.parametrize(
    "pe,true_sh",
    [
        (1,1.22),
        (5,1.61),
        (10,1.88),       
    ],
)
def test_sherwood_from_peclet(pe,true_sh):

    test_sh = collision_kernels.collision_kernels.sherwood_from_peclet(pe)

    assert test_sh == pytest.approx(true_sh, abs=0.01)


if __name__ == "__main__":
    test_sherwood_from_peclet(1,1.22)