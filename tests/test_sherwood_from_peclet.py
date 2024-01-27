import collision_kernels.generate_trajectories
import collision_kernels.process_trajectories

import numpy as np

import pytest

def sherwood_from_peclet(pe):
        
    single_pe_trajectories = collision_kernels.generate_trajectories.generate_trajectories(peclet=pe)
    sh_number = collision_kernels.process_trajectories.sherwood(single_pe_trajectories)

    return sh_number

@pytest.mark.parametrize(
    "pe,true_sh",
    [
        (1,1.22),
        (5,1.61),
        (10,1.88),       
    ],
)
def test_sherwood_from_peclet(pe,true_sh):

    test_sh = sherwood_from_peclet(pe)

    print(f'{test_sh=} {true_sh=}')

    assert test_sh == pytest.approx(true_sh, abs=0.5)


if __name__ == "__main__":
    test_sherwood_from_peclet(1,1.22)