import pychastic
import jax.numpy as jnp
import numpy as np
import time

big_r = 1
u_inf = jnp.array([0, 0, 1])


def drift(q):
    """
    Given location of the tracer find drift velocity -- Stokes flow around sphere of size big_r (stationary) and ambient flow u_inf.

    Locaiton is measured from the centre of the sphere.

    Compare:
    https://en.wikipedia.org/wiki/Stokes%27_law#Transversal_flow_around_a_sphere
    """

    abs_x = jnp.sum(q * q) ** 0.5

    xx_scale = (3 * big_r**3) / (4 * abs_x**5) - (3 * big_r) / (4 * abs_x**3)
    id_scale = -(big_r**3) / (4 * abs_x**3) - (3 * big_r) / (4 * abs_x) + 1

    xx_tensor = q[:, jnp.newaxis] * q[jnp.newaxis, :]
    id_tensor = jnp.eye(3)

    return (xx_scale * xx_tensor + id_scale * id_tensor) @ u_inf


def generate_trajectories(
    peclet=1, small_r=0.05, trials=1000, floor_r=10, r_rows=10, floor_h=10
):
    """
    Generate trajectories of particles in a simulation.

    Parameters
    ----------
    peclet : float, optional
        Peclet number defined as R u / D.

    small_r : float, optional
        Radius of the smal ball.

    trials : int, optional
        Number of trajectories.

    floor_r : int, optional
        Radius of the floor of trial cyllinder.

    floor_h : int, optional
        Vertical distance from initial condition to big ball centre.

    Returns
    -------
    np.array
        2 by `trials` array containing initial radial distance in
        first coord and outcome 1 - ball hit, 0 - ball missed.

    """
    noise = jnp.sqrt(1 / peclet) * jnp.eye(3)

    whole_time = time.time()

    for r in np.linspace(0, floor_r, r_rows):
        start_time = time.time()

        n = 0
        n_good = []

        tocat = jnp.array([1, 0, 0])[jnp.newaxis, :] * jnp.linspace(r, r, trials)[:, jnp.newaxis] + jnp.array([0, 0, floor_h])

        problem = pychastic.sde_problem.SDEProblem(
            drift,
            noise,
            x0=jnpposes,
            tmax=20.0,
        )

        solver = pychastic.sde_solver.SDESolver(dt=0.01)
        trajectory = solver.solve_many(problem, None, progress_bar=None)

    pass
