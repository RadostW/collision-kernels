def generate_trajectories(peclet=1, small_r=0.05, trials=1000, floor_r=10, floor_h=10):
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

    pass
