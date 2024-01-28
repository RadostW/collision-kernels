import collision_kernels.process_trajectories
import collision_kernels.generate_trajectories
import numpy as np
import matplotlib.pyplot as plt


def visualise_trajectories():
    peclet = 10
    floor_r = 5
    floor_h = 10
    r_mesh = 0.2
    trials = 10
    small_r = 1.0
    display_traj = 20

    initial = collision_kernels.generate_trajectories.construct_initial_condition(
        floor_r=floor_r, floor_h=floor_h, r_mesh=r_mesh, trials=trials
    )

    collision_data = collision_kernels.generate_trajectories.simulate_until_collides(
        drift=collision_kernels.generate_trajectories.stokes_around_unit_sphere,
        noise=collision_kernels.generate_trajectories.diffusion_function(peclet=peclet),
        initial=initial,
        small_r=small_r,
        floor_h=floor_h,        
    )

    plt.figure(figsize=(12, 10))

    trajectories = collision_data["trajectories"]
    for i in range(display_traj):
        r = (trajectories[i, :, 0] ** 2 + trajectories[i, :, 1] ** 2) ** 0.5
        z = trajectories[i, :, -1]

        if collision_data["ball_hit"][i]:
            color = "#2a2"
        elif collision_data["something_hit"][i]:
            color = "#aa2"
        else:
            color = "#a22"

        plt.plot(r, z, color=color)
        plt.scatter(r[-1], z[-1], s=8, color="k", zorder=5)
        plt.scatter(r[0], z[0], s=8, color="k", zorder=5)

    circle = plt.Circle((0, 0), 1, color="#222", fill=False)
    plt.gca().add_artist(circle)
    circle = plt.Circle((0, 0), 1 + small_r, color="#666", fill=False)
    plt.gca().add_artist(circle)

    plt.gca().set_aspect("equal")
    plt.axis([0, 5, -10.5, 10])

    plt.savefig("traj_visualize.svg", format='svg')

if __name__ == "__main__":
    visualise_trajectories()
