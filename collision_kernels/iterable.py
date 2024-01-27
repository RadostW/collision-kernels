import pychastic
import jax.numpy as jnp

# import matplotlib.pyplot as plt
import numpy as np
import time
import os
import sys

Pe_list = []
for n in range(-1, 7):
    for i in [1, 2, 5]:
        Pe_list = Pe_list + [i * 10**n]

cwd = os.getcwd()
big_r = 1
small_r = 0.05
u_inf = jnp.array([0, 0, 1])
Pe = Pe_list[int(sys.argv[1]) - 1]
trials = 3000
X_Max = 4
X_rows = 80
Z_Max = 2
Z_Min = -4
Z_rows = 50

print("calculating for Pe = ", Pe)


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


def noise(q):
    """
    Return diffusion coefficient from location
    """

    return jnp.sqrt(1 / Pe) * jnp.eye(3)


check = np.zeros((X_rows, Z_rows, 3))

m = 0
whole_time = time.time()
for x in np.linspace(0, X_Max, X_rows):
    start_time = time.time()

    poses = np.zeros((1, 3))

    n = 0
    n_good = []
    for z in np.linspace(Z_Min, Z_Max, Z_rows):
        tocat = jnp.array([1, 0, 0])[jnp.newaxis, :] * jnp.linspace(x, x, trials)[
            :, jnp.newaxis
        ] + jnp.array([0, 0, z])
        if z**2 + x**2 > 1:
            poses = np.concatenate((poses, tocat), axis=0)
            check[m, n, 0] = x
            check[m, n, 1] = z
            n_good = n_good + [n]
        n += 1

    poses = poses[1:]
    jnpposes = jnp.array(poses)

    problem = pychastic.sde_problem.SDEProblem(
        drift,
        noise,
        x0=jnpposes,
        tmax=20.0,
    )

    solver = pychastic.sde_solver.SDESolver(dt=0.01)
    trajectory = solver.solve_many(problem, None, progress_bar=None)

    z_num = 0
    for t in trajectory["solution_values"]:
        k = n_good[z_num // trials]
        if_touch = jnp.linalg.norm(t, axis=1) < big_r + small_r
        if_touch = if_touch * 1
        temp = jnp.sum(if_touch) > 0
        check[m, k, 2] += temp / trials
        z_num += 1

    end_time = time.time()

    execution_time = end_time - start_time
    # 	print(f"single step time: {execution_time} seconds")
    m += 1


print(check.shape)

where_o = jnp.linalg.norm(check, axis=2) != 0  # np.array([0,0,0])
to_export = check[where_o]

# x_cont = check[:, :, 0]
# y_cont = check[:, :, 1]
# z_cont = check[:, :, 2]


# Create the contour plot
# plt.contourf(x_cont, y_cont, z_cont, levels=30, cmap='viridis')  # You can adjust 'levels' and 'cmap' as needed

# Add color bar for reference
# plt.colorbar()

# Add labels and title (customize as needed)
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Pe: '+str())

# plt.gca().set_aspect("equal")

# Show the plot

# plt.savefig("rsyf"+str(int((10**3)*small_r))+".png")


# reshaped_data = check.reshape(-1, 3)

file_name = "pe" + str(int(10 * Pe)) + "_e_m1"

if os.path.isdir(cwd + "/" + file_name) != True:
    os.mkdir(cwd + "/" + file_name)

np.savetxt(
    cwd + "/" + file_name + "/" + file_name + ".txt",
    to_export,
    fmt="%f",
    delimiter="\t",
)

endcode_time = time.time()

code_time = endcode_time - whole_time
print(f"One Pe time: {code_time} seconds")

# duration = 1  # seconds
# freq = 440  # Hz
# os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))


# for t in trajectory["solution_values"][::10]:
#   plt.plot(jnp.sqrt(t[:, 0] ** 2 + t[:, 1] ** 2), t[:, 2], color="b", alpha=0.1)

# circle = plt.Circle((0, 0), 1, edgecolor="k", facecolor="none")
# plt.gca().add_patch(circle)
# plt.gca().set_aspect("equal")
# plt.xlim([0, 5])
# plt.ylim([-5, 5])
# plt.show()
