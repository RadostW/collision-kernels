import collision_kernels.collision_kernels
import tqdm
import numpy as np
import matplotlib.pyplot as plt

trials = 500
r_mesh = 0.005
floor_r = None
floor_h = 5


filename = "sh_vs_r_small"


def clift(Pe, list):
    zmienna = list
    return (zmienna/zmienna)*(1 / 2) * (1 + (1 + 2 * Pe) ** (1 / 3))


def acrivos(Pe):
    return 1 + (1 / 2) * Pe + (1 / 2) * (Pe**2) * np.log(Pe)


r_values = np.logspace(-3, 0, 30)

sherwood_values_Pe_2 = []
for small_r in tqdm.tqdm(r_values):
    sherwood_values_Pe_2 += [
        collision_kernels.collision_kernels.sherwood_from_peclet(
            10**2,
            small_r=small_r,
            trials=trials,
            r_mesh=r_mesh,
            floor_r=None,
            floor_h=floor_h,
        )
    ]

sherwood_values_Pe_3 = []
for small_r in tqdm.tqdm(r_values):
    sherwood_values_Pe_3 += [
        collision_kernels.collision_kernels.sherwood_from_peclet(
            10**3,
            small_r=small_r,
            trials=trials,
            r_mesh=r_mesh,
            floor_r=None,
            floor_h=floor_h,
        )
    ]

sherwood_values_Pe_4 = []
for small_r in tqdm.tqdm(r_values):
    sherwood_values_Pe_4 += [
        collision_kernels.collision_kernels.sherwood_from_peclet(
            10**4,
            small_r=small_r,
            trials=trials,
            r_mesh=r_mesh,
            floor_r=None,
            floor_h=floor_h,
        )
    ]


to_export = zip(r_values, sherwood_values_Pe_2)

with open(filename + "Pe2.txt", "w") as file:
    for point in to_export:
        file.write(f"{point[0]}\t{point[1]}\n")


to_export = zip(r_values, sherwood_values_Pe_3)

with open(filename + "Pe3.txt", "w") as file:
    for point in to_export:
        file.write(f"{point[0]}\t{point[1]}\n")

to_export = zip(r_values, sherwood_values_Pe_4)

with open(filename + "Pe4.txt", "w") as file:
    for point in to_export:
        file.write(f"{point[0]}\t{point[1]}\n")


r_args = np.logspace(-3, 0, 300)
sh_Pe2 = clift(10**2, r_args)
sh_Pe3 = clift(10**3, r_args)
sh_Pe4 = clift(10**4, r_args)


plt.loglog(
    r_values,
    sherwood_values_Pe_2,
    marker="o",
    linestyle="None",
    markersize=4,
    color="#a66",
    label=r"Numerical output $Pe=10^2$ [1]",
)
plt.loglog(
    r_values,
    sherwood_values_Pe_3,
    marker="o",
    linestyle="None",
    markersize=4,
    color="#6a6",
    label=r"Numerical output $Pe=10^3$ [1]",
)
plt.loglog(
    r_values,
    sherwood_values_Pe_4,
    marker="o",
    linestyle="None",
    markersize=4,
    color="#66a",
    label=r"Numerical output $Pe=10^4$ [1]",
)

plt.loglog(r_args, sh_Pe2, color="#a22", label=r"Clift, $Pe=10^2$ [1]")
plt.loglog(r_args, sh_Pe3, color="#2a2", label=r"Clift, $Pe=10^3$ [1]")
plt.loglog(r_args, sh_Pe4, color="#22a", label=r"Clift, $Pe=10^4$ [1]")
# plt.loglog(pe_args, sh_acrivos, color='#22a', label = r"Acrivos \& Taylor")

plt.xlabel(r"Radius of syfka ($r$) [1]")
plt.ylabel(r"Sherwood number ($Sh$)")
plt.legend()

plt.savefig(filename + ".png")
