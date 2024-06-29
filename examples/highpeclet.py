import collision_kernels.collision_kernels
import tqdm
import numpy as np
import matplotlib.pyplot as plt

trials = 400
r_mesh = 0.025
floor_r = None
floor_h = 5


filename = "test"


def clift(Pe):
    return (1 / 2) * (1 + (1 + 2 * Pe) ** (1 / 3))


def acrivos(Pe):
    return 1 + (1 / 2) * Pe + (1 / 2) * (Pe**2) * np.log(Pe)


peclet_values = np.logspace(0.5, 6, 30)

sherwood_values_r1 = []
for Pe in tqdm.tqdm(peclet_values):
    sherwood_values_r1 += [
        collision_kernels.collision_kernels.sherwood_from_peclet(
            Pe,
            small_r=0.01,
            trials=trials,
            r_mesh=r_mesh,
            floor_r=None,
            floor_h=floor_h,
        )
    ]

sherwood_values_r5 = []
for Pe in tqdm.tqdm(peclet_values):
    sherwood_values_r5 += [
        collision_kernels.collision_kernels.sherwood_from_peclet(
            Pe,
            small_r=0.05,
            trials=trials,
            r_mesh=r_mesh,
            floor_r=None,
            floor_h=floor_h,
        )
    ]

sherwood_values_r10 = []
for Pe in tqdm.tqdm(peclet_values):
    sherwood_values_r10 += [
        collision_kernels.collision_kernels.sherwood_from_peclet(
            Pe,
            small_r=0.1,
            trials=trials,
            r_mesh=r_mesh,
            floor_r=None,
            floor_h=floor_h,
        )
    ]

sherwood_values_r20 = []
for Pe in tqdm.tqdm(peclet_values):
    sherwood_values_r20 += [
        collision_kernels.collision_kernels.sherwood_from_peclet(
            Pe,
            small_r=0.2,
            trials=trials,
            r_mesh=r_mesh,
            floor_r=None,
            floor_h=floor_h,
        )
    ]

sherwood_values_r50 = []
for Pe in tqdm.tqdm(peclet_values):
    sherwood_values_r50 += [
        collision_kernels.collision_kernels.sherwood_from_peclet(
            Pe,
            small_r=0.5,
            trials=trials,
            r_mesh=r_mesh,
            floor_r=None,
            floor_h=floor_h,
        )
    ]

to_export = zip(peclet_values, sherwood_values_r1)

with open(filename + "r1.txt", "w") as file:
    for point in to_export:
        file.write(f"{point[0]}\t{point[1]}\n")


to_export = zip(peclet_values, sherwood_values_r5)

with open(filename + "r5.txt", "w") as file:
    for point in to_export:
        file.write(f"{point[0]}\t{point[1]}\n")

to_export = zip(peclet_values, sherwood_values_r10)

with open(filename + "r10.txt", "w") as file:
    for point in to_export:
        file.write(f"{point[0]}\t{point[1]}\n")


to_export = zip(peclet_values, sherwood_values_r20)

with open(filename + "r20.txt", "w") as file:
    for point in to_export:
        file.write(f"{point[0]}\t{point[1]}\n")

to_export = zip(peclet_values, sherwood_values_r50)

with open(filename + "r50.txt", "w") as file:
    for point in to_export:
        file.write(f"{point[0]}\t{point[1]}\n")

pe_args = np.logspace(-1, 6, 600)
sh_clift = clift(pe_args)
sh_acrivos = clift(pe_args)


plt.loglog(
    peclet_values,
    sherwood_values_r1,
    marker="o",
    linestyle="None",
    markersize=4,
    color="#a66",
    label=r"Numerical output $r=1 [1]$",
)
plt.loglog(
    peclet_values,
    sherwood_values_r5,
    marker="o",
    linestyle="None",
    markersize=4,
    color="#6a6",
    label=r"Numerical output $r=5 [1]$",
)
plt.loglog(
    peclet_values,
    sherwood_values_r10,
    marker="o",
    linestyle="None",
    markersize=4,
    color="#66a",
    label=r"Numerical output $r=10 [1]$",
)
plt.loglog(
    peclet_values,
    sherwood_values_r20,
    marker="o",
    linestyle="None",
    markersize=4,
    color="#6aa",
    label=r"Numerical output $r=20 [1]$",
)
plt.loglog(
    peclet_values,
    sherwood_values_r50,
    marker="o",
    linestyle="None",
    markersize=4,
    color="#aa6",
    label=r"Numerical output $r=50 [1]$",
)

plt.loglog(pe_args, sh_clift, color="#a22", label=r"Clift")
# plt.loglog(pe_args, sh_acrivos, color='#22a', label = r"Acrivos \& Taylor")

plt.xlabel(r"Peclet number ($Pe$)")
plt.ylabel(r"Sherwood number ($Sh$)")
plt.legend()

plt.savefig(filename + ".png")
