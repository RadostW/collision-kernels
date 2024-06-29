import collision_kernels.collision_kernels
import tqdm
import numpy as np
import matplotlib.pyplot as plt


def clift(Pe):
    return (1/2)*(1+(1+2*Pe)**(1/3))

peclet_values_high=np.logspace(4,6,13)
peclet_values_small=np.logspace(0,4,13)

sherwood_values=[]
for Pe in tqdm.tqdm(peclet_values_small):
    sherwood_values += [collision_kernels.collision_kernels.sherwood_from_peclet(
    Pe,
    small_r=0.1,
    trials=30,
    r_mesh=0.05,
    floor_r=5,
    floor_h=5,)]

for Pe in tqdm.tqdm(peclet_values_high):
    sherwood_values += [collision_kernels.collision_kernels.sherwood_from_peclet(
    Pe,
    small_r=0.1,
    trials=30,
    r_mesh=0.005,
    floor_r="highPe",
    floor_h=5,)]

pe_args=np.logspace(-1,6,500)
sh_clift=clift(pe_args)

plt.loglog(np.concatenate((peclet_values_small,peclet_values_high)), sherwood_values, marker='o', linestyle='None', markersize=8, color='#6a6', label = "Numerical output")

plt.loglog(pe_args, sh_clift, color='#a22', label = "Clift approximation")

plt.xlabel(r'Peclet number ($Pe$)')
plt.ylabel(r'Sherwood number ($Sh$)')
plt.legend()



plt.show()