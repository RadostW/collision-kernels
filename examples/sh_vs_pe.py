import collision_kernels.collision_kernels
import tqdm
import numpy as np
import matplotlib.pyplot as plt


def clift(Pe):
    return (1/2)*(1+(1+2*Pe)**(1/3))

peclet_values=np.logspace(-1,4,13)

sherwood_values=[]
for Pe in tqdm.tqdm(peclet_values):
    sherwood_values += [collision_kernels.collision_kernels.sherwood_from_peclet(
    Pe,
    small_r=0.05,
    trials=10,
    r_mesh=0.1,
    floor_r=5,
    floor_h=5,)]

pe_args=np.logspace(-1,4.5,500)
sh_clift=clift(pe_args)


plt.loglog(peclet_values, sherwood_values, marker='o', linestyle='None', markersize=8, color='#6a6', label = "Numerical output")

plt.loglog(pe_args, sh_clift, color='#a22', label = "Clift approximation")

plt.xlabel(r'Peclet number ($Pe$)')
plt.ylabel(r'Sherwood number ($Sh$)')
plt.legend()



plt.show()