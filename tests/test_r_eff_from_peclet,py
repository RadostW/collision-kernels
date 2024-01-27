import tqdm
import numpy as np
import matplotlib.pyplot as plt
import collision_kernels.generate_trajectories
import collision_kernels.process_trajectories

def reff_from_peclet(
    peclet,
    small_r=0.05,
    trials=100,
    r_mesh=0.1,
    floor_r=5,
    floor_h=5,):
        
    single_pe_trajectories = collision_kernels.generate_trajectories.generate_trajectories(
    peclet,
    small_r=small_r,
    trials=trials,
    r_mesh=r_mesh,
    floor_r=floor_r,
    floor_h=floor_h,)

    effective_radius = collision_kernels.process_trajectories.effective_radius(single_pe_trajectories)

    return effective_radius


peclet_values=np.logspace(0,4,4)

sherwood_values=[]
for Pe in tqdm.tqdm(peclet_values):
    sherwood_values += [reff_from_peclet(Pe)]

print(list(zip(peclet_values,sherwood_values)))

plt.loglog(peclet_values, sherwood_values, marker='o', linestyle='None', markersize=8, color='blue')

plt.show()