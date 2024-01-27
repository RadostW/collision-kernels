import collision_kernels.collision_kernels
import tqdm


peclet_values=[1,2,5]

sherwood_values=[]
for Pe in tqdm.tqdm(peclet_values):
    sherwood_values += [collision_kernels.collision_kernels.sherwood_from_peclet(Pe)]

print(list(zip(peclet_values,sherwood_values)))