import collision_kernels.generate_trajectories
import collision_kernels.process_trajectories

def Sherwood_vs_peclet(Pe):
        
    single_pe_trajectories = collision_kernels.generate_trajectories.generate_trajectories(peclet=Pe)

    sh_number = collision_kernels.process_trajectories.sherwood(single_pe_trajectories)

    return sh_number


peclet_values=[1,2,5]


sherwood_values=[]
for Pe in peclet_values:
    sherwood_values += [Sherwood_vs_peclet(Pe)]

print(list(zip(peclet_values,sherwood_values)))