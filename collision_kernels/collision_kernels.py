import collision_kernels.generate_trajectories
import collision_kernels.process_trajectories

def sherwood_from_peclet(pe):
        
    single_pe_trajectories = collision_kernels.generate_trajectories.generate_trajectories(peclet=pe)
    sh_number = collision_kernels.process_trajectories.sherwood(single_pe_trajectories,pe)

    return sh_number