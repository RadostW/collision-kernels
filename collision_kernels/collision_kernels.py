import collision_kernels.generate_trajectories
import collision_kernels.process_trajectories

def sherwood_from_peclet(
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

    sh_number = collision_kernels.process_trajectories.sherwood(single_pe_trajectories,peclet,small_r=small_r)

    return sh_number