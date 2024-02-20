import numpy as np

def clift(peclet):
    return (1/2)*(1+(1+2*peclet)**(1/3))

def area_from_clift(peclet, small_r):
    return ((4*np.pi*(1+small_r)**2 )*clift(peclet)/peclet)

def sherwood_from_area(effective_area, small_r):
    r_eff = (effective_area/np.pi)**0.5
    return (peclet / 4) * ((r_eff / (small_r + 1)) ** 2)