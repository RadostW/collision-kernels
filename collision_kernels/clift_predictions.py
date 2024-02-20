import numpy as np

def sherwood(peclet):
    return (1/2)*(1+(1+2*peclet)**(1/3))


def effective_area(peclet):
    return ((4*np.pi*(1)**2 )*sherwood(peclet)/peclet)