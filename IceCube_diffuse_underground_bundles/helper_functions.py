import numpy as np
import crflux.models as pm
import  mceq_underground_helper_forsplines as helper

def sl_depth(d,angle):
    '''
    get slant dpeth from depth [km] and angle [deg]
    '''
    return d/np.cos(np.deg2rad(angle))