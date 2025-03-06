import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.lines as mlines
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm
import pickle
import numpy as np

import crflux.models as pm
import mceq_underground_helpers_oneregion as mh

import click

def X(d):
    ''' calculate slant depth ad a given vertical depth in km'''
    s_d = mh.slant_depths
    angles = mh.angles
 
    return d/np.cos(np.deg2rad(angles))

def dNmu_dmu(d,month, ptype, cs_p, cs_k, e0,e1=None): # month = str
    '''
    calculate muon flux per multiplicity

    Params
    -------
    d : float
        Vertical depth in km: 1.5 or 3.5
    
    month: string
        jan, apr or jul

    ptype : int
        particle id: 2212 = p, 5626 = Fe

    cs : float  > 0
        Pion-air cross section modification factor. 1 = default (from Sibyll2.3c).


    '''
    angle = 0.0

    #get depth according to angle
    x_mod = X(d)

    #get number of muons per multiplicity
    dNmudmu = np.ndarray(shape=(len(x_mod),101),dtype='float')     
    for i in range(len(x_mod)):

        dNmudmu[i] = 1e4*mh.mult_dist(x_mod[i],
                                    0.,
                                    pm.GlobalSplineFitBeta(),"yields_" +month,
                                    ptype,
                                    cs_p, cs_k, e0,e1,
                                    norm=False
                                ) / mh.rates(x_mod[i], angle, month, ptype, cs_p, cs_k, e0,e1)
    return dNmudmu

def R(m,dN_dNmu):
    ''' 
    calculate R from dN_mu from unmodified and modofied cs  

    Params:
    ----------
    m : array
        multiplicity vector
    dN_dNmu : array
        muon flux per multiplicity
        '''
    int_low = np.zeros(len(dN_dNmu[:,0]))
    int_high = np.zeros(len(dN_dNmu[:,0]))
    for i in range(len(dN_dNmu[:,0])):
        int_low[i] = np.trapezoid(dN_dNmu[i,1:19],m[1:19])
        int_high[i] = np.trapezoid(dN_dNmu[i,59:],m[59:])    

    return int_high / int_low

def R_normalized(m,R_mod,d,ptype):
    '''
        Normalization of R to april as default atmsophere and Sibyll2.3c

    '''
    # default parameters
    dNu_dmu_apr = dNmu_dmu(d,month="apr", ptype=ptype, cs_p=1.0, cs_k=1.0, e0=3.65) #default cs
    R_def_apr = R(m,dNu_dmu_apr)
    
    return R_mod/R_def_apr

def R_normalized_threshold(m,R_mod,d,ptype,e0,e1=None):
    '''
        Normalization of R to april as default atmsophere and Sibyll2.3c
        e0 is set according to the calculation step to find optimal e0

    '''
    # default parameters
    dNu_dmu_apr = dNmu_dmu(d,month="apr", ptype=ptype, cs_p=1.0, cs_k=1.0, e0=e0,e1=e1) #default cs
    R_def_apr = R(m,dNu_dmu_apr)
    
    return R_mod/R_def_apr


@click.command()
@click.option('--calculation','-c', help='k-pi, threshold, general')

def main(calculation):
    '''
        k-pi: only iterate between k2 and p2
        threshold: change only e0
        general: detailed change of p1 and p2 in different combinations
    '''
  
    calc_tag = f'{calculation}'  

    m = mh.n_mu_vec # muon multiplicity

    if calc_tag == 'k-pi':
        cs_p_values = [0.8,0.9,0.99,1.0,1.01,1.1,1.2] 
        cs_k_values = [0.8,0.9,0.99,1.0,1.01,1.1,1.2]     
        ptype_values = [2212] 
        season_values = ["jan", "apr", "jul"] #  seasons
        e0_values = [3.65]
        e1_values = None
        

    elif calc_tag == 'threshold':
               
        cs_p_values = [0.99,1.00,1.01] #[0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]  # List of cross-section values: pion-air
        cs_k_values = [1.00]
        ptype_values = [2212] 
        season_values = ["jan", "apr", "jul"]  #  seasons
        e0_values = [2.05,2.15,2.25,2.35,2.45,2.55,2.65,2.75,2.85,2.95,3.05,3.15,3.25,3.35,3.45,3.55,3.65,3.75,3.85,3.95,4.05,4.15,4.25,4.35,4.45,4.55,4.65,4.75,4.85,4.95]
        e1_values = None

    elif calc_tag =='energybin_test':
        cs_p_values = [1.00,1.01] #[0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]  # List of cross-section values: pion-air
        cs_k_values = [1.00]
        ptype_values = [2212] 
        season_values = ["jan", "apr", "jul"]  #  seasons
        e0_values = [2.05,2.15,2.25,2.35,2.45,2.55,2.65,2.75,2.85,2.95,3.05,3.15,3.25,3.35,3.45,3.55,3.65,3.75,3.85,3.95,4.05,4.15,4.25,4.35,4.45,4.55,4.65,4.75,4.85]
        e1_values = [2.25,2.35,2.45,2.55,2.65,2.75,2.85,2.95,3.05,3.15,3.25,3.35,3.45,3.55,3.65,3.75,3.85,3.95,4.05,4.15,4.25,4.35,4.45,4.55,4.65,4.75,4.85,4.95,5.05]
        pairwise = True

    

    # initlialize helper   
    mh.initialize_flux_dicts(
        ptype_values, cs_p_values, cs_k_values, e0_values, e1_values if 'e1_values' in locals() else [None],pairwise=pairwise
    ) 
    mh.some_function_that_uses_angles()
    mh.some_function_that_uses_c_wi()
    
        
    #dictionary
    results = {}  # Dictionary to store the results
    d_values = [1.5, 3.5]# detectpr depth: 1.5 or 3.5km

    for d in d_values:  
       
        x_mod = X(d) # for specific depth  

        
        for cs_p in cs_p_values:
            for cs_k in cs_k_values:
                for ptype in ptype_values:
                    for season in season_values:
                        if cs_p == 1.0:
                            # Special case when cs_p = 1.0: handle the case where e0 is given and e1 is None
                            for e0 in e0_values:
                                # Call functions to compute R with special handling for cs_p=1.0
                                dNmu_dmu_mod = dNmu_dmu(d, season, ptype, cs_p, cs_k, e0)
                                R_mod = R(m, dNmu_dmu_mod)
                                if calc_tag == 'threshold':
                                    R_norm = R_normalized_threshold(m, R_mod, d, ptype, e0)
                                    print('e0 changed in normalization of R')
                                else:
                                    R_norm = R_normalized(m, R_mod, d, ptype)

                                # Store the result in the dictionary, use e1='inf' for this special case
                                results[(str(d), str(cs_p), str(cs_k), str(ptype), season, str(e0), "inf")] = R_norm
                        else:
                            # For other cs_p values: handle both pairwise and non-pairwise combinations of e0 and e1
                            for e0, e1 in zip(e0_values, e1_values if e1_values is not None else [None]):
                                # Call functions to compute R for each pair of e0 and e1
                                dNmu_dmu_mod = dNmu_dmu(d, season, ptype, cs_p, cs_k, e0)
                                R_mod = R(m, dNmu_dmu_mod)
                                if calc_tag == 'threshold':
                                    R_norm = R_normalized_threshold(m, R_mod, d, ptype, e0)
                                    print('e0 changed in normalization of R')
                                else:
                                    R_norm = R_normalized(m, R_mod, d, ptype)

                                # Store the result in the dictionary with both e0 and e1 (or "inf" if e1 is None)
                                results[(str(d), str(cs_p), str(cs_k), str(ptype), season, str(e0), str(e1) if e1 is not None else "inf")] = R_norm

    with open("/hetghome/khymon/cs-files/R_value_const_pi-air_k-air_sibyll23c_smooth_oneregion_" + str(calc_tag) + ".pkl", "wb") as f:
        pickle.dump(results, f)




if __name__ == '__main__':
    main()

    
