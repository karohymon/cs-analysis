import click
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.lines as mlines
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm
from MCEq.core import MCEqRun
import mceq_config as config
import crflux.models as pm

import sys
import os
import pickle





def main():


   
    mceq = MCEqRun(
        interaction_model="SIBYLL23C",
        theta_deg=0.0,
        primary_model=(pm.GlobalSplineFitBeta, None),
        density_model = ('MSIS00_IC',('South Pole','January')),
    )

    mceq.set_interaction_model("SIBYLL23C", force=True)
    mceq.set_theta_deg(0)

    cos_thetas = np.arange(0.5, 1.001, 0.1)
    thetas = np.degrees(np.arccos(cos_thetas))

    # calcualte tuned flux for different seasons
    


    flux_k = []  #muon flux from K component
    flux_pi = []
    flux_conv= []
    flux_pr = []
    flux_tot = []

    for ia, theta in enumerate(thetas):  # Loop over angles
        mceq.set_theta_deg(theta)
        
        # Optionally set the season-dependent atmospheric model here
        mceq.density_model.set_season('April')  
        
        # Solve for the current angle and season
        mceq.solve()
        
        # Append the solution for the current angle
        flux_k.append(
            mceq.get_solution("k_mu+", mag=0) + mceq.get_solution("k_mu-", mag=0)
        )
        flux_pi.append(
            mceq.get_solution("pi_mu+", mag=0) + mceq.get_solution("pi_mu-", mag=0)
        )
        flux_conv.append(
            mceq.get_solution("conv_mu+", mag=0) + mceq.get_solution("conv_mu-", mag=0)
        )
        flux_pr.append(
            mceq.get_solution("pr_mu+", mag=0) + mceq.get_solution("pr_mu-", mag=0)
        )
        flux_tot.append(
            mceq.get_solution("total_mu+", mag=0) + mceq.get_solution("total_mu-", mag=0)
        )


 

  

    #fluxes need to be saved seaparately for each particle type


    pickle.dump(
        [mceq.e_grid, cos_thetas, flux_tot, flux_conv , flux_pr, flux_k, flux_pi ],
        open("/hetghome/khymon/cs-files/surface_fluxes_parent_apr_GSF_Sibyll23c.pkl", "wb"),
    )

    
if __name__ == '__main__':
    main()