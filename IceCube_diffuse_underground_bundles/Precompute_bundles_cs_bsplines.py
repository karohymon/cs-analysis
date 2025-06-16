'''
    Apply smooth transition of cross section modifications instead of a step function increase or decrease.
'''

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
from MCEq.data import InteractionCrossSections
import sys
import os
import pickle

#os.chdir('/hetghome/khymon/cs-analysis/')
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, parent_dir)
# Import the class
from cs_modifier_bsplines import ModIntCrossSections


@click.command()
@click.option('--scale_factor_a','-a', help='cs modification for region a at 1 TeV') 
@click.option('--scale_factor_b','-b', help='cs modification for region b pion at 10**4.5 GeV') 
@click.option('--scale_factor_c','-c', help='cs modification for region c pion at 100 TeV') 
@click.option('--interactionmodel','-m', help='hadr. interaction model', default="SIBYLL2.3c")
@click.option('--ptype','-p', help='particle id', default="211")


def main(scale_factor_a, scale_factor_b, scale_factor_c,interactionmodel,ptype):


    scale_factor_a = float(f'{scale_factor_a}')
    scale_factor_b = float(f'{scale_factor_b}')
    scale_factor_c = float(f'{scale_factor_c}')
    
    ptype= int(f'{ptype}')

    nucleus = 2212


    #initialize mceq instances
    mceq_tune = MCEqRun(
            interaction_model=interactionmodel,
            theta_deg=0.0,
            primary_model=(pm.HillasGaisser2012, "H3a"),
            density_model = (('MSIS00_IC',('SouthPole','January')))
        )

    mceq_air = MCEqRun(
        interaction_model="SIBYLL23C",
        theta_deg=0.0,
        primary_model=(pm.GlobalSplineFitBeta, None),
        density_model = ('MSIS00_IC',('South Pole','January')),
    )

    mceq_air.set_interaction_model("SIBYLL23C", force=True)
    mceq_air.set_theta_deg(0)

    #cos_thetas = np.linspace(0.5, 1.0, num=6)
    #thetas = np.degrees(np.arccos(cos_thetas))
    thetas = np.array([0., 30., 60.])
    cos_thetas = np.cos(np.radians(thetas))
    

    # modify cross section
    modcs = ModIntCrossSections(mceq_air._mceq_db, interaction_model="SIBYLL2.3c",ptype=ptype, a=scale_factor_a, b=scale_factor_b, c=scale_factor_c) 
    mceq_tune._int_cs = modcs # add modification to cross section in mceq instance
    mceq_tune.set_interaction_model(interactionmodel, force=True) # necessary to force cross section change

    #test if tuning is correct:
    print('ratio ' + str(ptype) + 'cross section tuned/untuned: ', modcs.get_cs(ptype, mbarn=True)/InteractionCrossSections(mceq_air._mceq_db, interaction_model=interactionmodel).get_cs(211, mbarn=True))
    

    # calcualte tuned flux for different seasons
    seasons = ['January', 'April', 'July']
    surface_flux_GSF = []

    for season in tqdm(seasons):  # Loop over seasons
        season_flux = []  # Temporary container for the current season's results
        for ia, theta in enumerate(thetas):  # Loop over angles
            mceq_tune.set_theta_deg(theta)
            
            # Optionally set the season-dependent atmospheric model here
            mceq_tune.density_model.set_season(season)  # Uncomment and adjust if applicable
            
            # Solve for the current angle and season
            mceq_tune.solve()
            
            # Append the solution for the current angle
            season_flux.append(
                mceq_tune.get_solution("mu+", mag=0) + mceq_tune.get_solution("mu-", mag=0)
            )
        # Append the results for the current season
        surface_flux_GSF.append(season_flux)

    config.debug_level = 0
    cr_grid = mceq_air.e_grid[30:-10] 

    # ground muon spectrum calculated for given CR energy
    ground_muspec_prim_energies = []

    for season in tqdm(seasons):  # Loop over seasons
        season_energies = []  # Temporary container for the current season's results
        for ei, eprim in enumerate(tqdm(cr_grid)):  # Loop over primary energies
            theta_energies = np.zeros((thetas.shape[0], mceq_air.dim))
            if nucleus==2212:
                mceq_tune.set_single_primary_particle(E=eprim, pdg_id=nucleus)
            else:
                mceq_tune.set_single_primary_particle(E=eprim, corsika_id=nucleus)
            for ia, theta in enumerate(thetas):  # Loop over angles
                mceq_tune.set_theta_deg(theta)
                
                # Optionally set the season-dependent atmospheric model here
                mceq_tune.density_model.set_season(season)  # Uncomment and adjust if applicable
                
                mceq_tune.solve()
                theta_energies[ia, :] = mceq_tune.get_solution("mu+", mag=0) + mceq_tune.get_solution("mu-", mag=0)
            
            season_energies.append(theta_energies)
        # Append the results for the current season
        ground_muspec_prim_energies.append(season_energies)

    # adapt file names
    
    pickle.dump(
        [mceq_air.e_grid, cos_thetas, cr_grid, ground_muspec_prim_energies[0], ground_muspec_prim_energies[1], ground_muspec_prim_energies[2]],
        open(f"/hetghome/khymon/cs-files/spline_mod_fluxes/ground_muspec_prim_energies_season_cstune{nucleus}_pid{ptype}"
            f"_a{float(scale_factor_a):.2f}"
            f"_b{float(scale_factor_b):.2f}"
            f"_c{float(scale_factor_c):.2f}.pkl", "wb"),
            
    )
    pickle.dump(
        [mceq_air.e_grid, cos_thetas, surface_flux_GSF[0], surface_flux_GSF[1], surface_flux_GSF[2]],
        open(f"/hetghome/khymon/cs-files/spline_mod_fluxes/surface_fluxes_season{nucleus}_pid{ptype}"
            f"_a{float(scale_factor_a):.2f}"
            f"_b{float(scale_factor_b):.2f}"
            f"_c{float(scale_factor_c):.2f}.pkl", "wb"),

    )


    
if __name__ == '__main__':
    main()