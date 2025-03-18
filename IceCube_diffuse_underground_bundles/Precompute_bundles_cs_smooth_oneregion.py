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
from cs_modifier_with_smooth_transition_oneregion import ModIntCrossSections


@click.command()
@click.option('--scale_factor_p','-a', help='cs modification for region 1 pion, 1 = untuned for const, 0 = untuned for exp') 
@click.option('--scale_factor_k','-c', help='cs modification for region 1 pion, 1 = untuned for const, 0 = untuned for exp') 
@click.option('--e0','-l', help='threshold above which 1st modification is applied')
@click.option('--e1', '-t', help='Threshold above which 2nd modification is applied (optional)', type=float, default=None)
@click.option('--increase','-i', help='const or exp', default='const')
@click.option('--interactionmodel','-m', help='hadr. interaction model', default="SIBYLL2.3c")
@click.option('--nucleus','-n', help='proton: 2212 or iron: 5626', default="2212")


def main(scale_factor_p, scale_factor_k, e0,e1, increase,interactionmodel,nucleus):



    scale_factor_p = float(f'{scale_factor_p}')
    scale_factor_k = float(f'{scale_factor_k}')
    
    e0 = float(f'{e0}')
    increase = f'{increase}'
    nucleus = int(f'{nucleus}')

    # convert to list for input into class
    scale_factor_region1 = [scale_factor_p,scale_factor_k]
  

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

    cos_thetas = np.linspace(0.5, 1.0, num=6)
    thetas = np.degrees(np.arccos(cos_thetas))

    # modify cross section
    modcs = ModIntCrossSections(mceq_air._mceq_db, interaction_model="SIBYLL2.3c", scale_factor_region1=scale_factor_region1, e0=e0, e1=e1,  increase='const') 

    mceq_tune._int_cs = modcs # add modification to cross section in mceq instance
    mceq_tune.set_interaction_model(interactionmodel, force=True) # necessary to force cross section change

    #test if tuning is correct:
    print('ratio pion cross section tuned/untuned: ', modcs.get_cs(211, mbarn=True)/InteractionCrossSections(mceq_air._mceq_db, interaction_model=interactionmodel).get_cs(211, mbarn=True))
    print('ratio kaon cross section tuned/untuned: ', modcs.get_cs(321, mbarn=True)/InteractionCrossSections(mceq_air._mceq_db, interaction_model=interactionmodel).get_cs(321, mbarn=True))
    
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
    if e1 is None:
        if scale_factor_p == 1.0 and scale_factor_k == 1.0:
            pickle.dump(
                [mceq_air.e_grid, cos_thetas, cr_grid, ground_muspec_prim_energies[0], ground_muspec_prim_energies[1], ground_muspec_prim_energies[2]],
                open(f"/hetghome/khymon/cs-files/smooth-transition/ground_muspec_prim_energies_season_cstune{nucleus}"
                    f"_pi{float(scale_factor_p):.2f}"
                    f"_k{float(scale_factor_k):.2f}.pkl", "wb"),
                    
            )
            pickle.dump(
                [mceq_air.e_grid, cos_thetas, surface_flux_GSF[0], surface_flux_GSF[1], surface_flux_GSF[2]],
                open(f"/hetghome/khymon/cs-files/smooth-transition/surface_fluxes_season{nucleus}"
                    f"_pi{float(scale_factor_p):.2f}"
                    f"_k{float(scale_factor_k):.2f}.pkl", "wb"),
            )

        else:
            pickle.dump(
                [mceq_air.e_grid, cos_thetas, cr_grid, ground_muspec_prim_energies[0], ground_muspec_prim_energies[1], ground_muspec_prim_energies[2]],
                open(f"/hetghome/khymon/cs-files/smooth-transition/ground_muspec_prim_energies_season_cstune{nucleus}"
                    f"_pi{float(scale_factor_p):.2f}"
                    f"_k{float(scale_factor_k):.2f}"
                    f"_e0{float(e0):.2f}_{increase}.pkl", "wb"),
            
                    
            )
            pickle.dump(
                [mceq_air.e_grid, cos_thetas, surface_flux_GSF[0], surface_flux_GSF[1], surface_flux_GSF[2]],
                open(f"/hetghome/khymon/cs-files/smooth-transition/surface_fluxes_season{nucleus}"
                    f"_pi{float(scale_factor_p):.2f}"
                    f"_k{float(scale_factor_k):.2f}"
                    f"_e0{float(e0):.2f}_{increase}.pkl", "wb"),
            )
    else: 
        pickle.dump(
            [mceq_air.e_grid, cos_thetas, cr_grid, ground_muspec_prim_energies[0], ground_muspec_prim_energies[1], ground_muspec_prim_energies[2]],
            open(f"/hetghome/khymon/cs-files/smooth-transition/ground_muspec_prim_energies_season_cstune{nucleus}"
                f"_pi{float(scale_factor_p):.2f}"
                f"_k{float(scale_factor_k):.2f}"
                f"_e0{float(e0):.2f}_{increase}"
                f"_e1{float(e1):.2f}.pkl", "wb"),
                
        )
        pickle.dump(
            [mceq_air.e_grid, cos_thetas, surface_flux_GSF[0], surface_flux_GSF[1], surface_flux_GSF[2]],
            open(f"/hetghome/khymon/cs-files/smooth-transition/surface_fluxes_season{nucleus}"
                f"_pi{float(scale_factor_p):.2f}"
                f"_k{float(scale_factor_k):.2f}"
                f"_e0{float(e0):.2f}_{increase}"
                f"_e1{float(e1):.2f}.pkl", "wb"),
                
        )


    
if __name__ == '__main__':
    main()