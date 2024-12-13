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
from cs_modifier import ModIntCrossSections


@click.command()
@click.option('--scale_factor_k','-s', help='list with cs modification kaon, 1 = untuned for const, 0 = untuned for exp')
@click.option('--scale_factor_p','-f', help='list with cs modification pion, 1 = untuned for const, 0 = untuned for exp')
@click.option('--threshold','-t', help='threshold above which modification is applied', default='1.e4')
@click.option('--increase','-i', help='const or exp', default='const')
@click.option('--interactionmodel','-m', help='hadr. interaction model', default="SIBYLL2.3c")

def main(scale_factor_p, scale_factor_k, threshold,increase,interactionmodel):


    scale_factor_p = float(f'{scale_factor_p}')
    scale_factor_k = float(f'{scale_factor_k}')
    threshold = float(f'{threshold}')
    increase = f'{increase}'

    scale_factor = [scale_factor_p,scale_factor_k]


    mceq_air = MCEqRun(
        interaction_model="SIBYLL23C",
        theta_deg=0.0,
        primary_model=(pm.GlobalSplineFitBeta, None),
        density_model = ('MSIS00_IC',('South Pole','January')),
    )

    mceq_air.set_interaction_model("SIBYLL23C", force=True)
    mceq_air.set_theta_deg(0)

    cos_thetas = np.arange(0.8, 1.001, 0.1)
    thetas = np.degrees(np.arccos(cos_thetas))

    #initialize mceq instances
    mceq_tune = MCEqRun(
        interaction_model="SIBYLL23C",
        theta_deg=0.0,
        primary_model=(pm.GlobalSplineFitBeta, None),
        density_model = (('MSIS00_IC',('SouthPole','January')))
    )


    # modify cross section
    modcs = ModIntCrossSections(mceq_air._mceq_db, interaction_model=interactionmodel, scale_factor=scale_factor,threshold=threshold,increase=increase) #scale_factor=[1.,1.3],threshold=1.e4,increase='const')
    modcs.load(interaction_model=interactionmodel)

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
            mceq_tune.set_single_primary_particle(E=eprim, pdg_id=2212)
            for ia, theta in enumerate(thetas):  # Loop over angles
                mceq_tune.set_theta_deg(theta)
                
                # Optionally set the season-dependent atmospheric model here
                mceq_tune.density_model.set_season(season)  # Uncomment and adjust if applicable
                
                mceq_tune.solve()
                theta_energies[ia, :] = mceq_tune.get_solution("mu+", mag=0) + mceq_tune.get_solution("mu-", mag=0)
            
            season_energies.append(theta_energies)
        # Append the results for the current season
        ground_muspec_prim_energies.append(season_energies)

    pickle.dump(
        [mceq_air.e_grid, cos_thetas, cr_grid, ground_muspec_prim_energies[0], ground_muspec_prim_energies[1], ground_muspec_prim_energies[2]],
        open("/hetghome/khymon/cs-files/ground_muspec_prim_energies_season_cstune_pi" + str(scale_factor_p) + "_k" + str(scale_factor_k) + "_" +str(threshold)+ str(increase)+ ".pkl", "wb"),
    )
    pickle.dump(
        [mceq_air.e_grid, cos_thetas, surface_flux_GSF[0], surface_flux_GSF[1] , surface_flux_GSF[2]],
        open("/hetghome/khymon/cs-files/surface_fluxes_season_pi" + str(scale_factor_p) + "_k" + str(scale_factor_k) + "_" +str(threshold)+ str(increase)+ ".pkl", "wb"),
    )

    
if __name__ == '__main__':
    main()