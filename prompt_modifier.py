import numpy as np
import matplotlib.pyplot as plt
import mceq_config as config
from MCEq.core import MCEqRun
import crflux.models as pm

from utils.helpers import *
from utils import config_handler
import click
import os


def runmceq(mceq_object,ptype,doys,angles,pr_factor): #create 
    '''
        Calculation of MCEq flux

        ptype: string (numu, nue or mu)
        mceq_object: mceq object with or without modified settings. MSIS00 atmosphere is required for the calculation of daily flux
        doys: list (int) of days in year from 1 to 366
        angles: list (float), supported angles between 0 and 180
        pr_factor : float, increase/decrease prompt flux component
    '''

    mag = 0
    flux = [] #np.ndarray(shape=(len(angles),len(doys),121))

    for sim_angle in angles:
        mceq_object.set_theta_deg(sim_angle)
        flux.append([])
        for d in doys:
            mceq_object.density_model.set_doy(d)
            mceq_object.density_model.calculate_density_spline()
            mceq_object._calculate_integration_path(int_grid=None, grid_var='X',force=True)
            mceq_object.solve()
            if ptype == 'numu':
                flux[-1].append(mceq_object.get_solution('conv_numu', mag) + mceq_object.get_solution('conv_antinumu' ,mag) + pr_factor * (mceq_object.get_solution('pr_numu', mag) + mceq_object.get_solution('pr_antinumu' ,mag)))
            elif ptype == 'mu':
                flux[-1].append(mceq_object.get_solution('conv_mu+', mag) + mceq_object.get_solution('conv_mu-' ,mag) + pr_factor * (mceq_object.get_solution('pr_mu+', mag) + mceq_object.get_solution('pr_mu-' ,mag)))
            else:
                print('particle type is not defined.')
                exit()

    return np.array(flux)

@click.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--ptype', '-p',help='Enter dataset ptype: numu, mu', default ='numu') 
@click.option('--pr_factor','-f', help='prompt flux factor')

def main(config_file,ptype, pr_factor):

    # get arguments 
    ptype = f'{ptype}'
    pr_factor = np.float(f'{pr_factor}')

    # read folders from config
    conf_folders = config_handler.get_json_config(str(config_file))
    flux_dir = conf_folders['fluxes']
    os.makedirs(flux_dir, exist_ok=True)

    #initialize mceq instances
    interactionmodel = 'SIBYLL2.3c'

    mceq= MCEqRun(
        interaction_model=interactionmodel,
        theta_deg=0.0,
        primary_model=(pm.HillasGaisser2012, "H3a"),
        density_model = (('MSIS00_IC',('SouthPole','January')))
    )
    
    # define angles and days for flux calculation
    angles = angular_bins(ptype, 2)
    doy = doys(5) # take every 5th day for fast testing purposes

    # calculate flux
    flux = runmceq(mceq,ptype,doy,angles,pr_factor)

    # save file
    np.save(flux_dir + ptype + '_' + str(pr_factor) + 'prompt_' + interactionmodel+'_mceqflux.npy',flux) 



    
if __name__ == '__main__':
    main()