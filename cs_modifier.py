import numpy as np
import matplotlib.pyplot as plt
import mceq_config as config
from MCEq.core import MCEqRun
import crflux.models as pm
from MCEq.data import InteractionCrossSections
from utils.helpers import *
from utils import config_handler
import click
import os


# class for modifying cross section in MCEq
class ModIntCrossSections(InteractionCrossSections):
    def __init__(self, mceq_hdf_db, interaction_model="SIBYLL2.3c", scale_factor=None,threshold =1., increase = 'const'): # first pion then kaon, incrase = const or exp
        self.scale_factor = scale_factor  # Default scaling factor
        self.threshold = threshold
        self.increase = increase
        self.interaction_model = interaction_model
        super().__init__(mceq_hdf_db, interaction_model)  # Call parent constructor

    def modify_cs(self, scale_factor=None,threshold =None, increase = None):
        """ Kaon Pion modification abovethreshold - exp or const increase """
        if scale_factor is None:
            scale_factor = self.scale_factor  # Use the default if not specified
        if threshold is None:
            threshold = self.threshold

        e_range = self.energy_grid.c > threshold
        e0 = self.energy_grid.c[e_range][0]
        print(f"Applying mod with scaling factor: {scale_factor}, threshold e0: {e0}")

        if self.increase == 'exp':
            for p, sf in zip([211, 321], scale_factor):    # Loop over particle types # modify pions
                self.index_d[p][e_range] *= (self.energy_grid.c[e_range] / e0) ** sf

        elif self.increase == 'const':
            for p, sf in zip([211, 321], scale_factor):    
                self.index_d[p][e_range] *= (sf)

    def load(self, interaction_model):
        """ Load the interaction model and apply the modification function """
        super().load(interaction_model)
        self.modify_cs()


def runmceq(mceq_object,ptype,doys,angles): #create 
    '''
        Calculation of MCEq flux

        ptype: string (numu, nue or mu)
        mceq_object: mceq object with or without modified settings. MSIS00 atmosphere is required for the calculation of daily flux
        doys: list (int) of days in year from 1 to 366
        angles: list (float), supported angles between 0 and 180
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
                flux[-1].append(mceq_object.get_solution('total_numu', mag) + mceq_object.get_solution('total_antinumu' ,mag))
            elif ptype == 'nue':
                flux[-1].append(mceq_object.get_solution('total_nue', mag) + mceq_object.get_solution('total_antinue' ,mag))
            elif ptype == 'mu':
                flux[-1].append(mceq_object.get_solution('total_mu+', mag) + mceq_object.get_solution('total_mu-' ,mag))
            else:
                print('particle type is not defined.')
                exit()

    return np.array(flux)

@click.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--ptype', '-p',help='Enter dataset ptype: numu, mu', default ='numu') 
@click.option('--scale_factor_k','-s', help='list with cs modification kaon, 1 = untuned for const, 0 = untuned for exp')
@click.option('--scale_factor_p','-f', help='list with cs modification pion, 1 = untuned for const, 0 = untuned for exp')
@click.option('--threshold','-t', help='threshold above which modification is applied', default='1.e4')
@click.option('--increase','-i', help='const or exp', default='const')
@click.option('--interactionmodel','-m', help='hadr. interaction model', default="SIBYLL2.3c")

def main(config_file,ptype, scale_factor_p, scale_factor_k, threshold,increase,interactionmodel):

    # get arguments 
    ptype = f'{ptype}'
    scale_factor_p = np.float(f'{scale_factor_p}')
    scale_factor_k = np.float(f'{scale_factor_k}')
    threshold = np.float(f'{threshold}')
    increase = f'{increase}'

    scale_factor = [scale_factor_p,scale_factor_k]

    # read folders from config
    conf_folders = config_handler.get_json_config(str(config_file))
    flux_dir = conf_folders['fluxes']
    os.makedirs(flux_dir, exist_ok=True)

    #initialize mceq instances
    mceq_tune = MCEqRun(
        interaction_model=interactionmodel,
        theta_deg=0.0,
        primary_model=(pm.HillasGaisser2012, "H3a"),
        density_model = (('MSIS00_IC',('SouthPole','January')))
    )

    mceq= MCEqRun(
        interaction_model=interactionmodel,
        theta_deg=0.0,
        primary_model=(pm.HillasGaisser2012, "H3a"),
        density_model = (('MSIS00_IC',('SouthPole','January')))
    )

    # modify cross section
    modcs = ModIntCrossSections(mceq._mceq_db, interaction_model=interactionmodel, scale_factor=scale_factor,threshold=threshold,increase=increase) #scale_factor=[1.,1.3],threshold=1.e4,increase='const')
    modcs.load(interaction_model=interactionmodel)

    mceq_tune._int_cs = modcs # add modification to cross section in mceq instance
    mceq_tune.set_interaction_model(interactionmodel, force=True) # necessary to force cross section change

    #test if tuning is correct:
    print('ratio pion cross section tuned/untuned: ', modcs.get_cs(211, mbarn=True)/InteractionCrossSections(mceq._mceq_db, interaction_model=interactionmodel).get_cs(211, mbarn=True))
    print('ratio kaon cross section tuned/untuned: ', modcs.get_cs(321, mbarn=True)/InteractionCrossSections(mceq._mceq_db, interaction_model=interactionmodel).get_cs(321, mbarn=True))

    # define angles and days for flux calculation
    angle_edges = angular_bins(ptype, 2)
    angles = (angle_edges[:-1] + angle_edges[1:]) / 2

    doy = doys(5) # take every 5th day for fast testing purposes

    # calculate flux
    flux = runmceq(mceq_tune,ptype,doy,angles)

    # save file
    np.save(flux_dir + ptype + '_' + str(scale_factor_p) + 'pion_' + str(scale_factor_k) + 'kaon_' +  str(threshold) + '_' + increase + '_'+ interactionmodel+'_mceqflux.npy',flux) 



    
if __name__ == '__main__':
    main()