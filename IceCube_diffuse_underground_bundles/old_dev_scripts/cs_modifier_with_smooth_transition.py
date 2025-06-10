import numpy as np
import matplotlib.pyplot as plt
import mceq_config as config
from MCEq.core import MCEqRun
import crflux.models as pm
from MCEq.data import InteractionCrossSections
from scipy.interpolate import CubicSpline
from utils.helpers import *
from utils import config_handler
import click
import os

class ModIntCrossSections(InteractionCrossSections):
    def __init__(self, mceq_hdf_db, interaction_model="SIBYLL2.3c", scale_factor_region1=None, scale_factor_region2=None, e0=1., e1=None, increase='const'):
        self.scale_factor_region1 = scale_factor_region1  # Scaling factor for e0 to e1
        self.scale_factor_region2 = scale_factor_region2  # Scaling factor for > e1
        self.e0 = e0
        self.e1 = e1
        self.increase = increase
        self.interaction_model = interaction_model
        super().__init__(mceq_hdf_db, interaction_model)  # Call parent constructor

    def modify_cs(self, scale_factor_region1=None, scale_factor_region2=None, e0=None, e1=None, increase=None):
        """Modify pion and kaon interaction cross-sections in two energy regions."""
        
        # Set defaults
        if scale_factor_region1 is None:
            scale_factor_region1 = self.scale_factor_region1
        if scale_factor_region2 is None:
            scale_factor_region2 = self.scale_factor_region2
        if e0 is None:
            e0 = self.e0
        if e1 is None:
            e1 = self.e1  # e1 might still be None here

        # Energy ranges
        e_range1 = (self.energy_grid.c > e0)
        if e1 is not None:  
            e_range1 &= (self.energy_grid.c < e1)

            # Find the index at e1 and start the second region from the second bin after e1
            idx_at_e0 = np.searchsorted(self.energy_grid.c, e0)
            e_range1 = self.energy_grid.c >= self.energy_grid.c[idx_at_e0 + 1]
            
            # Find the index at e1 and start the second region from the second bin after e1
            idx_at_e1 = np.searchsorted(self.energy_grid.c, e1)
            e_range1 &= self.energy_grid.c <= self.energy_grid.c[idx_at_e1 - 1]  # set upper bound region 1

            e_range2 = self.energy_grid.c >= self.energy_grid.c[idx_at_e1 + 1]  # Start from the second bin after e1
        else:
            e_range2 = np.zeros_like(self.energy_grid.c, dtype=bool)

        print(f"Applying mod: region1 ({e0}-{e1}) with {scale_factor_region1}, region2 (> {e1}) with {scale_factor_region2}")

        # Apply scaling factors first for the energy range before e1 (excluding idx_at_e1)
        if self.increase == 'exp':
            for p, sf1, sf2 in zip([211, 321], scale_factor_region1, scale_factor_region2):
                self.index_d[p][e_range1] *= (self.energy_grid.c[e_range1] / e0) ** sf1
        elif self.increase == 'const':
            for p, sf1, sf2 in zip([211, 321], scale_factor_region1, scale_factor_region2):
                self.index_d[p][e_range1] *= sf1

       
        # Now handle smooth transitions for e0 and e1 if necessary
        if e1 is not None:
            self.smooth_transition(e0, e1)

    def smooth_transition(self, e0, e1):
        """Smooth transition for values around e1 using interpolation."""
        
        if e0 is None or e1 is None:
            raise ValueError("e0 and e1 must be specified.")
        
        # Energy ranges
        e_range1 = (self.energy_grid.c > e0)
        if e1 is not None:  
            e_range1 &= (self.energy_grid.c < e1)

            # Find the index at e1 and start the second region from the second bin after e1
            idx_at_e0 = np.searchsorted(self.energy_grid.c, e0)
            e_range1 = self.energy_grid.c >= self.energy_grid.c[idx_at_e0 + 1]
            
            # Find the index at e1 and start the second region from the second bin after e1
            idx_at_e1 = np.searchsorted(self.energy_grid.c, e1)
            e_range1 &= self.energy_grid.c <= self.energy_grid.c[idx_at_e1 - 1]  # set upper bound region 1

            e_range2 = self.energy_grid.c >= self.energy_grid.c[idx_at_e1 + 1]  # Start from the second bin after e1
        else:
            e_range2 = np.zeros_like(self.energy_grid.c, dtype=bool)
        
        # Find the index at e1 and e0
        idx_at_e0 = np.searchsorted(self.energy_grid.c, e0)
        idx_at_e1 = np.searchsorted(self.energy_grid.c, e1)

        # The index at e1 will be the first position where the energy is greater than or equal to e1
        energy_at_e0 = self.energy_grid.c[idx_at_e0] if idx_at_e0 < len(self.energy_grid.c) else None
        energy_at_e1 = self.energy_grid.c[idx_at_e1] if idx_at_e1 < len(self.energy_grid.c) else None
        
        # Ensure e1 is within the energy grid
        if energy_at_e0 is None or energy_at_e0 > self.energy_grid.c[-1]:
            print(f"Warning: e1 = {e1} is outside the energy grid, using the last energy grid value instead.")
            energy_at_e0 = self.energy_grid.c[-1]  # Set to last value if out of bounds

        if energy_at_e1 is None or energy_at_e1 > self.energy_grid.c[-1]:
            print(f"Warning: e1 = {e1} is outside the energy grid, using the last energy grid value instead.")
            energy_at_e1 = self.energy_grid.c[-1]  # Set to last value if out of bounds
        
        # Interpolation of scaling factors between energy bins around e1
        # Iterate through the particles and scaling factors
        # Loop over particles, scale factors for each region (region1 and region2)
        for p, sf1, sf2 in zip([211, 321], self.scale_factor_region1, self.scale_factor_region2):
            print(f"Processing particle {p} with sf1={sf1} and sf2={sf2}")

            if self.increase == 'exp':
                # Avoid modifying idx_at_e1 or bins after it in this loop
                if e1 is not None:
                    self.index_d[p][(idx_at_e1+1):] = self.index_d[p][(idx_at_e1+1):] * (self.energy_grid.c[idx_at_e1:] / e0) ** sf2
            elif self.increase == 'const':
                self.index_d[p][(idx_at_e1+1):] *= sf2  # Scale for region2 from e1 +1 onward
            
            # Interpolation at e0
            cs_e0 = CubicSpline(
                [self.energy_grid.c[idx_at_e0 - 1], self.energy_grid.c[idx_at_e0 + 1]],
                [1., sf1],
                bc_type=((1, 0.0), (1, 0.0))
            )
            interpolated_value_e0 = cs_e0(energy_at_e0)
            self.index_d[p][idx_at_e0] *= interpolated_value_e0

            # Interpolation at e1
            cs_e1 = CubicSpline(
                np.append(self.energy_grid.c[e_range1], [self.energy_grid.c[idx_at_e1 + 1]]),
                np.append(sf1 * np.ones(len(self.energy_grid.c[e_range1])), [sf2]),
                bc_type=((1, 0.0), (1, 0.0))
            )
            interpolated_value_e1 = cs_e1(energy_at_e1)
            self.index_d[p][idx_at_e1] *= interpolated_value_e1

            # Single plot combining both e0 and e1 for each particle
            #plt.figure(figsize=(6, 4))
            
            # Plot for e0
            #plt.plot([self.energy_grid.c[idx_at_e0 - 1], self.energy_grid.c[idx_at_e0 + 1]], [1., sf1], 'ro', label="Original Points ")
            #plt.plot(energy_at_e0, interpolated_value_e0, 'bo', label="Interpolated Points")
            
            # Plot for e1
            #plt.plot([self.energy_grid.c[idx_at_e1 - 1], self.energy_grid.c[idx_at_e1 + 1]], [sf1, sf2], 'ro')
            #plt.plot(energy_at_e1, interpolated_value_e1, 'bo')
            
            # Interpolation curves
            #start_0 = self.energy_grid.c[idx_at_e0 - 1]
            #end_0 = self.energy_grid.c[idx_at_e0 + 1]
            #start_1 = self.energy_grid.c[idx_at_e1 - 1]
            #end_1 = self.energy_grid.c[idx_at_e1 + 1]
            #numbers_0 = np.linspace(start_0, end_0,50)
            #numbers_1 = np.linspace(start_1, end_1,50)

            #plt.plot(numbers_0, cs_e0(numbers_0), label='Interpolation Curve',color='black')
            #plt.plot(numbers_1, cs_e1(numbers_1),color='black')

            #plot region 1
            #plt.plot([end_0,start_1],[sf1,sf1],color='black')
            
            # Graph settings
            #plt.xscale("log")
            #plt.xlabel("Energy [GeV]")
            #plt.ylabel("Scaling Factor")
            #plt.legend()
            #plt.grid()
            #plt.title(f'Particle {p} - Interpolations at e0 and e1')
            #plt.show()

    def load(self, interaction_model):
        """Load the interaction model and apply the modification function."""
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