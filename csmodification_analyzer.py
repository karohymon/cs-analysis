
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import mceq_config as config
from MCEq.core import MCEqRun
import crflux.models as pm
import scienceplots
import scipy.optimize as opt
from utils.helpers import *
import matplotlib.gridspec as gridspec

class SensitivityAnalyzer:
    def __init__(self, ptype, scale_factor_p, scale_factor_k, doys, threshold, increase, interactionmodel = None):
        """
        Initialize the SensitivityAnalyzer with the parameters. This class analyzes the variation of cross sections regarding the seasonal flux variation.

        Parameters: - change description later
        - ptype (str): Target particle: mu or numu. (total flux)
        - cs_p (float): Pion-air cs factor (increase or decrease)
        - cs_k (float): Kaon-air cs factor (increase or decrease)
        - doys (list): List of days of the year for analysis.
        - const (string): threshold above whic cross section is changed.
        - interactionmodel (string): must be available in MCEq - default is Sibyll2.3c (None)

        """
        self.ptype = ptype
        self.scale_factor_p = scale_factor_p
        self.scale_factor_k = scale_factor_k
        self.threshold = threshold
        self.increase = increase
        self.interactionmodel = interactionmodel
       
        self.doys = doys
        self.cs_mod =  self.ptype + '_' + str(self.scale_factor_p) + 'pion_' + str(self.scale_factor_k) + 'kaon_' +  str(self.threshold) + '_' + self.increase + '_' + self.interactionmodel
               
    def ebins_gev(self):
        #create 1 bin but keep list format for compatibility with the other functions - for analysis of GeV muons
        ebins = np.logspace(0, 1, num=2)
        return ebins
    
    def ebins_rate(self):
        #create 1 bin but keep list format for compatibility with the other functions - rate for detections in icecube
        ebins = np.logspace(2, 6, num=2)
        return ebins
    
    
    def ebins_analysis(self):
        step = 0.3
        ebins = np.logspace(2.1, 6, num=int((6 - 2.1) / step) + 1)
        return ebins


    def energybins_mceq(self):
        mceq = MCEqRun(
            interaction_model="SIBYLL2.3c",
            theta_deg=0.0,
            primary_model=(pm.HillasGaisser2012, "H3a"),
        )

        energy = mceq.e_grid
        return energy

    def load_data(self): # change

        self.flux_tuned = np.load(f'/data/user/khymon/cs-analysis/{self.ptype}_{self.scale_factor_p}pion_{self.scale_factor_k}kaon_{self.threshold}_{self.increase}_{self.interactionmodel}_mceqflux.npy')
        if self.increase == 'const':
            self.flux_untuned = np.load(f'/data/user/khymon/cs-analysis/{self.ptype}_1.0pion_1.0kaon_{self.threshold}_{self.increase}_{self.interactionmodel}_mceqflux.npy')   
        elif self.increase == 'exp':
            self.flux_untuned = np.load(f'/data/user/khymon/cs-analysis/{self.ptype}_0.0pion_0.0kaon_{self.threshold}_{self.increase}_{self.interactionmodel}_mceqflux.npy')
         

    def get_sv_amplitude(self, flux, energy, ebins, angles_edges, doys):
        masks = []
        flux_year = np.mean(flux, axis=1)

        sv_amplitude = np.ndarray(shape=(len(angles_edges) - 1, len(ebins)-1, len(doys)), dtype=float)  
        
        for j in range(len(angles_edges) - 1):
            for i in range(len(ebins) - 1):
                mask = (energy >= ebins[i]) & (energy < ebins[i + 1])
                masks.append(mask)
                sv_amplitude[j][i] = np.sum(flux[j, :, mask], axis=0) / np.sum(flux_year[j, mask])   



        return sv_amplitude

    def get_amplitudes(self,ebins):
        angles_edges = angular_bins(self.ptype, 2)
        energy = self.energybins_mceq()
        #ebins = self.ebins_analysis()

        self.sv_amplitude_tuned = self.get_sv_amplitude(self.flux_tuned, energy, ebins, angles_edges, self.doys)
        self.sv_amplitude_untuned = self.get_sv_amplitude(self.flux_untuned, energy, ebins, angles_edges, self.doys)


    def deviation_sv_amplitude_plot(self):
        angles_edges =  angular_bins(self.ptype, 2)
     
        energy = self.energybins_mceq()
      
        if self.ptype == 'numu':
            tag = ['allbins', 'rate']

        elif self.ptype == 'mu':
            tag = ['allbins', 'gev', 'rate']

        for e in range(len(tag)):
            if e ==0 :
                ebins = self.ebins_analysis()
            if e ==1: 
                ebins = self.ebins_gev()
            elif e==2:
                ebins = self.ebins_rate()

            self.sv_amplitude_tuned = self.get_sv_amplitude(self.flux_tuned, energy, ebins, angles_edges, self.doys)
            self.sv_amplitude_untuned = self.get_sv_amplitude(self.flux_untuned, energy, ebins, angles_edges, self.doys)

            overall_deviation = np.ndarray(shape=(len(angles_edges) - 1, len(ebins)), dtype=float)
            self.sensitivity = self.sv_amplitude_tuned / self.sv_amplitude_untuned # change this line

            for j in range(len(angles_edges) - 1):
                for i in range(len(ebins) - 1):
                    overall_deviation[j][i] = np.sum(np.abs(self.sensitivity[j][i] - 1))

            plt.figure(figsize=(12, 6))
            if e == 0:
                plotlooplength = len(ebins) - 4
            else:
                plotlooplength = len(ebins) -1
            for i in range(plotlooplength):
                plt.plot(self.doys, self.sensitivity[0][i], marker='.', linestyle='-', label=f"{np.round(np.log10(ebins[i]), decimals=1)} $\leq$ log(E/GeV) $\leq$ {np.round(np.log10(ebins[i + 1]), decimals=1)}")
            plt.axhline(y=1, color='black', linestyle='--', label='Original Value')
            plt.title('Daily Sensitivity Across the Year')
            plt.ylabel('Daily deviation from seasonal variation amplitude')
            if self.ptype == 'numu':
                plt.ylim(0.98,1.02)
            elif self.ptype == 'mu':
                plt.ylim(0.98,1.02)
            else:
                print('particle type is not defined.')
                exit()
            plt.xlabel('Day of Year')
            plt.xlim()
            plt.legend(ncol=2)
            plt.grid(True)
            plt.text(50,0.99,f'{self.ptype} {self.scale_factor_k}xkaon {self.scale_factor_p}pion threshold{self.threshold}GeV')
            plt.savefig(f'/home/khymon/Plots/cs-analysis/sv_amplitude_mcs_{tag[e]}_dailysensitivity{self.cs_mod}_zenith{np.round(angles_edges[0], decimals=0)}-{np.round(angles_edges[1], decimals=0)}.png', bbox_inches='tight')
          
            return self.sensitivity
        
    def sensitivity_plot(self):
        angles_edges =  angular_bins(self.ptype, 2)
        tag = ['allbins', 'rate']
     
        energy = self.energybins_mceq()

        for e in range(len(tag)):
            if e ==0 :
                ebins = self.ebins_analysis()
                plotlooplength = len(ebins) - 4
            elif e==1:
                ebins = self.ebins_rate()
                plotlooplength = len(ebins) -1
               
            self.sv_amplitude_tuned = self.get_sv_amplitude(self.flux_tuned, energy, ebins, angles_edges, self.doys)
            self.sv_amplitude_untuned = self.get_sv_amplitude(self.flux_untuned, energy, ebins, angles_edges, self.doys)

            overall_deviation = np.ndarray(shape=(len(angles_edges) - 1, len(ebins)), dtype=float)
            self.sensitivity = self.sv_amplitude_tuned / self.sv_amplitude_untuned # change this line

            for j in range(len(angles_edges) - 1):
                for i in range(len(ebins) - 1):
                    overall_deviation[j][i] = np.sum(np.abs(self.sensitivity[j][i] - 1))

            plt.figure(figsize=(12, 6))
            
            if self.scale_factor_k != 1.:
                if self.scale_factor_k > 1.:
                    scale_fac = self.scale_factor_k - 1.
                else:
                    scale_fac =  -self.scale_factor_k
            elif self.scale_factor_p != 1.:
                if self.scale_factor_p > 1.:
                    scale_fac = self.scale_factor_p -1.
                else:
                    scale_fac = - self.scale_factor_p

            for i in range(plotlooplength):
                    plt.plot(self.doys, self.sensitivity[0][i]/scale_fac, marker='.', linestyle='-', label=f"{np.round(np.log10(ebins[i]), decimals=1)} $\leq$ log(E/GeV) $\leq$ {np.round(np.log10(ebins[i + 1]), decimals=1)}")
                
            print('scale factor =', scale_fac)
            plt.ylabel('d amplt. / d$\sigma')
        
            plt.xlabel('Day of Year')
            plt.xlim()
            plt.legend(ncol=2)
            plt.grid(True)
            plt.title(f'{self.ptype} {self.scale_factor_k}xkaon {self.scale_factor_p}pion threshold{self.threshold}GeV')
            plt.savefig(f'dailysensitivity{self.cs_mod}_zenith{np.round(angles_edges[0], decimals=0)}-{np.round(angles_edges[1], decimals=0)}_{tag[e]}.png', bbox_inches='tight')


            
    
    def season_analysis(self):

        angles_edges =  angular_bins(self.ptype, 2)
        energy = self.energybins_mceq()
        ebins = self.ebins_analysis()      

        self.sv_amplitude_tuned = self.get_sv_amplitude(self.flux_tuned, energy, ebins, angles_edges, self.doys)
        self.sensitivity = self.deviation_sv_amplitude_plot()
        sens = self.sensitivity[0]
        ebins =  self.ebins_analysis()
       
        x_min = np.zeros(len(ebins)-1)
        x_max = np.zeros(len(ebins)-1)

        y_min = np.zeros(len(ebins)-1)
        y_max = np.zeros(len(ebins)-1)

        # Initialize lists to store results
        min_indices = []
        max_indices = []

        for j in range(min(len(ebins)-1, sens.shape[0])):  # Ensure indices are within bounds
            indices = np.where(sens[j, :] > 1)[0]  # Find indices where values are > 0
            if len(indices) > 0:  # Check if there are any values > 0
                min_indices.append(indices.min())  # Minimum index
                max_indices.append(indices.max())  # Maximum index
            else:
                min_indices.append(None)  # No values > 0, store None
                max_indices.append(None)
                    
            y_max[j] = self.sv_amplitude_tuned[0,j,np.argmax(sens[j])]
            y_min[j] = self.sv_amplitude_tuned[0,j,np.argmin(sens[j])]
            x_max[j] = self.doys[np.argmax(sens[j])]
            x_min[j] = self.doys[np.argmin(sens[j])]

        return x_min, x_max, y_min, y_max, min_indices, max_indices

    def sv_amplitude_plot(self):
        angles_edges  = angular_bins(self.ptype, 2)
        ebins = self.ebins_analysis()

        A4_width_inches = 8.27 * 0.5
        A4_height_inches = A4_width_inches * 0.75 * 2
        figsize = (A4_width_inches, A4_height_inches)

        for j in range(len(angles_edges) - 1):
            fig = plt.figure(figsize=figsize, dpi=500)
            fig.tight_layout()

            plt.rcParams.update({'font.size': 10})
            plt.style.use('science')
            plt.style.use('tableau-colorblind10')
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

            gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])

            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])
            #ax1.text(100,1.1,str(np.round(angles_edges[j],decimals=1)) + '$^{\circ} \leq \Theta \leq $ ' + str(np.round(angles_edges[j+1])) + '$^{\circ}$')
            

            for i in range(len(ebins) - 1):
                if i < 7:
                    ax1.plot(self.doys, self.sv_amplitude_untuned[j][i], color=colors[i],
                            label=str(np.round(np.log10(ebins[i]),decimals=1)) + r'$\leq \log(E/\text{GeV})\leq$'+ str(np.round(np.log10(ebins[i+1]),decimals=1)))
                    ax1.plot(self.doys, self.sv_amplitude_tuned[j][i], ls='--', color=colors[i])
                else:
                    ax2.plot(self.doys, self.sv_amplitude_untuned[j][i], color=colors[i - 7],
                            label=str(np.round(np.log10(ebins[i]),decimals=1)) + r'$\leq \log(E/\text{GeV})\leq$'+ str(np.round(np.log10(ebins[i+1]),decimals=1)))
                    ax2.plot(self.doys, self.sv_amplitude_tuned[j][i], ls='--', color=colors[i - 7])

            ax1.set_ylim(0.88, 1.11)
            ax1.minorticks_on()
            ax1.set_ylabel('Ratio daily flux to annual average')
            ax1.set_xlabel('Day of Year')
            ax1.legend(fontsize = 'small',ncol=2)
            ax2.set_ylim(0.88, 1.11)
            ax2.minorticks_on()
            ax2.set_ylabel('Ratio daily flux to annual average')
            ax2.set_xlabel('Day of Year')
            ax2.legend(fontsize = 'small',ncol=2)
            plt.grid(True)
            plt.savefig(f'/home/khymon/Plots/cs-analysis/sv_amplitude_mcs_allbins_{self.cs_mod}_zenith{np.round(angles_edges[j], decimals=0)}-{np.round(angles_edges[j + 1], decimals=0)}.png', bbox_inches='tight')

    def plot_flux(self):

        energy = self.energybins_mceq()   

        flux_year_tuned = np.mean(self.flux_tuned, axis=1)
        flux_year_untuned = np.mean(self.flux_untuned, axis=1)

        fig = plt.figure(figsize=(12,6))
        fig.tight_layout()

        gs = gridspec.GridSpec(4, 1)
        axes1 = fig.add_subplot(gs[:-2])
        axes2 = fig.add_subplot(gs[-2], sharex=axes1)
        fig.subplots_adjust(hspace = .001)

        axes1.plot(energy,flux_year_untuned[0],label = 'untuned')
        axes1.plot(energy,flux_year_tuned[0],label =  self.cs_mod, ls='--')
        axes1.set_xscale('log')
        axes1.set_yscale('log')
        axes2.set_xlabel('$E$/GeV')
        axes1.set_ylabel('flux')
        axes1.set_xlim(1.e2,1e6)
        axes1.set_ylim(1.e-20,1e-3)
        axes1.legend(loc='upper right')

        axes2.plot(energy,flux_year_tuned[0]/flux_year_untuned[0])
        axes2.set_ylim(0.6,1.4)
        axes2.set_ylabel('tuned/untuned')
        fig.savefig(f'/home/khymon/Plots/cs-analysis/avgflux_{self.cs_mod}_zenith90-100.png', bbox_inches='tight')
