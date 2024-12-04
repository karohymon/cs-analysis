

from MCEq.core import MCEqRun
import crflux.models as pm
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
import scienceplots
from utils import config_handler
from utils.mceqflux_interpolation import *



def avg_flux_at_bin_mid(mceq_spectrum, analysis_bin_edges,mceq_edges):
    '''
    Calculate avg mceq flux for particular season at energy bins of analysis
    
    Parameters
    -----------
    mceq_spectrum: array-like, floats
        mceq flux at mceq_edges in untis (GeV cm2 s ssr)^{-1} for particular season
        shape(angles_center,100) # cut out highest energies because unphysical values in cascade equation solution

    analysis_bin_edges
    
    mceq_edges

    Returns
    ----------
    flux_analysis_center : array-like
        Flux at analysis bin center.
    
    '''
    flux_analysis_center =  np.ndarray(shape=(5,len(analysis_bin_edges)-1),dtype=float)    
    
    for i in range(5):
        flux_analysis_center[i] = MCEq_bin_averages(mceq_spectrum[i], analysis_bin_edges,mceq_edges)
    
    return flux_analysis_center

def profile_energyrange(conf_folders,nu_msis,ebin_index):

    '''
        Calculate production profile for a given energy bin.

        Params
        -------

        conf_folders : dict
            Path to folders
        
        nu_msis : array-like
            Interpolated mceq on analysis grid for a given month/day

        ebin_index : int
            Energy bin index for which the profile should be calculated.

        Returns
        ---------
        R_per_X_bin : array-like (12, 100)
            Rate per X for each energy bin

    '''
    
    # reshape correctly
    nu_msis = nu_msis.T[ebin_index]

    # load Aeff
    Aeff_2D = np.load(conf_folders['fluxes'] + 'effective_area_21002_2D_90-100.npy')

    # set zenith angle
    max_zenith = np.cos(np.deg2rad(100.))
    min_zenith = np.cos(np.deg2rad(90.))
    angles_edges = np.arccos(np.linspace(min_zenith,max_zenith,6))*180./np.pi # edges theta
    angles = np.zeros(5)
    for i in range(len(angles)):
        angles[i] = np.mean([angles_edges[i],angles_edges[i+1]])
        # calculate bin midth for theta
    angles = np.round(angles,decimals=2)

    # bins analysis
    step = 0.3
    bin_edges = np.logspace(2.1, 6, num=int((6 - 2.1) / step) + 1)
    
    R_per_X_per_zenith = np.ndarray(dtype=float,shape=(5,100)) # for given season

    dEgrid = bin_edges[ebin_index+1] - bin_edges[ebin_index]
    
   
    #dEgrid =energy_bin[1:] -energy_bin[:-1]
    geom_factor =  - np.diff(np.linspace(0,max_zenith,6)) * 2 * np.pi #dCos int(dPhi # adapt for 90-110 zenith bins

    for a in range(len(angles)):
        Aeff_center = Aeff_2D# effective area per analysis energy bin and zenith band 

        
        for  i in range(100): # loop over height

            R_per_X_per_zenith[a][i] = geom_factor[a] * np.sum(dEgrid*nu_msis[a][i]*Aeff_center)
            
    R_per_X_bin = np.sum(R_per_X_per_zenith, axis=0)
    
    return R_per_X_bin

def production_profile(conf_folders,interactionmodel,cs_k,cs_p,season, mceq_run):

    '''
    Plot production profile for selected energy bins.

    Params
    --------
    conf_folders : dict
        Path to folders

    mceq_run : MCEqRun instance just needed for height

    
    '''    
    # set zenith angle

    threshold = 1.e4
    increase = 'const'

    max_zenith = np.cos(np.deg2rad(110.))
    min_zenith = np.cos(np.deg2rad(90.))
    angles_edges = np.arccos(np.linspace(min_zenith,max_zenith,6))*180./np.pi # edges theta
    angles = np.zeros(5)
    for i in range(len(angles)):
        angles[i] = np.mean([angles_edges[i],angles_edges[i+1]])
        # calculate bin midth for theta
    angles = np.round(angles,decimals=2)

    step = 0.3
    bin_edges = np.logspace(2.1, 6, num=int((6 - 2.1) / step) + 1)

    mceq_run.set_density_model(('MSIS00_IC',('SouthPole', "July"))) # 
    mceq_run.set_theta_deg(angles[4])

    n_pts = 100
    X_grid = np.linspace(0.1, mceq_run.density_model.max_X, n_pts) # grid for spefic direction
    Xvec = np.logspace(-3, np.log10(3000),100)
    h_msis = mceq_run.density_model.s_lX2h(np.log(Xvec))
    h = (h_msis[1:] + h_msis[:-1])/2 

    #read  calculated flux
    flux = np.load(conf_folders['fluxes'] + "numu_"+ cs_p + "pion_" + cs_k + "kaon_" +  str(threshold) + "_" + increase + "_"+ interactionmodel +"_msis_sameheight_lowergrid_" + season+ ".npy")
   
    e_center = (mceq_run.e_bins[:-1] + mceq_run.e_bins[1:]) / 2
    mceq_interp_bin_mid = np.ndarray(shape=(100,5,len(bin_edges)-1),dtype="float") #acg per zenith band
 
    for i in range(100): 
        mceq_interp_bin_mid[i] = avg_flux_at_bin_mid(flux[:,:,i], bin_edges ,e_center)
       

    # get profiles and plot
    R_per_X = np.ndarray(shape=(len(bin_edges)-1,100),dtype = 'float')
    for e in range(len(bin_edges)-1):
        R_per_X[e] = profile_energyrange(conf_folders,mceq_interp_bin_mid,e)

    profile = np.ndarray(shape=(len(bin_edges)-1,99),dtype='float')

    for i in range(len(bin_edges)-1):
        profile[i] = np.diff(R_per_X[i])/np.sum(np.diff(R_per_X[i]))

    return profile, h

def plot_profiles(conf_folders,interactionmodel1,cs_k1,cs_p1,season1,interactionmodel2,cs_k2,cs_p2,season2):

    step = 0.3
    bin_edges = np.logspace(2.1, 6, num=int((6 - 2.1) / step) + 1)

    mceq_run = MCEqRun(
    interaction_model='SIBYLL2.3c',
    primary_model=(pm.HillasGaisser2012, 'H3a'),
    theta_deg=0.0
    )

    # get profiles
    profile1, h = production_profile(conf_folders,interactionmodel1,cs_k1,cs_p1,season1, mceq_run)
    profile2, _ = production_profile(conf_folders,interactionmodel2,cs_k2,cs_p2,season2, mceq_run)

    # plot
    A4_width_inches = 0.5*8.27  # Width of A4 paper in inches
    A4_height_inches = A4_width_inches * 0.75 # Height is one-third of width
    figsize = (A4_width_inches, A4_height_inches)

    # Create subplots using GridSpec
    fig = plt.figure(figsize=figsize,dpi=500)
    fig.tight_layout(pad=0.4)

    plt.rcParams.update({'font.size': 10})
    plt.style.use('science')

    plt.style.use('tableau-colorblind10')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] 

    ax = plt.subplot()
    ax.plot(np.append(h*10**(-5),0),np.append(profile1[6],profile1[6][-1]),label=str(np.log10(bin_edges[6])) + '$\leq \log (E/$GeV$) \leq $' + str(np.log10(bin_edges[7])),color=colors[7]) # get mid X + convert X to  h
    ax.plot(np.append(h*10**(-5),0),np.append(profile1[7],profile1[7][-1]),label=str(np.round(np.log10(bin_edges[7]),decimals=2)) + '$\leq \log (E/$GeV$) \leq $' + str(np.log10(bin_edges[6])),color=colors[2])
    ax.plot(np.append(h*10**(-5),0),np.append(profile1[8],profile1[8][-1]),label=str(np.log10(bin_edges[8])) + '$\leq \log (E/$GeV$) \leq $' + str(np.log10(bin_edges[9])),color=colors[5])
    ax.plot(np.append(h*10**(-5),0),np.append(profile1[9],profile1[9][-1]),label=str(np.log10(bin_edges[9])) + '$\leq \log (E/$GeV$) \leq $' + str(np.log10(bin_edges[10])),color=colors[1])

    ax.plot(np.append(h*10**(-5),0),np.append(profile2[6],profile2[6][-1]),color=colors[7],ls='--') # get mid X + convert X to  h
    ax.plot(np.append(h*10**(-5),0),np.append(profile2[7],profile2[7][-1]),color=colors[2],ls='--')
    ax.plot(np.append(h*10**(-5),0),np.append(profile2[8],profile2[8][-1]),color=colors[5],ls='--')
    ax.plot(np.append(h*10**(-5),0),np.append(profile2[9],profile2[9][-1]),color=colors[1],ls='--')



    # Dummy lines for January and July to show in legend in color[3]
    line_jan = plt.Line2D([0], [0], color='black',  label=interactionmodel1 +  ': '+ cs_p1 +'xpi, ' + cs_k1 + 'xK') 
    line_jul = plt.Line2D([0], [0], color='black',  linestyle='--', label=interactionmodel2 +  ': '+ cs_p2 +'xpi, ' + cs_k2 + 'xK')

    # Legend for line styles (solid and dashed), placed inside the plot
    line_style_legend = ax.legend(handles=[line_jan, line_jul], loc='upper left', frameon=False)

    ax.set_xlabel('$h$ / km')
    ax.set_ylabel('Neutrino production PDF')
    ax.set_xlim(0,50)
    ax.minorticks_on()
    ax.tick_params(axis='x', which='both', direction='in', bottom=True, top=True)
    ax.tick_params(axis='y', which='both', direction='in', left=True, right=True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles,labels, loc='upper center', bbox_to_anchor=(0.5, 1.07), ncol=2)

    plt.savefig(conf_folders['plot_directory'] + 'profile_' + interactionmodel1 + '_' + cs_k1 + '_k_' +cs_p1 + '_pi_' + season1+'-'+ interactionmodel2 + '_' + cs_k2 + '_k_' +cs_p2 + '_pi_' + season2 +'.png', bbox_inces='tight') 

    
    

    

#conf_folders = config_handler.get_json_config(str(config_file))

  

#plot_production_profile(conf_folders,interactionmodel1,cs_k1,cs_p1,season1,interactionmodel2,cs_k2,cs_p2,season2,mceq_run))

