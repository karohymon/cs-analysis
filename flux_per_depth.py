""" calcualte flux per depth for production profile with changed cs for neutrinos only
"""

from MCEq.core import MCEqRun
import crflux.models as pm
import numpy as np
import click
import os
from cs_modifier import ModIntCrossSections
from utils import config_handler

def save_mceq_sameheight(conf_folders,season,particle, mceq_run):
      
    # set zenith angle
    max_zenith = np.cos(np.deg2rad(100.))
    min_zenith = np.cos(np.deg2rad(90.))
    angles_edges = np.arccos(np.linspace(min_zenith,max_zenith,2))*180./np.pi # edges theta
    angles = np.zeros(1)
    for i in range(len(angles)):
        angles[i] = np.mean([angles_edges[i],angles_edges[i+1]])
        # calculate bin midth for theta
    angles = np.round(angles,decimals=2)
    
    #density model 
    mceq_run.set_density_model(('MSIS00_IC',('SouthPole', "July"))) # 
    mceq_run.set_theta_deg(angles[0])
    
    n_pts = 100
    X_grid = np.linspace(0.1, mceq_run.density_model.max_X, n_pts) # grid for spefic direction
    Xvec = np.logspace(-3, np.log10(3000),100)
    h_msis = mceq_run.density_model.s_lX2h(np.log(Xvec))
    
    
    mceq_run.set_density_model(('MSIS00_IC',('SouthPole', season)))
    
    mag = 0 # differential flux * E^0
  
   
    nu_msis = [] #  flux shape:(zenith,energy,Xgrid)
   

    for a in range(len(angles)):
        print('zenith:', angles[a])
        mceq_run.set_theta_deg(angles[a])    
        
        Xvec = mceq_run.density_model.h2X(h_msis) # use same grid in following iterations of zenith angle
        X_grid = mceq_run.density_model.s_h2X(h_msis)
        
        mceq_run.density_model.calculate_density_spline()
        mceq_run._calculate_integration_path(int_grid=Xvec, grid_var='X',force=True)
        mceq_run.solve(int_grid=Xvec, grid_var='X')
                
        nu_msis.append(np.zeros((mceq_run.dim, Xvec.size)))
        for xi, X in enumerate(Xvec):
            nu_msis[a][:,xi] = (mceq_run.get_solution('total_numu',mag=0,grid_idx=xi) + mceq_run.get_solution('total_antinumu',mag=0,grid_idx=xi))
                
           
            
    np.save( conf_folders['prod_profiles'] + particle + "_msis_sameheight_lowergrid_" + season+ ".npy",nu_msis)


@click.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--scale_factor_k','-s', help='list with cs modification kaon, 1 = untuned for const, 0 = untuned for exp')
@click.option('--scale_factor_p','-f', help='list with cs modification pion, 1 = untuned for const, 0 = untuned for exp')
@click.option('--interactionmodel','-m', help='hadr. interaction model', default="SIBYLL2.3c")


def main(config_file,scale_factor_k,scale_factor_p,interactionmodel):

    scale_factor_p = np.float(f'{scale_factor_p}')
    scale_factor_k = np.float(f'{scale_factor_k}')
    threshold = 1.e4
    increase = 'const'

    scale_factor = [scale_factor_p,scale_factor_k]

    # read folders from config
    conf_folders = config_handler.get_json_config(str(config_file))
    flux_dir = conf_folders['fluxes']
    os.makedirs(flux_dir, exist_ok=True)

    conf_folders = config_handler.get_json_config(str(config_file))

    mceq_tune = MCEqRun(
    interaction_model=interactionmodel,
    primary_model=(pm.HillasGaisser2012, 'H3a'),
    theta_deg=0.0
    )

    mceq = MCEqRun(
    interaction_model=interactionmodel,
    primary_model=(pm.HillasGaisser2012, 'H3a'),
    theta_deg=0.0)
  

   # modify cross section
    modcs = ModIntCrossSections(mceq._mceq_db, interaction_model=interactionmodel, scale_factor=scale_factor,threshold=threshold,increase=increase) #scale_factor=[1.,1.3],threshold=1.e4,increase='const')
    modcs.load(interaction_model=interactionmodel)

    mceq_tune._int_cs = modcs # add modification to cross section in mceq instance
    mceq_tune.set_interaction_model(interactionmodel, force=True) # necessary to force cross section change




    save_mceq_sameheight(conf_folders,"January",'numu' + '_' + str(scale_factor_p) + 'pion_' + str(scale_factor_k) + 'kaon_' +  str(threshold) + '_' + increase + '_'+ interactionmodel,mceq_tune)
    save_mceq_sameheight(conf_folders,"July",'numu'+ '_' + str(scale_factor_p) + 'pion_' + str(scale_factor_k) + 'kaon_' +  str(threshold) + '_' + increase + '_'+ interactionmodel,mceq_tune)
    
if __name__ == '__main__':
    main()