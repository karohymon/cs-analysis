
"""numu Aeff from IceCube simulation
"""



import numpy as np
import os
import pandas as pd
import click
from utils import config_handler
import pandas as pd
import h5py




@click.command()
@click.argument('config_file', type=click.Path(exists=True))

def main(config_file):

    # get config settings
    conf_folders = config_handler.get_json_config(str(config_file))
 
    # Load the hdf file
    inputdir = conf_folders['Pass2MC']

    nfiles = 9979. 
    
    filename =inputdir +  '21002_90-120.h5'
    
    # read in file
    theta_min = 90
    theta_max =  100

    f = h5py.File(filename,'r')

    dict = {'MCPrimary1.energy' : f['MCPrimary1']['energy'],
            'MCPrimary1.zenith' : f['MCPrimary1']['zenith'], 
            'I3MCWeightDict.OneWeight' :  f['I3MCWeightDict']['OneWeight'],
            'I3MCWeightDict.TypeWeight':  f['I3MCWeightDict']['TypeWeight'],
            'I3MCWeightDict.TotalWeight':  f['I3MCWeightDict']['TotalWeight'],
            'I3MCWeightDict.NEvents' : f['I3MCWeightDict']['NEvents']}
          
    df = pd.DataFrame(dict)

    df = df[df['MCPrimary1.zenith'].between(np.deg2rad(theta_min),np.deg2rad(theta_max))]
    
    energy = df['MCPrimary1.energy'].values
    cos_theta = np.cos(df['MCPrimary1.zenith'])
    oneweight = df['I3MCWeightDict.OneWeight'].values
    type_weight = df['I3MCWeightDict.TypeWeight'].values  
    nevents = df['I3MCWeightDict.NEvents'].values * nfiles * type_weight

    costh_bins =np.arccos(np.linspace(np.cos(np.deg2rad(theta_max)),np.cos(np.deg2rad(theta_min)),2))[::-1]

    step = 0.3
    bin_edges = np.logspace(2.1, 6, num=int((6 - 2.1) / step) + 1)
    E_bins = bin_edges       
    delta_E = np.diff(E_bins)
    delta_costh = np.diff(np.cos(costh_bins[::-1])) 
    factor = delta_E[:,np.newaxis]*delta_costh*2.0*np.pi
    OneW_2D,_,_ = np.histogram2d(energy, np.arccos(cos_theta), bins = [E_bins,costh_bins], weights =type_weight* oneweight/nevents)
    
    #save Aeff at bin edges      
    Aeff_2D = OneW_2D / factor
    
    np.save(conf_folders['fluxes'] + 'effective_area_21002_2D_' + str(theta_min) + '-' + str(theta_max) + '.npy',Aeff_2D)
    
  
if __name__ == '__main__':
    main()

