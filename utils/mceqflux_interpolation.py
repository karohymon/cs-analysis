''' Helper functions for interpolation of mceq flux to analysis enery grid'''

import numpy as np



def MCEq_bin_averages(mceq_spectrum, analysis_bin_edges,mceq_edges):
    '''
    interpolate MCEq spectrum at a new energy grid. Flux is averaged over new bin edges.
    
    Paramteters
    -------------
    
    mceq_spectrum: array-like
        Flux at discrete energy grid for a given zenith angle and season

    analysis_bin_edges: array-like
        analysis bin edges
    
    mceq_edges: array-like
        mceq grid edges

    Returns
    ---------
    MCEq flux at bin average
    '''
    
    mceq_widths = []
    analysis_widths = []
    for i in range(1, len(mceq_edges)):
        width = mceq_edges[i] - mceq_edges[i-1]
        mceq_widths.append(width)
        
    for i in range(1, len(analysis_bin_edges)):
        analysis_widths.append(analysis_bin_edges[i] - analysis_bin_edges[i-1])
    
    # mceq spectrum aat energy bin center
    flux_center = np.zeros(len(mceq_edges)-1)
    e_center = np.zeros(len(mceq_edges)-1)
    for j in range(len(mceq_edges)-1):
        flux_center[j] = np.mean([mceq_spectrum[j],mceq_spectrum[j+1]])
        e_center[j] = np.mean([mceq_edges[j],mceq_edges[j+1]])
        
  
    
    cumspec = np.cumsum(flux_center*mceq_widths)

    return np.diff(np.interp(analysis_bin_edges, e_center, cumspec))/analysis_widths

def avgflux_at_bin_mid(mceq_spectrum, analysis_bin_edges,mceq_edges,Aeff):
    '''
    Calculate avg mceq flux for particular season at energy bins of analysis
    
    Parameters
    -----------
    mceq_spectrum: array-like, floats
        mceq flux at mceq_edges in untis (GeV cm2 s ssr)^{-1} for particular season
        shape(angles_center,100) # cut out highest energies because unphysical values in cascade equation solution

    Aeff : array-like
        2D effective area for each zenith bins. Bins must match analysis bins and columns zenith bins
    
    
    '''
    flux_analysis_center =  np.ndarray(shape=(5,len(analysis_bin_edges)-1),dtype=float)    
    
    for i in range(5):
        flux_analysis_center[i] = MCEq_bin_averages(mceq_spectrum[i], analysis_bin_edges,mceq_edges)
    
    return (flux_analysis_center[0] * Aeff[:,0] + flux_analysis_center[1] * Aeff[:,1] + flux_analysis_center[2] * Aeff[:,2] + \
                flux_analysis_center[3] * Aeff[:,3] + flux_analysis_center[4] * Aeff[:,4])/np.sum(Aeff,axis=1)
    

