import numpy as np


def angular_bins(ptype, nbins):
        if ptype == 'numu':
            max_zenith = np.cos(np.deg2rad(110.))
            min_zenith = np.cos(np.deg2rad(90.))
        elif ptype == 'mu':
            max_zenith = np.cos(np.deg2rad(30.))
            min_zenith = np.cos(np.deg2rad(0.))
        
        angles_edges = np.arccos(np.linspace(min_zenith, max_zenith, nbins + 1)) * 180. / np.pi
        angles = np.zeros(nbins)
        for i in range(len(angles)):
            angles[i] = np.mean([angles_edges[i], angles_edges[i + 1]])
        angles = np.round(angles, decimals=2)

        return angles_edges

def doys(frequency):
    return np.arange(1, 362, frequency, dtype=int)
