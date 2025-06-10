import numpy as np
import crflux.models as pm
import  mceq_underground_helpers_oneregion_threshold as helper

def sl_depth(d,angle):
    '''
    get slant dpeth from depth [km] and angle [deg]
    '''
    return d/np.cos(np.deg2rad(angle))

def get_log_y(X, angle,cs_p,cs_k,cs_pr,e0,e1,season,threshold):
    '''
        calculate log10 of dN/dN_mu in (cm2 s sr)^{-1} at depth [km], angle [deg]
        cs_p, cs_k, cs_pr : factors of cross section modification
        threshold : add integraiton threshold in GeV
        2212: proton (used for mean muon number estimation)
        e0: float (2.05, 2.15,..., 4.95) - log(E/GeV) from which cross section is modified
        e1: float or str
            e0 + 0.2 or "inf"
        season: str (jan, apr, jul)
    '''

    log_y = np.log10(1e4*helper.mult_dist(X,
                            angle,
                            pm.GlobalSplineFitBeta(),"yields_" + season,
                            2212,
                            cs_p, cs_k,cs_pr,e0,e1,
                            norm=False,threshold=threshold#))[0]
                        ) / helper.rates(X, angle, season, 2212, cs_p, cs_k,cs_pr, e0,e1,threshold))

    return log_y

def get_derivative(x_log, log_y):
    '''
        calculate derivative from log10(multiplicity per bundle) & log10(dN/dN_mu)

        x_log: floats, positive, at least len 100. Must have same grid as in helper.
    '''

    deriv_low = (log_y[19] - log_y[1])/(np.log10(x_log[19])-np.log10(x_log[1])) 
    deriv_high = (log_y[99] - log_y[59])/(np.log10(x_log[99])-np.log10(x_log[59])) 

    return np.array([deriv_low, deriv_high])