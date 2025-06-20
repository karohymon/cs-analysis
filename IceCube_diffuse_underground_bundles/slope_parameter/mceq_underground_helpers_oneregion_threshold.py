import numpy as np
import scipy.interpolate as ip
import pickle
from daemonflux import Flux
import pathlib

# Energies in [MeV]

_e_bins = np.logspace(1.9, 14, 122)[:-30] * 1e-3  # Bin edges in GeV
_e_widths = np.diff(_e_bins)  # Bin widths
mute_energies = np.sqrt(
    _e_bins[1:] * _e_bins[:-1]
)  # Bin centers == mceq_egrid [:dim_ug]
dim_ug = len(mute_energies)
mceq_bins = np.logspace(1.9, 14, 122) * 1e-3
mceq_egrid = np.sqrt(mceq_bins[1:] * mceq_bins[:-1])
cr_grid = mceq_egrid[30:-10]


# Define global variables for angles and c_wi
angles = None
c_wi = None
cos_thetas = None



# Define base directory as a function to call

def initialize_flux_dicts(ptype_values, cs_p_values, cs_k_values, cs_pr_values ,e0_values, e1_values=None, pairwise=False):
    global flux_files, muspec_files, intp_surface_fluxes, intp_ground_mu_yields, surface_fluxes, ground_muspec_energies, cos_thetas, angles, c_wi

    base_dir = pathlib.Path(__file__).parent
    cs_dir = pathlib.Path("/hetghome/khymon/cs-files/smooth-transition")  # Cross-section tuned files

    # If e1_values is None, set e1_list to contain only None
    e1_list = e1_values if e1_values is not None else [None]

    # Create the flux_files and muspec_files dictionaries
    flux_files = {}
    muspec_files = {}

    for ptype in ptype_values:
        for cs_p in cs_p_values:
            for cs_k in cs_k_values:
                for cs_pr in cs_pr_values:
                    if cs_p == 1.0 and cs_k==1.0 and cs_pr==1.0:
                        # Special case when all cs are the same as in Sibyll2.3c: only use one specific combination of e0 and e1 = None
                        #for e0 in range(len(e0)): #only one e0 needed technically

                        e0 = e0_values[0]  # Assign a default value for accessing the keys correctly

                         # add key but cs_pr is not indicated in the file names   
                        flux_files[(ptype, round(cs_p, 2), round(cs_k, 2), round(cs_pr, 2), round(e0, 2), "inf")] = \
                            cs_dir / f"surface_fluxes_season{ptype}_pi{cs_p:.2f}_k{cs_k:.2f}.pkl"#_e0{e0:.2f}_const.pkl"
                            
                        muspec_files[(ptype, round(cs_p, 2), round(cs_k, 2), round(cs_pr, 2),  round(e0, 2), "inf")] = \
                            cs_dir / f"ground_muspec_prim_energies_season_cstune{ptype}_pi{cs_p:.2f}_k{cs_k:.2f}.pkl"#_e0{e0:.2f}_const.pkl"
                    else:
                        # Handle case where pairwise is True
                        if pairwise:
                            for e0, e1 in zip(e0_values, e1_list):  # Pair e0 and e1 element-wise
                                print(e0,e1)
                                e0 = float(e0)  # Convert NumPy scalar to float
                                e1 = float(e1)  # Convert e1 as well if needed
                                print('helper script', e0, e1,cs_p, cs_k, cs_pr)

                                #add case for cs_pr
                                if cs_pr== 1.0:
                                    
                                    # file without cs_pr                            
                                    flux_files[(ptype, round(cs_p, 2), round(cs_k, 2), round(cs_pr, 2), round(e0, 2), round(float(e1), 2) if e1 != "inf" else "inf")] = \
                                        cs_dir / f"surface_fluxes_season{ptype}_pi{cs_p:.2f}_k{cs_k:.2f}_e0{e0:.2f}_const_{'inf' if e1 is None else f'e1{e1:.2f}'}.pkl"
                                    
                                    muspec_files[(ptype, round(cs_p, 2), round(cs_k, 2), round(cs_pr, 2), round(e0, 2), round(float(e1), 2) if e1 != "inf" else "inf")] = \
                                    cs_dir / f"ground_muspec_prim_energies_season_cstune{ptype}_pi{cs_p:.2f}_k{cs_k:.2f}_e0{e0:.2f}_const_{'inf' if e1 is None else f'e1{e1:.2f}'}.pkl"

                                else:
                                    flux_files[(ptype, round(cs_p, 2), round(cs_k, 2), round(cs_pr, 2), round(e0, 2), round(float(e1), 2) if e1 != "inf" else "inf")] = \
                                        cs_dir / f"surface_fluxes_season{ptype}_pi{cs_p:.2f}_k{cs_k:.2f}_pr{cs_pr:.2f}_e0{e0:.2f}_const_{'inf' if e1 is None else f'e1{e1:.2f}'}.pkl"
                                    
                                    muspec_files[(ptype, round(cs_p, 2), round(cs_k, 2), round(cs_pr, 2), round(e0, 2), round(float(e1), 2) if e1 != "inf" else "inf")] = \
                                    cs_dir / f"ground_muspec_prim_energies_season_cstune{ptype}_pi{cs_p:.2f}_k{cs_k:.2f}_pr{cs_pr:.2f}_e0{e0:.2f}_const_{'inf' if e1 is None else f'e1{e1:.2f}'}.pkl"

                                
                        
                        else:
                            # Handle pairwise=False
                            print("pairwise=False")

                            if cs_p == 1.0 and cs_k == 1.0 and cs_pr == 1.0:
                                # Special case: load only one file without looping over e0
                                e0 = e0_values[0]  # dummy key value for consistency
                                flux_files[(ptype, round(cs_p, 2), round(cs_k, 2), round(cs_pr, 2), round(e0, 2), "inf")] = \
                                    cs_dir / f"surface_fluxes_season{ptype}_pi{cs_p:.2f}_k{cs_k:.2f}.pkl"

                                muspec_files[(ptype, round(cs_p, 2), round(cs_k, 2), round(cs_pr, 2), round(e0, 2), "inf")] = \
                                    cs_dir / f"ground_muspec_prim_energies_season_cstune{ptype}_pi{cs_p:.2f}_k{cs_k:.2f}.pkl"

                            else:
                                for e0 in e0_values:
                                    if e1_list:  # e1 variation case
                                        for e1 in e1_list:
                                            e1_str = "inf" if e1 is None else round(e1, 2)
                                            e1_suffix = "" if e1 is None else f"_e1{e1:.2f}"
                                            flux_files[(ptype, round(cs_p, 2), round(cs_k, 2), round(cs_pr, 2), round(e0, 2), e1_str)] = \
                                                cs_dir / f"surface_fluxes_season{ptype}_pi{cs_p:.2f}_k{cs_k:.2f}_pr{cs_pr:.2f}_e0{e0:.2f}_const{e1_suffix}.pkl"

                                            muspec_files[(ptype, round(cs_p, 2), round(cs_k, 2), round(cs_pr, 2), round(e0, 2), e1_str)] = \
                                                cs_dir / f"ground_muspec_prim_energies_season_cstune{ptype}_pi{cs_p:.2f}_k{cs_k:.2f}_pr{cs_pr:.2f}_e0{e0:.2f}_const{e1_suffix}.pkl"

                                    else:  # e1_list empty or None
                                        flux_files[(ptype, round(cs_p, 2), round(cs_k, 2), round(cs_pr, 2), round(e0, 2), "inf")] = \
                                            cs_dir / f"surface_fluxes_season{ptype}_pi{cs_p:.2f}_k{cs_k:.2f}_pr{cs_pr:.2f}_e0{e0:.2f}_const.pkl"

                                        muspec_files[(ptype, round(cs_p, 2), round(cs_k, 2), round(cs_pr, 2), round(e0, 2), "inf")] = \
                                            cs_dir / f"ground_muspec_prim_energies_season_cstune{ptype}_pi{cs_p:.2f}_k{cs_k:.2f}_pr{cs_pr:.2f}_e0{e0:.2f}_const.pkl"
    # Load the data dynamically for surface fluxes and ground mu energies
    surface_fluxes = {}
    ground_muspec_energies = {}

    for key, file in flux_files.items():
        with open(file, "rb") as f:
            data = pickle.load(f)
            _, cos_thetas, *fluxes = data
            surface_fluxes[key] = {"jan": np.asarray(fluxes[0]), "apr": np.asarray(fluxes[1]), "jul": np.asarray(fluxes[2])}

    for key, file in muspec_files.items():
        with open(file, "rb") as f:
            data = pickle.load(f)
            _, cos_thetas, cr_grid, *fluxes = data
            ground_muspec_energies[key] = {
                "jan": np.asarray(fluxes[0]).swapaxes(0, 1),
                "apr": np.asarray(fluxes[1]).swapaxes(0, 1),
                "jul": np.asarray(fluxes[2]).swapaxes(0, 1),
            }

    for key in surface_fluxes:
        print(f"Key: {key}")  # Print the key
        print(f"Length of cos_thetas: {len(cos_thetas)}")  # Print the length of cos_thetas    
       
        for season, flux in surface_fluxes[key].items():
            print(f"Shape of flux for season {season}: {flux.shape}")  # Print the shape of flux  
    # Create interpolators for surface fluxes
    intp_surface_fluxes = {
        key: {
            season: ip.interp1d(cos_thetas, flux, axis=0, kind="linear")
            for season, flux in surface_fluxes[key].items()
        }
        for key in surface_fluxes
    }

      

    # Create interpolators for ground mu yields
    intp_ground_mu_yields = {
        key: {
            season: [ 
                ip.interp1d(cos_thetas, ground_muspec_energies[key][season][:, ie, :], axis=0, kind="linear")
                for ie in range(len(cr_grid))
            ]
            for season in ground_muspec_energies[key]  # iterate over season keys
        }
        for key in ground_muspec_energies  # iterate over the keys of the ground_muspec_energies dictionary
    }

    #print("Available keys:", list(intp_ground_mu_yields.keys()))

    # Now calculate angles and c_wi globally
    angles = np.degrees(np.arccos(cos_thetas))  # Convert from cos(theta) to theta in degrees
    c_wi = np.diff(cos_thetas)  # Calculate the differential width in cos(theta)

def some_function_that_uses_angles():
    global angles
    if angles is None:
        raise ValueError("angles not initialized. Call initialize_flux_dicts() first.")
    # Use angles here

def some_function_that_uses_c_wi():
    global c_wi
    if c_wi is None:
        raise ValueError("c_wi not initialized. Call initialize_flux_dicts() first.")
    # Use c_wi here




dmnflux = Flux("IceCube")

# Slant depths in [km.w.e.] and angles in [degrees]

_X_MIN = 0.5
_X_MAX = 14

_SLANT_DEPTHS = np.linspace(_X_MIN, _X_MAX, int(2 * (_X_MAX - _X_MIN) + 1))
#step = 0.5  # Since 2 * (X_MAX - X_MIN) + 1 suggests 0.5 spacing
#_SLANT_DEPTHS = np.arange(_X_MIN, _X_MAX + step, step)
# _ANGLES = np.degrees(np.arccos(_X_MIN / _SLANT_DEPTHS))


# Function to compute angles from cos_thetas
def compute_angles():
    global cos_thetas
    if cos_thetas is not None:
        angles = np.degrees(np.arccos(cos_thetas))
        return angles
    else:
        raise ValueError("cos_thetas is not initialized yet.")

slant_depths = _SLANT_DEPTHS
#angles = np.degrees(np.arccos(cos_thetas))
#c_wi = np.diff(cos_thetas)


# Multiplicity vector
#n_mu_vec = np.logspace(0, 2, 100)
n_mu_vec = np.linspace(1, 100, 101)
e_mu_bu_vec = np.logspace(1, 10, 101)

tensor_fname = (
    pathlib.Path(__file__).parent / "water_0.997_1000000_Survival_Probabilities.txt"
)


def load_survival_probability_tensor(file_name=tensor_fname):
    """
    Load survival probability tensor from file.

    Args:
        file_name (str): Name of the file containing the tensor.

    Returns:
        numpy.ndarray: Survival probability tensor.
    """
    return np.reshape(
        np.loadtxt(file_name)[:, 3],
        (
            len(mute_energies),
            len(_SLANT_DEPTHS),
            len(mute_energies),
        ),
    )


utensor = load_survival_probability_tensor().swapaxes(0, 1).swapaxes(1, 2)


def get_bins_and_width_from_centers(vector):
    """
    Get bin edges and widths from bin centers.

    Args:
        vector (numpy.ndarray): Bin centers.

    Returns:
        tuple: Tuple containing bin edges and bin widths.
    """
    vector_log = np.log10(vector)
    steps = vector_log[1] - vector_log[0]
    bins_log = vector_log - 0.5 * steps
    bins_log = np.resize(bins_log, vector_log.size + 1)
    bins_log[-1] = vector_log[-1] + 0.5 * steps
    bins = 10**bins_log
    widths = bins[1:] - bins[:-1]
    return bins, widths

## modified code ----------------------------------------------------------------- #

def _flux(angle, flux_label, ptype=2212, cs_p=1.0, cs_k=1.0, cs_pr = 1.0, e0 =1000.0, e1=None, iecr=None): # added new arguments: ptype and cs
    """
    Calculate the flux.

    
    Args:
        angle (float): Angle in degrees.
        flux_label (str): Flux type ('jan', 'apr', 'jul', 'yields_apr', 'yields_jul').
        ptype (int): Primary particle type (2212 = proton, 402 = helium, 5626 = iron).
        cs (float): Cross-section scaling factor (0.5, 1.0, 1.5).
        iecr (int, optional): Energy index for ground muon spectrum.

    Returns:
        numpy.ndarray: Flux.
    """

    cth = np.cos(np.radians(angle))

    # Directly set the maximum value to 1.0 for cos(theta)
    cos_thetas_max_adjusted = 1.0  # Explicitly set the upper bound to 1.0

    # Check that the min and max of cth are within the bounds of cos_thetas
    assert np.min(cth) >= cos_thetas[0] and np.max(cth) <= cos_thetas_max_adjusted




    if cs_p == 1.0 and cs_k==1.0 and cs_pr==1.0:
        e1 = 'inf'  # Explicitly set e1 to 'inf' when cs_p = 1.0
    else:
        # Find the correct e1 value for cs_p = 1.01
        possible_e1_values = sorted(set(k[5] for k in intp_ground_mu_yields.keys() if k[:5] == (ptype, cs_p, cs_k, cs_pr, e0)))
        e1 = possible_e1_values[0] if possible_e1_values else None  # Pick the first valid e1

       
    key = (ptype, cs_p, cs_k, cs_pr, e0, e1)
    #print('key in helper',key)

    if key not in intp_ground_mu_yields:
        raise KeyError(f"Invalid key: {key}, available keys: {list(intp_ground_mu_yields.keys())}")

        

    
    if flux_label == "daemonflux":
        return mute_energies**-3 * Flux.flux(mute_energies, angle, quantity="muflux")
    elif flux_label in intp_surface_fluxes.get(key, {}):
        return intp_surface_fluxes[key][flux_label](cth)[:dim_ug] # for calling jan, apr, jul
    elif iecr is not None and flux_label in intp_ground_mu_yields.get(key, {}):
        return intp_ground_mu_yields[key][flux_label][iecr](cth)[:dim_ug]
    elif flux_label == "yields_jan":
        assert iecr is not None
        return intp_ground_mu_yields[(ptype, cs_p, cs_k, cs_pr, e0, e1 if e1 is not None else "inf")]["jan"][iecr](cth)[:dim_ug]
    elif flux_label == "yields_apr":
        assert iecr is not None
        return intp_ground_mu_yields[(ptype, cs_p, cs_k, cs_pr, e0, e1 if e1 is not None else "inf")]["apr"][iecr](cth)[:dim_ug]
    elif flux_label == "yields_jul":
        assert iecr is not None
        return intp_ground_mu_yields[(ptype, cs_p, cs_k, cs_pr, e0, e1 if e1 is not None else "inf")]["jul"][iecr](cth)[:dim_ug]

    raise ValueError(f"Unknown flux label '{flux_label}' for ptype={ptype}, cs_p='{cs_p}', cs_k='{cs_k}',  cs_pr='{cs_pr}',e0='{e0}'")
    


## old code ----------------------------------------------------------------- #


def flux(depth, angle, flux_label, ptype, cs_p, cs_k, cs_pr, e0, iecr,e1=None): #only way where e1 and ecr are switches
    """
    Calculate the flux.

    Args:
        depth (float): Depth.
        angle (float): Angle in degrees.
        flux_label (str): Surface flux type.

    Returns:
        numpy.ndarray: Underground flux.
    """
    if np.isscalar(depth):
        if depth <= 0.0:
            return _flux(angle, flux_label, ptype, cs_p, cs_k, cs_pr, e0, e1, iecr)
        else:
            assert np.min(depth) >= slant_depths[0] and np.max(depth) < slant_depths[-1]
        # depth = np.array([depth])
        fl = _flux(angle, flux_label, ptype, cs_p, cs_k, cs_pr, e0, e1, iecr)
        idx = np.argmax(slant_depths > depth)
        frange = (
            utensor[idx - 1 : idx + 1].dot(np.nan_to_num(fl * _e_widths)) / _e_widths
        )
        with np.errstate(all="ignore"):
            return np.nan_to_num(
                np.exp(
                    ip.interp1d(
                        slant_depths[idx - 1 : idx + 1],
                        np.log(frange),
                        axis=0,
                        kind="linear",
                    )(depth)
                )
            )
    else:
        fl = _flux(angle, flux_label, ptype, cs_p, cs_k, cs_pr, e0, e1, iecr)
        frange = utensor.dot(np.nan_to_num(fl * _e_widths)) / _e_widths
        with np.errstate(all="ignore"):
            return np.nan_to_num(
                np.exp(
                    ip.interp1d(
                        slant_depths,
                        np.log(frange),
                        axis=0,
                        kind="linear",
                    )(depth)
                )
            )


def integrated_flux(X, flux_label, ptype, cs_p, cs_k, cs_pr, e0, e1=None, iecr=None):
    """
    Get the integrated flux.

    Args:
        X (float): Depth.
        flux_label (str): Surface flux type.

    Returns:
        numpy.ndarray: Integrated flux.
    """
    integrated_flux = np.zeros_like(mute_energies)
    for icth, cth in enumerate(cos_thetas):
        if X / cth > _X_MAX:
            continue
        fl = flux(X / cth, angles[icth], flux_label, ptype, cs_p, cs_k, cs_pr, e0, iecr, e1) * c_wi[0]

        integrated_flux += fl  # if fl > 0 else 0.0
    return integrated_flux


def rates(depth, angle, flux_label, ptype, cs_p, cs_k, cs_pr, e0, e1, iecr=None,threshold=500):
    """
    Calculate the rates.

    Args:
        depth (float): Depth.
        angle (float): Angle in degrees.
        flux_label (str): Surface flux type.

        threshold: energy above which the rate or flux is integrated in GeV.
    Returns:
        numpy.ndarray: Underground rates.
    """
    ufluxes = flux(depth, angle, flux_label, ptype, cs_p, cs_k, cs_pr, e0, iecr, e1)

    # Ensure mute_energies and ufluxes are arrays
    mute_energies_arr = np.asarray(mute_energies)
    ufluxes_arr = np.asarray(ufluxes)

    # Apply threshold filter
    mask = mute_energies_arr > threshold

    # If ufluxes is 1D, apply mask directly
    if ufluxes_arr.ndim == 1:
        filtered_fluxes = ufluxes_arr[mask]
    else:
        filtered_fluxes = ufluxes_arr[:, mask]  # For higher-dimensional cases

    filtered_energies = mute_energies_arr[mask]

    # Integrate only above threshold
    return np.trapz(filtered_fluxes, filtered_energies)


def integrated_rates(depth, flux_label, ptype, cs_p, cs_k, cs_pr, e0, e1, iecr=None):
    """
    Get the integrated rates.

    Args:
        surf_flux (numpy.ndarray): Surface flux.

    Returns:
        float: Integrated rates.
    """
    return np.trapz(
        [rates(depth, angle, flux_label, ptype, cs_p, cs_k, cs_pr, e0, e1, iecr=None) for angle in angles],
        cos_thetas,
    )


def mean_e(depth, angle, iecr, flux_label, ptype, cs_p, cs_k, cs_pr, e0, e1=None, flcutoff=1e-15):
    """
    Calculate the mean energy.

    Args:
        flux (numpy.ndarray): Flux.
        egrid (numpy.ndarray): Energy grid.
        flcutoff (float): Flux cutoff.

    Returns:
        float: Mean energy.
    """
    fl = flux(depth, angle, flux_label, ptype, cs_p, cs_k, cs_pr, e0, iecr, e1)
    fl[fl <= 0.0] *= 0
    fl = np.nan_to_num(fl)
    assert np.sum(fl) > 0.0, "Flux is zero everywhere"
    flint = np.trapz(fl, mute_energies)
    if flint < flcutoff:
        return 0.0
    return np.trapz((mute_energies * fl), mute_energies) / flint


def integrated_mean_e(depth, ptype, cs_p, cs_k, cs_pr, e0, iecr,  e1=None, flcutoff=1e-15):
    """
    Calculate the mean energy.

    Args:
        flux (numpy.ndarray): Flux.
        egrid (numpy.ndarray): Energy grid.
        flcutoff (float): Flux cutoff.

    Returns:
        float: Mean energy.
    """
    fl = integrated_flux(depth, "yields", ptype, cs_p, cs_k, cs_pr,  e0,e1, iecr=None)
    fl[fl <= 0.0] *= 0
    fl = np.nan_to_num(fl)
    assert np.sum(fl) > 0.0, "Flux is zero everywhere"
    flint = np.trapz(fl, mute_energies)
    if flint < flcutoff:
        return 0.0
    return np.trapz((mute_energies * fl), mute_energies) / flint


def mean_mult(depth, angle, flux_label, ptype, cs_p, cs_k, cs_pr, e0, e1=None, iecr=None,threshold=500):
    """
    Calculate the mean multiplicity.

    Args:
        flux (numpy.ndarray): Flux.

    Returns:
        float: Mean multiplicity.
    """

    ufluxes = flux(depth, angle, flux_label, ptype, cs_p, cs_k, cs_pr, e0, iecr, e1)

    # Apply threshold to energy and flux
    mask = mute_energies > threshold
    filtered_fluxes = ufluxes[mask]
    filtered_energies = mute_energies[mask]

    # Integrate above the threshold
    return np.trapz(filtered_fluxes, filtered_energies)


def mean_bundle_energy(depth, angle, flux_label, ptype, cs_p, cs_k, cs_pr, e0, e1=None, iecr=None):
    """
    Calculate the mean multiplicity.

    Args:
        flux (numpy.ndarray): Flux.

    Returns:
        float: Mean multiplicity.
    """
    return np.trapz(
        mute_energies * flux(depth, angle, flux_label, ptype, cs_p, cs_k, cs_pr, e0, iecr, e1),
        mute_energies,
    )


def integrated_mean_mult(depth, flux_label, ptype, cs_p, cs_k, cs_pr, e0, e1=None, iecr=None):
    """
    Calculate the mean multiplicity.

    Args:
        flux (numpy.ndarray): Flux.

    Returns:
        float: Mean multiplicity.
    """
    return np.trapz(
        integrated_flux(depth, flux_label, ptype, cs_p, cs_k, cs_pr, e0, e1, iecr),
        mute_energies,
    )


def mult_dist(depth, angle, pmodel, flux_label, ptype, cs_p, cs_k, cs_pr, e0, e1=None,norm=True, threshold=500):
    # Unweighted multiplicity vector for all CR energies at specific depth
    mult_vec = np.array(
        [mean_mult(depth, angle, flux_label, ptype, cs_p, cs_k, cs_pr, e0, e1, ei,threshold) for ei, e_cr in enumerate(cr_grid)] #needs yields
    )
    # Truncate to only include energies with multiplicity > 1e-2
    cr_grid_tr = cr_grid[mult_vec > 1e-1]
    mult_vec = mult_vec[mult_vec > 1e-1]

    # Fit log-log spline to multiplicity vs CR energy
    s_ecr = ip.UnivariateSpline(np.log(mult_vec), np.log(cr_grid_tr), s=0, k=1)
    # Fit inverse
    s_nmu = ip.UnivariateSpline(np.log(cr_grid_tr), np.log(mult_vec), s=0, k=1)
    sd_ecr = s_ecr.derivative()

    def ecr(nmu):
        # Ecr(Nmu)
        return np.exp(s_ecr(np.log(nmu)))

    def nmu(ecr):
        # Nmu(Ecr)
        return np.exp(s_nmu(np.log(ecr)))

    def decr(nmu):
        # dNmu/dEcr
        return np.exp(sd_ecr(np.log(nmu)))

    # dN/dNmu = dN/dEcr * dEcr/dNmu
    n_mu_spec = np.zeros_like(n_mu_vec)
    for inm, n_mu in enumerate(n_mu_vec):
        n_mu_spec[inm] = np.sum(pmodel.tot_nucleon_flux(ecr(n_mu))) * 1e-4 * decr(n_mu)
    return n_mu_spec / n_mu_spec[0] if norm is True else n_mu_spec

def mult_dist_test(depth, angle, pmodel, flux_label, ptype, cs_p, cs_k, cs_pr, e0, e1=None,norm=True, threshold=500):
    # Unweighted multiplicity vector for all CR energies at specific depth
    mult_vec = np.array(
        [mean_mult(depth, angle, flux_label, ptype, cs_p, cs_k, cs_pr, e0, e1, ei,threshold) for ei, e_cr in enumerate(cr_grid)] #needs yields
    )
    # Truncate to only include energies with multiplicity > 1e-2
    cr_grid_tr = cr_grid[mult_vec > 1e-1]
    mult_vec = mult_vec[mult_vec > 1e-1]

    # Fit log-log spline to multiplicity vs CR energy
    s_ecr = ip.UnivariateSpline(np.log(mult_vec), np.log(cr_grid_tr), s=0, k=1)
    # Fit inverse
    s_nmu = ip.UnivariateSpline(np.log(cr_grid_tr), np.log(mult_vec), s=0, k=1)
    sd_ecr = s_ecr.derivative()

    def ecr(nmu):
        # Ecr(Nmu)
        return np.exp(s_ecr(np.log(nmu)))

    def nmu(ecr):
        # Nmu(Ecr)
        return np.exp(s_nmu(np.log(ecr)))

    def decr(nmu):
        # dNmu/dEcr
        return np.exp(sd_ecr(np.log(nmu)))

    # dN/dNmu = dN/dEcr * dEcr/dNmu
    n_mu_spec = np.zeros_like(n_mu_vec)
    ecr_nmu = np.zeros_like(n_mu_vec)
    for inm, n_mu in enumerate(n_mu_vec):
        n_mu_spec[inm] = np.sum(pmodel.tot_nucleon_flux(ecr(n_mu))) * 1e-4 * decr(n_mu)
        ecr_nmu[inm] = ecr(n_mu)
    return n_mu_spec / n_mu_spec[0] if norm is True else n_mu_spec, ecr_nmu


def bundle_energy_dist(depth, angle, pmodel, ptype, cs_p, cs_k, cs_pr, e0, e1=None, norm=True):
    # Unweighted multiplicity vector for all CR energies at specific depth
    be_vec = np.array(
        [
            mean_bundle_energy(depth, angle, "yields", ptype, cs_p, cs_k, cs_pr, e0, e1, ei)
            for ei, e_cr in enumerate(cr_grid)
        ]
    )
    # Truncate to only include energies with multiplicity > 1e-2
    cr_grid_tr = cr_grid[be_vec > 1e3]
    be_vec = be_vec[be_vec > 1e3]

    # Fit log-log spline to multiplicity vs CR energy
    s_ecr = ip.UnivariateSpline(np.log(be_vec), np.log(cr_grid_tr), s=0, k=1)
    # Fit inverse
    s_nmu = ip.UnivariateSpline(np.log(cr_grid_tr), np.log(be_vec), s=0, k=1)
    sd_ecr = s_ecr.derivative()

    def ecr(eb):
        # Ecr(eb)
        return np.exp(s_ecr(np.log(eb)))

    def eb(ecr):
        # Nmu(Ecr)
        return np.exp(s_nmu(np.log(ecr)))

    def decr(eb):
        # de_mu/dEcr
        return np.exp(sd_ecr(np.log(eb)))

    # dN/dEmu = dN/dEcr * dEcr/dEmu
    eb_mu_spec = np.zeros_like(e_mu_bu_vec)
    for ieb, eb in enumerate(e_mu_bu_vec):
        eb_mu_spec[ieb] = np.sum(pmodel.tot_nucleon_flux(ecr(eb))) * 1e-4 * decr(eb)
    return eb_mu_spec / eb_mu_spec[0] if norm is True else eb_mu_spec


#def get_integrated_mult_dist(pmodel, norm=True):
#    mult_vec = np.array(
#        [
#            ug_mean_mult(integrated_flux(ground_muspec_prim_energies_apr[ei]))
#            for ei, e_cr in enumerate(cr_grid)
#        ]
#    )
#    cr_grid_tr = cr_grid[mult_vec > 1e-2]
#    mult_vec = mult_vec[mult_vec > 1e-2]
#
#    s_ecr = ip.UnivariateSpline(np.log(mult_vec), np.log(cr_grid_tr))
#    s_nmu = ip.UnivariateSpline(np.log(cr_grid_tr), np.log(mult_vec))
#    sd_ecr = s_ecr.derivative()
#
#    ecr = lambda nmu: np.exp(s_ecr(np.log(nmu)))
#    nmu = lambda ecr: np.exp(s_nmu(np.log(ecr)))
#    decr = lambda nmu: np.exp(sd_ecr(np.log(nmu)))
#    n_mu_spec = np.zeros_like(n_mu_vec)
#    for inm, n_mu in enumerate(n_mu_vec):
#        n_mu_spec[inm] = np.sum(pmodel.tot_nucleon_flux(ecr(n_mu))) * 1e4 * decr(n_mu)
#    return n_mu_spec / n_mu_spec[0]