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


# Define base directory
base_dir = pathlib.Path(__file__).parent
cs_dir = pathlib.Path("/hetghome/khymon/cs-files/smooth-transition")  # Cross-section tuned files

# Generate flux file paths dynamically - adpt here
flux_files = {
    (ptype, round(cs_p1,2), round(cs_p2, 1), round(cs_k2, 1)): cs_dir / f"surface_fluxes_season{ptype}_pi{cs_p1:.2f}_{cs_p2:.2f}_k1.00_{cs_k2:.2f}_e0{e0:.1f}_e1{e1:.1f}const.pkl"
    for ptype in [2212]#, 402, 5626]
    for cs_p1 in [1.00]#[0.95,1.00,1.05]#np.round(np.arange(0.95, 1.06, 0.05), 1)#np.round(np.arange(0.95, 1.06, 0.05), 1)  # Ensure precise values
    for cs_p2 in [1.1,1.0,0.9]#np.round(np.arange(0.90, 1.1, 0.5), 1)#np.round(np.arange(0.5, 1.6, 0.1), 1) 
    for cs_k2 in [0.9, 1.0, 1.1]
    for e0 in [1.e3]
    for e1 in [1.e4]
}

muspec_files = {
    (ptype, round(cs_p1,2), round(cs_p2, 1), round(cs_k2, 1)): cs_dir / f"ground_muspec_prim_energies_season_cstune{ptype}_pi{cs_p1:.2f}_{cs_p2:.2f}_k1.00_{cs_k2:.2f}_e0{e0:.1f}_e1{e1:.1f}const.pkl"
    for ptype in [2212]#, 402, 5626]
    for cs_p1 in [1.00]#[0.95,1.00,1.05]#np.round(np.arange(0.95, 1.06, 0.05), 1)#np.round(np.arange(0.95, 1.06, 0.05), 1)  # Ensure precise values
    for cs_p2 in [1.1,1.0,0.9]#[0.9,1.0,1.1]#np.round(np.arange(0.90, 1.1, 0.5), 1)#np.round(np.arange(0.5, 1.6, 0.1), 1) 
    for cs_k2 in [0.9, 1.0, 1.1]
    for e0 in [1.e3]
    for e1 in [1.e4]
    # k is kept constant for now
}



# Load and store fluxes dynamically
surface_fluxes = {}
ground_muspec_energies = {}

#assign seasonal fluxes
for (ptype, cs_p1, cs_p2, cs_k2), file in flux_files.items():
    with open(file, "rb") as f:
        data = pickle.load(f)
        _, cos_thetas, *fluxes = data  # Unpack all fluxes
        surface_fluxes[(ptype, cs_p1, cs_p2, cs_k2)] = {
            "jan": np.asarray(fluxes[0]),
            "apr": np.asarray(fluxes[1]),
            "jul": np.asarray(fluxes[2]),
        }

for (ptype, cs_p1, cs_p2, cs_k2), file in muspec_files.items():
    with open(file, "rb") as f:
        data = pickle.load(f)
        _, cos_thetas, cr_grid, *fluxes = data
        ground_muspec_energies[(ptype, cs_p1, cs_p2, cs_k2)] = {
            "jan": np.asarray(fluxes[0]).swapaxes(0, 1),
            "apr": np.asarray(fluxes[1]).swapaxes(0, 1),
            "jul": np.asarray(fluxes[2]).swapaxes(0, 1),
        }

# Create interpolators dynamically
intp_surface_fluxes = {
    (ptype, cs_p1, cs_p2, cs_k2): {
        season: ip.interp1d(cos_thetas, flux, axis=0, kind="linear")
        for season, flux in surface_fluxes[(ptype, cs_p1, cs_p2, cs_k2)].items()
    }
    for (ptype, cs_p1, cs_p2, cs_k2) in surface_fluxes
}

intp_ground_mu_yields = {
    (ptype, cs_p1, cs_p2, cs_k2): {
        season: [
            ip.interp1d(cos_thetas, ground_muspec_energies[(ptype, cs_p1, cs_p2, cs_k2)][season][:, ie, :], axis=0, kind="linear")
            for ie in range(len(cr_grid))
        ]
        for season in ground_muspec_energies[(ptype, cs_p1, cs_p2, cs_k2)]
    }
    for (ptype, cs_p1, cs_p2, cs_k2) in ground_muspec_energies
}




dmnflux = Flux("IceCube")

# Slant depths in [km.w.e.] and angles in [degrees]

_X_MIN = 0.5
_X_MAX = 14

_SLANT_DEPTHS = np.linspace(_X_MIN, _X_MAX, int(2 * (_X_MAX - _X_MIN) + 1))
# _ANGLES = np.degrees(np.arccos(_X_MIN / _SLANT_DEPTHS))

slant_depths = _SLANT_DEPTHS
angles = np.degrees(np.arccos(cos_thetas))
c_wi = np.diff(cos_thetas)


# Multiplicity vector
# n_mu_vec = np.logspace(0, 2, 100)
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

def _flux(angle, flux_label, ptype=2212, cs_p1=1.0, cs_p2 = 1.0, cs_k2= 1.0, iecr=None): # added new arguments: ptype and cs
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
    assert np.min(cth) >= cos_thetas[0] and np.max(cth) <= cos_thetas[-1]

    key = (ptype, cs_p1, cs_p2, cs_k2)

    if flux_label == "daemonflux":
        return mute_energies**-3 * Flux.flux(mute_energies, angle, quantity="muflux")
    elif flux_label in intp_surface_fluxes.get(key, {}):
        return intp_surface_fluxes[key][flux_label](cth)[:dim_ug] # for calling jan, apr, jul
    elif iecr is not None and flux_label in intp_ground_mu_yields.get(key, {}):
        return intp_ground_mu_yields[key][flux_label][iecr](cth)[:dim_ug]
    elif flux_label == "yields_jan":
        assert iecr is not None
        return intp_ground_mu_yields[(ptype, cs_p1, cs_p2, cs_k2)]["jan"][iecr](cth)[:dim_ug]
    elif flux_label == "yields_apr":
        assert iecr is not None
        return intp_ground_mu_yields[(ptype, cs_p1, cs_p2, cs_k2)]["apr"][iecr](cth)[:dim_ug]
    elif flux_label == "yields_jul":
        assert iecr is not None
        return intp_ground_mu_yields[(ptype, cs_p1, cs_p2, cs_k2)]["jul"][iecr](cth)[:dim_ug]

    raise ValueError(f"Unknown flux label '{flux_label}' for ptype={ptype}, cs_p1='{cs_p1}', cs_p2 ='{cs_p2}', cs_k2={cs_k2}")
    


## old code ----------------------------------------------------------------- #


def flux(depth, angle, flux_label, ptype, cs_p1, cs_p2, cs_k2, iecr):
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
            return _flux(angle, flux_label, ptype, cs_p1, cs_p2, cs_k2, iecr)
        else:
            assert np.min(depth) >= slant_depths[0] and np.max(depth) < slant_depths[-1]
        # depth = np.array([depth])
        fl = _flux(angle, flux_label, ptype, cs_p1, cs_p2, cs_k2, iecr)
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
        fl = _flux(angle, flux_label, ptype, cs_p1, cs_p2, cs_k2, iecr)
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


def integrated_flux(X, flux_label, ptype,cs_p1, cs_p2, cs_k2, iecr=None):
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
        fl = flux(X / cth, angles[icth], flux_label, ptype, cs_p1, cs_p2, cs_k2, iecr) * c_wi[0]

        integrated_flux += fl  # if fl > 0 else 0.0
    return integrated_flux


def rates(depth, angle, flux_label, ptype, cs_p1, cs_p2, cs_k2, iecr=None):
    """
    Calculate the rates.

    Args:
        depth (float): Depth.
        angle (float): Angle in degrees.
        flux_label (str): Surface flux type.

    Returns:
        numpy.ndarray: Underground rates.
    """
    ufluxes = flux(depth, angle, flux_label, ptype, cs_p1, cs_p2, cs_k2, iecr)

    rates = np.trapz(np.atleast_2d(ufluxes), mute_energies, axis=1)
    return rates


def integrated_rates(depth, flux_label, ptype, cs_p1, cs_p2, cs_k2, iecr=None):
    """
    Get the integrated rates.

    Args:
        surf_flux (numpy.ndarray): Surface flux.

    Returns:
        float: Integrated rates.
    """
    return np.trapz(
        [rates(depth, angle, flux_label, ptype, cs_p1, cs_p2, cs_k2, iecr=None) for angle in angles],
        cos_thetas,
    )


def mean_e(depth, angle, iecr, flux_label, ptype, cs_p1, cs_p2, cs_k2, flcutoff=1e-15):
    """
    Calculate the mean energy.

    Args:
        flux (numpy.ndarray): Flux.
        egrid (numpy.ndarray): Energy grid.
        flcutoff (float): Flux cutoff.

    Returns:
        float: Mean energy.
    """
    fl = flux(depth, angle, flux_label, ptype, cs_p1, cs_p2, cs_k2, iecr)
    fl[fl <= 0.0] *= 0
    fl = np.nan_to_num(fl)
    assert np.sum(fl) > 0.0, "Flux is zero everywhere"
    flint = np.trapz(fl, mute_energies)
    if flint < flcutoff:
        return 0.0
    return np.trapz((mute_energies * fl), mute_energies) / flint


def integrated_mean_e(depth, ptype, cs_p1, cs_p2, cs_k2, iecr, flcutoff=1e-15):
    """
    Calculate the mean energy.

    Args:
        flux (numpy.ndarray): Flux.
        egrid (numpy.ndarray): Energy grid.
        flcutoff (float): Flux cutoff.

    Returns:
        float: Mean energy.
    """
    fl = integrated_flux(depth, "yields", ptype, cs_p1, cs_p2, cs_k2, iecr=None)
    fl[fl <= 0.0] *= 0
    fl = np.nan_to_num(fl)
    assert np.sum(fl) > 0.0, "Flux is zero everywhere"
    flint = np.trapz(fl, mute_energies)
    if flint < flcutoff:
        return 0.0
    return np.trapz((mute_energies * fl), mute_energies) / flint


def mean_mult(depth, angle, flux_label, ptype, cs_p1, cs_p2, cs_k2, iecr=None):
    """
    Calculate the mean multiplicity.

    Args:
        flux (numpy.ndarray): Flux.

    Returns:
        float: Mean multiplicity.
    """
    return np.trapz(
        flux(depth, angle, flux_label, ptype, cs_p1, cs_p2, cs_k2, iecr),
        mute_energies,
    )


def mean_bundle_energy(depth, angle, flux_label, ptype, cs_p1, cs_p2, cs_k2, iecr=None):
    """
    Calculate the mean multiplicity.

    Args:
        flux (numpy.ndarray): Flux.

    Returns:
        float: Mean multiplicity.
    """
    return np.trapz(
        mute_energies * flux(depth, angle, flux_label, ptype, cs_p1, cs_p2, cs_k2, iecr),
        mute_energies,
    )


def integrated_mean_mult(depth, flux_label, ptype, cs_p1, cs_p2, cs_k2, iecr=None):
    """
    Calculate the mean multiplicity.

    Args:
        flux (numpy.ndarray): Flux.

    Returns:
        float: Mean multiplicity.
    """
    return np.trapz(
        integrated_flux(depth, flux_label, ptype, cs_p1, cs_p2, cs_k2, iecr),
        mute_energies,
    )


def mult_dist(depth, angle, pmodel, flux_label, ptype, cs_p1, cs_p2, cs_k2, norm=True):
    # Unweighted multiplicity vector for all CR energies at specific depth
    mult_vec = np.array(
        [mean_mult(depth, angle, flux_label, ptype, cs_p1, cs_p2, cs_k2, ei) for ei, e_cr in enumerate(cr_grid)] #needs yields
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


def bundle_energy_dist(depth, angle, pmodel, ptype, cs_p1, cs_p2, cs_k2, norm=True):
    # Unweighted multiplicity vector for all CR energies at specific depth
    be_vec = np.array(
        [
            mean_bundle_energy(depth, angle, "yields", ptype, cs_p1, cs_p2, cs_k2, ei)
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
