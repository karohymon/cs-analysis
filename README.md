# cs-analysis
Studies of the impact of variations in the pion-air and kaon-air cross section on muon and muon neutrino flux from atmospheric air showers.

Old stufies are located in **crosssection_initialstudy/** and **Icecube_diffuse_underground_bundles/old_dev_scripts/**  and **example_files/** (these folders have not yet been cleaned and some imports might fail due to updated path directories.)





Current analysis: fit sensitivity region of cross section modification for bundles bundles in IceCube

All scripts can be run with the virtual env: source /hetghome/khymon/pip_envs/cs_env/bin/activate or by setting it as jupyter kernel

## Analysis Pipeline
Analysis Pipeline in **IceCube_diffuse_underground_bundles/**:

1. **Precompute_bundles_cs_bsplines.py** for a,b,c = +/- 0.05 and 0. Cross section modification via bspline is implemented in **../cs_modifier.py**

2. **../test_bsplinesclass.ipynb**: calculate dsigma/da, dsigma/db/ dsimga/dc, save to file and plot cross section modification

3. **dNdNmu_fit.ipynb**: fit muon multiplicity spectrum and save fit of optimal a,b,c to file by interpolating uncertainties on muon multiplicity form Stef's paper. Plot moficiation of multiplicity spectrum.

4. **error_backpropa.ipynb**: Backprogapate uncertainty from fit to surface flux and cross section. Money Plot.


## Helpers and plotting scripts:

- **cs_model_mceq.ipynb**: Plot cross section pion-air, K-air, proton-air for differnt hadronic models.
- ***bspline_example.ipynb**: Spline development and test plots

in **IceCube_diffuse_underground_bundles/**:

- **mceq_underground_helper_forsplines.py**: Helper functions for underground flux, muon multiplicity spectrum and global variable definition
- **Precompute_muonflux_primaryfraction.py**: Calculation of muon underground flux and parent particle fraction. Corresponding plot in **undergroundspectrum_parent.ipynb**
- **plot_dNdNmu.ipynb**: plot multiplicity intensity spectrum with larger cross section variation
