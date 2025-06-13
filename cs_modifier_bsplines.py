import numpy as np
import matplotlib.pyplot as plt
import mceq_config as config
from MCEq.core import MCEqRun
import crflux.models as pm
from MCEq.data import InteractionCrossSections
from scipy.interpolate import CubicSpline
from utils.helpers import *
from utils import config_handler
import click
import os
from scipy.interpolate import splrep,splev

class ModIntCrossSections(InteractionCrossSections):
    def __init__(self, mceq_hdf_db, interaction_model="SIBYLL2.3c", ptype = 211, a=None,  b=None, c=None):
        self.a = a  #scaling factors in array shape
        self.b = b
        self.c = c
        self.ptype = ptype
        self.interaction_model = interaction_model
        super().__init__(mceq_hdf_db, interaction_model)  # Call parent constructor

    def bsplines_intp(self, a=None, b=None, c=None):
        log_e_grid = np.log10(self.energy_grid.c)

        # Define 3 control points (in log10(E))
        x_knots = np.array([-3,  0., 2., 3.0, 4.5, 6.0,7., 9.,11.])         # log10(E) values (e.g. 1e3, 3e4, 1e6)
        y_knots = np.array([0.,0., 0., a, b, c ,0., 0.,  0.])        # Values at the control points

        # Fit a quadratic B-spline (k=2), with s=0 to interpolate exactly
        tck = splrep(x_knots, y_knots, k=2, s=0)

        # Evaluate the spline at all grid points
        y_spline = splev(log_e_grid, tck)

        return y_spline

    
    def modify_cs(self, a=None, b=None, c=None):
        """Modify pion and kaon interaction cross-sections in two energy regions."""
        
        # Set defaults
        if a is None:
            a = self.a
        if b is None:
            b = self.b
        if c is None:
            c = self.c # Use class attribute if not passed explicitly
            
        logE = np.log10(self.energy_grid.c) # get log10 at bin center

        # get cross section modification factor:
        mod_factor = self.bsplines_intp( a, b, c)
        self.index_d[self.ptype] *= (1+mod_factor)#cs * (1+modfactor) 

    def load(self, interaction_model):
        """Load the interaction model and apply the modification function."""
        super().load(interaction_model)
        self.modify_cs()



