import pytest
import jax.numpy as jnp
from pysersic import FitSingle


prof_names = ['sersic','doublesersic','pointsource','exp','dev']
prof_vars = [ ['x_0','y_0','flux','r_eff','n','ellip','theta'],
        ['x_0','y_0','flux','f_1', 'r_eff_1','n_1','ellip_1', 'r_eff_2','n_2','ellip_2','theta'],
        ['x_0','y_0','flux'],
        ['x_0','y_0','flux','r_eff','ellip','theta'],
        ['x_0','y_0','flux','r_eff','ellip','theta'],]


#Implement some test of initialization etc, but not direcrtly fitting