import pytest
import jax.numpy as jnp
from pysersic.priors import BasePrior, autoprior
from numpyro.handlers import seed
from numpyro import distributions as dist

prof_names = ['sersic','doublesersic','pointsource','exp','dev']
prof_vars = [ ['xc','yc','flux','r_eff','n','ellip','theta'],
        ['xc','yc','flux','f_1', 'r_eff_1','n_1','ellip_1', 'r_eff_2','n_2','ellip_2','theta'],
        ['xc','yc','flux'],
        ['xc','yc','flux','r_eff','ellip','theta'],
        ['xc','yc','flux','r_eff','ellip','theta'],]

@pytest.mark.parametrize('prof, var_names', zip(prof_names,prof_vars) )
def test_prior_gen(prof, var_names):
    image = jnp.ones((100,100))
    prior_class = autoprior(image, prof)
    assert prior_class.check_vars(verbose = True)


def test_sky_sampling():
    x = jnp.arange(100)
    X,Y = jnp.meshgrid(x,x)

    prior_class = BasePrior(sky_type = 'none')
    with seed(rng_seed=1):
        params_1 = prior_class.sample_sky(X,Y)
    assert params_1 == 0

    prior_class = BasePrior(sky_type = 'flat')
    with seed(rng_seed=1):
        params_2 = prior_class.sample_sky(X,Y)
    assert params_2.shape == () # Should be single value

    prior_class = BasePrior(sky_type = 'tilted-plane')
    with seed(rng_seed=1):
        params_3 = prior_class.sample_sky(X,Y)
    assert params_3.shape == (100,100) # Should be 2D array 

    