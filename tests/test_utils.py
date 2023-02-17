import pytest
import jax.numpy as jnp
import pysersic.utils as utils
from numpyro.handlers import seed
from numpyro import distributions as dist
#TODO need to re-do tests

prof_names = ['sersic','doublesersic','pointsource','exp','dev']
prof_vars = [ ['xc','yc','flux','r_eff','n','ellip','theta'],
        ['xc','yc','flux','f_1', 'r_eff_1','n_1','ellip_1', 'r_eff_2','n_2','ellip_2','theta'],
        ['xc','yc','flux'],
        ['xc','yc','flux','r_eff','ellip','theta'],
        ['xc','yc','flux','r_eff','ellip','theta'],]

@pytest.mark.parametrize('prof, var_names', zip(prof_names,prof_vars) )
def test_prior_gen(prof, var_names):
    image = jnp.ones((100,100))
    prior_dict = utils.autoprior(image, prof)
    for k in prior_dict.keys():
        assert k in var_names


def test_sky_sampling():
    sky_prior = dict(sky0 =dist.TransformedDistribution(
                            dist.Normal(),
                            dist.transforms.AffineTransform(0.0,1e-4),),
                    sky1 =  dist.TransformedDistribution(
                            dist.Normal(),
                            dist.transforms.AffineTransform(0.0,1e-5),),
                    sky2 = dist.TransformedDistribution(
                            dist.Normal(),
                            dist.transforms.AffineTransform(0.0,1e-5),) )

    with seed(rng_seed=1):
        params_1 = utils.sample_sky(sky_prior, None)
    assert params_1 == 0
    
    with seed(rng_seed=1):
        params_2 = utils.sample_sky(sky_prior, 'flat')
    assert params_2.shape == () # Should be single value

    with seed(rng_seed=1):
        params_3 = utils.sample_sky(sky_prior, 'tilted-plane')
    assert params_3.shape == (3,) # Should be array of 3 values

    