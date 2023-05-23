import pytest
import jax.numpy as jnp
import numpy as np
from pysersic.priors import BasePrior, PySersicMultiPrior, PySersicSourcePrior, SourceProperties, estimate_sky
from numpyro.handlers import seed
from numpyro import distributions as dist

prof_names = ['sersic','doublesersic','pointsource','exp','dev']
prof_vars = [ ['xc','yc','flux','r_eff','n','ellip','theta'],
        ['xc','yc','flux','f_1', 'r_eff_1','n_1','ellip_1', 'r_eff_2','n_2','ellip_2','theta'],
        ['xc','yc','flux'],
        ['xc','yc','flux','r_eff','ellip','theta'],
        ['xc','yc','flux','r_eff','ellip','theta'],]

@pytest.mark.parametrize('mask', [None, jnp.zeros((100,100))])
def test_SourceProperties(mask):
    test_im = jnp.ones((100,100))
    SP = SourceProperties(test_im, mask = mask)
    SP.measure_properties()
    attribues = ['sky_guess','sky_guess_err', 'flux_guess','flux_guess_err', 'r_eff_guess','r_eff_guess_err','xc_guess','yc_guess']
    for var in attribues:
        assert hasattr(SP, var)

@pytest.mark.parametrize('prof, var_names', zip(prof_names,prof_vars) )
def test_prior_gen(prof, var_names):
    props = SourceProperties(jnp.ones((100,100)) )
    prior_class = props.generate_prior(profile_type = prof)
    assert prior_class.check_vars(verbose = True)

def test_sky_sampling():
    x = jnp.arange(100)
    X,Y = jnp.meshgrid(x,x)

    prior_class = BasePrior(sky_type = 'none')
    with seed(rng_seed=1):
        params_1 = prior_class.sample_sky(X,Y)
    assert params_1 == 0

    prior_class = BasePrior(sky_type = 'flat', sky_guess = 0, sky_guess_err = 1)
    with seed(rng_seed=1):
        params_2 = prior_class.sample_sky(X,Y)
    assert params_2.shape == () # Should be single value

    prior_class = BasePrior(sky_type = 'tilted-plane', sky_guess = 0, sky_guess_err = 1)
    with seed(rng_seed=1):
        params_3 = prior_class.sample_sky(X,Y)
    assert params_3.shape == (100,100) # Should be 2D array 

def test_PySersicSourcePrior():
    prior = PySersicSourcePrior('pointsource')
    prior.set_gaussian_prior('xc', 10.,1.)
    prior.set_gaussian_prior('yc', 10.,1.)
    prior.set_uniform_prior('flux', 0, 100)
    with seed(rng_seed=1):
        params = prior()
    assert params[0] == pytest.approx(8.852981, rel=1e-5)
    assert params[1] == pytest.approx(8.907836, rel=1e-5)
    assert params[2] == pytest.approx(37.12473, rel=1e-5)

def test_PySersicMultiPrior():
    catalog = {}
    catalog['x'] = [10,15,20]
    catalog['y'] = [20,15,10]
    catalog['flux'] = [100,100,100]
    catalog['r'] = [5,5,5]
    catalog['theta'] = [0,0,0]
    catalog['type'] = ['pointsource','exp','sersic']
    mp = PySersicMultiPrior(catalog)
    with seed(rng_seed=1):
        params = mp()

    assert params[0] == pytest.approx([ 8.852981, 18.907837, 93.42897 ], rel=1e-5)
    assert params[1] == pytest.approx([1.5656178e+01,  1.3875072e+01,  1.2103964e+02,
                 9.109129e+00,  3.2803586e-01, -4.2930365e-02] , rel=1e-5)
    assert params[2] == pytest.approx([ 1.82923260e+01,  1.13821745e+01,  1.21627632e+02,
                3.357822e+00,  7.022609e+00,  8.82269740e-02,
                -6.36531591e-01 ], rel=1e-5)
    
def test_sky_estimate():
    rng = np.random.default_rng(1234567)
    test_im = rng.normal(size = (100,100))
    med,std,npix = estimate_sky(test_im, n_pix_sample= 5)
    assert med == pytest.approx(0., abs = 1e-2)
    assert std == pytest.approx(1., abs = 1e-2)
    assert npix == 1900