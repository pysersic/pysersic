import pytest
import numpy as np
from pysersic import FitSingle,FitMulti, priors, rendering
from astropy.convolution import Gaussian2DKernel
from jax.random import PRNGKey
import arviz
from numpyro import distributions as dist,infer,sample, optim
from pysersic.pysersic import train_numpyro_svi_early_stop
from pysersic.priors import estimate_sky




im_s = np.zeros((40,40))
rng = np.random.default_rng(seed=10)
im_s += rng.normal(scale = 0.05, size = (40,40))
rms = np.ones(im_s.shape)*0.05
psf = Gaussian2DKernel(x_stddev=2.5).array
renderer = rendering.HybridRenderer((40,40), psf)
im_s = im_s + renderer.render_source( dict(xc = 20.,yc=20., flux = 200.), 'pointsource' )


@pytest.mark.parametrize('sky_type', ['none', 'flat', 'tilted-plane'])
def test_FitSingle_map(sky_type):
        props = priors.SourceProperties(im_s)
        
        prior = props.generate_prior(profile_type='pointsource', sky_type=sky_type)
        print (prior)
        fitter_single = FitSingle(im_s,rms,psf,prior)

        map_dict = fitter_single.find_MAP(rkey = PRNGKey(10))
        assert map_dict['xc'] == pytest.approx(20.017, rel = 1e-3)        
        assert map_dict['yc'] == pytest.approx(19.996, rel = 1e-3)
        assert map_dict['flux'] == pytest.approx(199.418, rel = 5e-2)

@pytest.mark.parametrize('method',['laplace','svi-mvn','svi-flow'])
def test_FitSingle_posterior(method):
        props = priors.SourceProperties(im_s)
        prior = props.generate_prior(profile_type='pointsource')

        fitter_single = FitSingle(im_s,rms,psf,prior)

        res = fitter_single.estimate_posterior(method, rkey = PRNGKey(3))
        post_sum = res.summary()

        assert post_sum['mean']['flux'] == pytest.approx(199.4, rel = 1e-2)
        assert post_sum['sd']['flux'] == pytest.approx(0.44, rel = 1e-1)


        assert post_sum['mean']['xc'] == pytest.approx(20.02, rel = 1e-2)
        assert post_sum['sd']['xc'] == pytest.approx(0.0085, rel = 2e-1)
        
        assert post_sum['mean']['yc'] == pytest.approx(19.99, rel = 1e-2)
        assert post_sum['sd']['yc'] ==  pytest.approx(0.008, rel = 1e-1)


def test_FitSingle_sample():
        props = priors.SourceProperties(im_s)
        prior = props.generate_prior(profile_type='pointsource')

        fitter_single = FitSingle(im_s,rms,psf,prior)

        res = fitter_single.sample(num_samples = 500,num_warmup = 500, num_chains = 1,rkey = PRNGKey(5))
        post_sum = arviz.summary(res.idata)

        assert post_sum['mean']['flux'] == pytest.approx(199.4, rel = 5e-2)
        assert post_sum['sd']['flux'] == pytest.approx(0.45, rel = 5e-2)

        assert post_sum['mean']['xc'] == pytest.approx(20.02, rel = 1e-2)
        assert post_sum['sd']['xc'] == pytest.approx(0.007, rel = 2e-1)
        
        assert post_sum['mean']['yc'] == pytest.approx(19.99, rel = 1e-2)
        assert post_sum['sd']['yc'] ==  pytest.approx(0.0085, rel = 1e-1)


im_m = np.zeros((40,40))
rng = np.random.default_rng(seed=12)
im_m += rng.normal(scale = 0.05, size = (40,40))
rms = np.ones(im_m.shape)*0.05
psf = Gaussian2DKernel(x_stddev=2.5).array
renderer = rendering.HybridRenderer((40,40), psf)
im_m = im_m + renderer.render_source( dict(xc = 10.,yc = 30., flux  = 150.), 'pointsource' )
im_m = im_m + renderer.render_source( dict(yc = 10.,xc = 30., flux  = 150.), 'pointsource' )


cat = {}
cat['x'] = [10.,30.]
cat['y'] = [30.,10.]
cat['r']= [0,0]
cat['flux'] = [150.,150.]
cat['type'] = ['pointsource','pointsource']
mp = priors.PySersicMultiPrior(catalog = cat, sky_type='none')
multi_fitter = FitMulti(im_m,rms,psf, mp)

sky_guess, sky_guess_err, n_pix_sky = estimate_sky(im_m)
@pytest.mark.parametrize('sky_type', ['none', 'flat', 'tilted-plane'])
def test_FitMulti_map(sky_type):
        mp = priors.PySersicMultiPrior(catalog = cat, sky_type=sky_type,sky_guess  = sky_guess, sky_guess_err=sky_guess_err)
        multi_fitter = FitMulti(im_m,rms,psf, mp)

        map_dict = multi_fitter.find_MAP(rkey = PRNGKey(10))

        assert map_dict['source_0']['xc'] == pytest.approx(10., rel = 1e-3)        
        assert map_dict['source_0']['yc'] == pytest.approx(30., rel = 1e-3)
        assert map_dict['source_0']['flux'] == pytest.approx(150.4, rel = 1e-3)

        assert map_dict['source_1']['xc'] == pytest.approx(30., rel = 1e-3)        
        assert map_dict['source_1']['yc'] == pytest.approx(10., rel = 1e-3)
        assert map_dict['source_1']['flux'] == pytest.approx(150., rel = 1e-3)

@pytest.mark.parametrize('method',['laplace','svi-mvn','svi-flow'])
def test_FitMulti_posterior(method):
        res = multi_fitter.estimate_posterior(method, rkey = PRNGKey(3))
        post_sum = res.summary()
        assert post_sum['mean']['flux_0'] == pytest.approx(150.4, rel = 1e-2)
        assert post_sum['sd']['flux_0'] == pytest.approx(0.42, rel = 1e-1)
        assert post_sum['mean']['flux_1'] == pytest.approx(150., rel = 1e-2)
        assert post_sum['sd']['flux_1'] == pytest.approx(0.45, rel = 1e-1)

        assert post_sum['mean']['xc_0'] == pytest.approx(10.0, rel = 1e-2)
        assert post_sum['sd']['xc_0'] == pytest.approx(0.01, rel = 1e-1)
        assert post_sum['mean']['xc_1'] == pytest.approx(30., rel = 1e-2)
        assert post_sum['sd']['xc_1'] == pytest.approx(0.01, rel = 1e-1)

        assert post_sum['mean']['yc_0'] == pytest.approx(30.0, rel = 1e-2)
        assert post_sum['sd']['yc_0'] == pytest.approx(0.01, rel = 1e-1)
        assert post_sum['mean']['yc_1'] == pytest.approx(10., rel = 1e-2)
        assert post_sum['sd']['yc_1'] == pytest.approx(0.01, rel = 1e-1)

def test_FitMulti_sample():
        res = multi_fitter.sample(num_samples = 500,num_warmup = 500, num_chains = 1,rkey = PRNGKey(5))
        post_sum = arviz.summary(res.idata)

        assert post_sum['mean']['flux_0'] == pytest.approx(150.4, rel = 1e-2)
        assert post_sum['sd']['flux_0'] == pytest.approx(0.44, rel =1e-1)
        assert post_sum['mean']['flux_1'] == pytest.approx(150.1, rel = 1e-2)
        assert post_sum['sd']['flux_1'] == pytest.approx(0.42, rel = 1e-1)

        assert post_sum['mean']['xc_0'] == pytest.approx(10.0, rel = 1e-2)
        assert post_sum['sd']['xc_0'] == pytest.approx(0.01, rel = 2e-1)
        assert post_sum['mean']['xc_1'] == pytest.approx(30., rel = 1e-2)
        assert post_sum['sd']['xc_1'] == pytest.approx(0.01, rel = 1e-1)

        assert post_sum['mean']['yc_0'] == pytest.approx(30.0, rel = 1e-2)
        assert post_sum['sd']['yc_0'] == pytest.approx(0.01, rel = 1e-1)
        assert post_sum['mean']['yc_1'] == pytest.approx(10., rel = 1e-2)
        assert post_sum['sd']['yc_1'] == pytest.approx(0.01, rel = 1e-1)

def test_train_numpyro_svi():
    def model():
        a = sample('a', dist.Normal())

    guide =infer.autoguide.AutoDelta(model)
    svi_kernel = infer.SVI(model,guide, optim.Adam(0.001), loss = infer.Trace_ELBO())
    res = train_numpyro_svi_early_stop(svi_kernel)
    assert float(res.params['a_auto_loc']) == pytest.approx(0, abs = 1e-2)