import pytest
import numpy as np
from pysersic import FitSingle,FitMulti, priors, rendering
from astropy.convolution import Gaussian2DKernel
import matplotlib.pyplot as plt
from jax.random import PRNGKey
import arviz

im = np.zeros((40,40))
rng = np.random.default_rng(seed=10)
im += rng.normal(scale = 0.05, size = (40,40))
rms = np.ones(im.shape)*0.05
psf = Gaussian2DKernel(x_stddev=2.5).array
renderer = rendering.HybridRenderer((40,40), psf)
im = im + renderer.render_pointsource(20.,20., 200.)

prior = priors.generate_pointsource_prior(im)

fitter_single = FitSingle(im,rms,psf,prior)

def test_FitSingle_map():
        map_dict = fitter_single.find_MAP(rkey = PRNGKey(10))
        assert map_dict['xc'] == pytest.approx(20.017, rel = 1e-3)        
        assert map_dict['yc'] == pytest.approx(19.996, rel = 1e-3)
        assert map_dict['flux'] == pytest.approx(199.418, rel = 1e-3)

@pytest.mark.parametrize('method',['laplace','svi-flow'])
def test_FitSingle_posterior(method):
        post_sum = fitter_single.estimate_posterior(method, rkey = PRNGKey(3))

        assert post_sum['mean']['flux'] == pytest.approx(199.4, rel = 1e-2)
        assert post_sum['sd']['flux'] == pytest.approx(0.43, rel = 5e-2)

        assert post_sum['mean']['xc'] == pytest.approx(20.02, rel = 1e-2)
        assert post_sum['sd']['xc'] == pytest.approx(0.0085, rel = 1e-1)
        
        assert post_sum['mean']['yc'] == pytest.approx(19.99, rel = 1e-2)
        assert post_sum['sd']['yc'] ==  pytest.approx(0.0085, rel = 1e-1)


def test_FitSingle_sample():
        res = fitter_single.sample(num_samples = 500,num_warmup = 500, num_chains = 1,rkey = PRNGKey(5))
        post_sum = arviz.summary(res.idata)

        assert post_sum['mean']['flux'] == pytest.approx(199.4, rel = 1e-2)
        assert post_sum['sd']['flux'] == pytest.approx(0.46, rel = 1e-2)

        assert post_sum['mean']['xc'] == pytest.approx(20.02, rel = 1e-2)
        assert post_sum['sd']['xc'] == pytest.approx(0.0085, rel = 1e-1)
        
        assert post_sum['mean']['yc'] == pytest.approx(19.99, rel = 1e-2)
        assert post_sum['sd']['yc'] ==  pytest.approx(0.0085, rel = 1e-1)


im = np.zeros((40,40))
rng = np.random.default_rng(seed=12)
im += rng.normal(scale = 0.05, size = (40,40))
rms = np.ones(im.shape)*0.05
psf = Gaussian2DKernel(x_stddev=2.5).array
renderer = rendering.HybridRenderer((40,40), psf)
im = im + renderer.render_pointsource(10.,30., 150.)+ renderer.render_pointsource(30.,10., 150.)

cat = {}
cat['x'] = [10.,30.]
cat['y'] = [30.,10.]
cat['r']= [-1,-1]
cat['flux'] = [150.,150.]
cat['type'] = ['pointsource','pointsource']
mp = priors.PySersicMultiPrior(catalog = cat, sky_type='none')
multi_fitter = FitMulti(im,rms,psf, mp)

def test_FitMulti_map():
        map_dict = multi_fitter.find_MAP(rkey = PRNGKey(10))

        assert map_dict['source_0']['xc'] == pytest.approx(10., rel = 1e-3)        
        assert map_dict['source_0']['yc'] == pytest.approx(30., rel = 1e-3)
        assert map_dict['source_0']['flux'] == pytest.approx(150.4, rel = 1e-3)

        assert map_dict['source_1']['xc'] == pytest.approx(30., rel = 1e-3)        
        assert map_dict['source_1']['yc'] == pytest.approx(10., rel = 1e-3)
        assert map_dict['source_1']['flux'] == pytest.approx(150., rel = 1e-3)

@pytest.mark.parametrize('method',['laplace','svi-flow'])
def test_FitMulti_posterior(method):
        post_sum = multi_fitter.estimate_posterior(method, rkey = PRNGKey(3))

        assert post_sum['mean']['flux_0'] == pytest.approx(150.4, rel = 1e-2)
        assert post_sum['sd']['flux_0'] == pytest.approx(0.42, rel = 5e-2)
        assert post_sum['mean']['flux_1'] == pytest.approx(150., rel = 1e-2)
        assert post_sum['sd']['flux_1'] == pytest.approx(0.45, rel = 5e-2)

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
        assert post_sum['sd']['flux_0'] == pytest.approx(0.44, rel = 5e-2)
        assert post_sum['mean']['flux_1'] == pytest.approx(150.1, rel = 1e-2)
        assert post_sum['sd']['flux_1'] == pytest.approx(0.45, rel = 5e-2)

        assert post_sum['mean']['xc_0'] == pytest.approx(10.0, rel = 1e-2)
        assert post_sum['sd']['xc_0'] == pytest.approx(0.01, rel = 1e-1)
        assert post_sum['mean']['xc_1'] == pytest.approx(30., rel = 1e-2)
        assert post_sum['sd']['xc_1'] == pytest.approx(0.01, rel = 1e-1)

        assert post_sum['mean']['yc_0'] == pytest.approx(30.0, rel = 1e-2)
        assert post_sum['sd']['yc_0'] == pytest.approx(0.01, rel = 1e-1)
        assert post_sum['mean']['yc_1'] == pytest.approx(10., rel = 1e-2)
        assert post_sum['sd']['yc_1'] == pytest.approx(0.01, rel = 1e-1)