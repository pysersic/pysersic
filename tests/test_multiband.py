import pytest
import numpy as np
import copy
from pysersic import FitSingle,FitMulti, priors, rendering
from astropy.convolution import Gaussian2DKernel
from jax.random import PRNGKey
import arviz
from numpyro import distributions as dist,infer,sample, optim
from pysersic.pysersic import train_numpyro_svi_early_stop
from pysersic.priors import estimate_sky
from pysersic.multiband import FitMultiBandBSpline, FitMultiBandPoly



rng = np.random.default_rng(seed=10)
psf = Gaussian2DKernel(x_stddev=2.5).array
renderer = rendering.HybridRenderer((40,40), psf)

band_list = ['a','b','c']
wv_list = np.array([0.5,1,1.5])
linked_params = ['flux',]
const_params = ['xc',]
wv_to_save = np.linspace(0.5,5, num = 10)

im_list = []
rms_list = []
for wv in wv_list:
    rms = np.ones((40,40))*0.05*wv
    im_s = np.zeros((40,40))
    im_s += rng.normal(scale = 0.05*wv, size = (40,40))
    im_s = im_s + renderer.render_source( dict(xc = 20.,yc=20., flux = 200.*wv), 'pointsource' )
    im_list.append(im_s)
    rms_list.append(rms)

prior = priors.PySersicSourcePrior('pointsource')
prior.set_gaussian_prior('xc', 20.,1.)
prior.set_gaussian_prior('yc', 20.,1.)
prior.set_uniform_prior('flux', 25.,400.)

fitter_list = []

for j in range(3):
    fitter_list.append(FitSingle(
        data=im_list[j], 
        rms = rms_list[j],
        psf = psf,
        prior = copy.deepcopy(prior), 
    ) )

@pytest.mark.parametrize('fitter', [FitMultiBandPoly, FitMultiBandBSpline] )
def test_multiband(fitter):
    mb_fitter = fitter(fitter_list=fitter_list,
                            wavelengths=wv_list,
                            band_names= band_list,
                            linked_params=linked_params,
                            const_params=const_params,
                            wv_to_save= wv_to_save)

    #Make sure everything was parsed correctly
    assert 'flux' in mb_fitter.linked_params
    assert 'yc' in  mb_fitter.unlinked_params
    assert 'xc' in  mb_fitter.const_params

    res = mb_fitter.estimate_posterior(rkey = PRNGKey(11))
    summ_dict = res.summary().to_dict(orient = 'index')

    #Make sure all of the variables that should exist do
    assert 'flux_at_wv[9]' in summ_dict
    assert 'flux_c' in summ_dict

    assert 'yc_c' in summ_dict
    assert 'xc' in summ_dict

    #Make sure some of the values are right to test if inference was successful
    assert summ_dict['flux_b']['mean'] == pytest.approx(200, rel = 5e-2)
    assert summ_dict['xc']['mean'] == pytest.approx(20, rel = 5e-2) 
    assert summ_dict['yc_b']['mean'] == pytest.approx(20, rel = 5e-2) 