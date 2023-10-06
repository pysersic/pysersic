from pysersic.rendering import PixelRenderer,FourierRenderer,HybridRenderer, BaseRenderer
from pysersic.rendering import render_sersic_2d
from astropy.convolution import Gaussian2DKernel
import pytest
import jax.numpy as jnp
from jax.scipy.special import gammainc
from scipy.integrate import dblquad
from functools import partial
 
kern = Gaussian2DKernel(x_stddev= 1.5)
psf = kern.array
err_tol = 0.01 # 1% error tolerence for total flux

@pytest.mark.parametrize("renderer", [PixelRenderer,FourierRenderer,HybridRenderer])
@pytest.mark.parametrize("pos", [(50,50),(50.5,50.5),(50.25,50.25) ,(50.,50.5),(50.5,50.)]) 
def test_point_source(renderer,pos):
    flux = 10.
    params = dict(flux = flux, xc = pos[0], yc = pos[1] )
    renderer_test = renderer((100,100), psf)
    im = renderer_test.render_source(params, 'pointsource')
    assert pytest.approx(im.sum(), rel = err_tol) == flux


@pytest.mark.parametrize("renderer", [PixelRenderer,FourierRenderer,HybridRenderer])
@pytest.mark.parametrize("pos", [(75.,75.),(75.5,75.5),])
@pytest.mark.parametrize("re", [3.,5.]) 
@pytest.mark.parametrize("n", [1.5,2.5]) 
@pytest.mark.parametrize("ellip", [0,0.5])
@pytest.mark.parametrize("theta", [0,3.14/4.]) 
def test_sersic(renderer,pos,re,n,ellip,theta):
    renderer_test = renderer((150,150), psf)
    flux = 10
    params = dict(flux = flux, xc = pos[0], yc = pos[1], n = n, ellip = ellip, theta = theta, r_eff = re )

    #Calculate fraction of flux contained in image
    int_test = partial(render_sersic_2d, xc = pos[0],yc = pos[1], flux = 10, r_eff = re, n = n, ellip = ellip,theta = theta)
    to_int = lambda x,y: float(int_test(x,y))
    lo_fun = lambda x: 0.
    hi_fun = lambda x: 150. 
    flux_int,_ = dblquad(to_int, 0.,150., lo_fun,hi_fun,epsrel=5.e-3)
    
    im = renderer_test.render_source(params, 'sersic')
    total = float( jnp.sum(im) )
    assert pytest.approx(flux_int, rel = err_tol) == total

@pytest.mark.parametrize("renderer", [PixelRenderer,FourierRenderer,HybridRenderer])
@pytest.mark.parametrize("prof", ['exp','dev']) 
@pytest.mark.parametrize("pos", [(75.,75.),(75.5,75.5),])
@pytest.mark.parametrize("re", [3.,5.]) 
@pytest.mark.parametrize("ellip", [0,0.5])
@pytest.mark.parametrize("theta", [0,3.14/4.]) 
def test_exp_dev(renderer,prof,pos,re,ellip,theta):
    flux = 10.
    params = dict(flux = flux, xc = pos[0], yc = pos[1], ellip = ellip, theta = theta, r_eff = re )
    renderer_test = renderer((150,150), psf)
    im = renderer_test.render_source(params, prof)
    
    if prof == 'exp':
        n=1.
    if prof == 'dev':
        n=4.

    #Calculate fraction of flux contained in image
    int_test = partial(render_sersic_2d, xc = pos[0],yc = pos[1], flux = 10, r_eff = re, n = n, ellip = ellip,theta = theta)
    to_int = lambda x,y: float(int_test(x,y))

    lo_fun = lambda x: 0.
    hi_fun = lambda x: 150. 
    flux_int,_ = dblquad(to_int, 0.,150., lo_fun,hi_fun,epsrel=5.e-3)

    total = float( jnp.sum(im) )
    assert pytest.approx(flux_int, rel = err_tol) == total

