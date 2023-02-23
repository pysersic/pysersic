from pysersic.rendering import PixelRenderer,FourierRenderer,HybridRenderer, BaseRenderer
from pysersic.utils.rendering_utils import render_sersic_2d
from astropy.convolution import Gaussian2DKernel
import pytest
import jax.numpy as jnp
from jax.scipy.special import gammainc
from scipy.integrate import dblquad
from functools import partial

kern = Gaussian2DKernel(x_stddev= 2.5)
psf = kern.array
err_tol = 0.025 # 2.5% error tolerence

@pytest.mark.parametrize("renderer", [PixelRenderer,FourierRenderer,HybridRenderer])
@pytest.mark.parametrize("pos", [(50,50),(50.5,50.5),(50.25,50.25) ,(50.,50.5),(50.5,50.)]) 
def test_point_source(renderer,pos):
    flux = 10.
    renderer_test = renderer((100,100), psf)
    im = renderer_test.render_pointsource(pos[0],pos[1],flux)
    assert pytest.approx(im.sum(), rel = err_tol) == flux


@pytest.mark.parametrize("renderer", [PixelRenderer,FourierRenderer,HybridRenderer])
@pytest.mark.parametrize("pos", [(100.,100.),(100.5,100.5),])
@pytest.mark.parametrize("re", [5.,10.]) 
@pytest.mark.parametrize("n", [1.5,2.5]) 
@pytest.mark.parametrize("ellip", [0,0.5])
@pytest.mark.parametrize("theta", [0,3.14/4.]) 
def test_sersic(renderer,pos,re,n,ellip,theta):
    renderer_test = renderer((200,200), psf)
    flux = 10
    #Calcualte fraction of flux contained in image
    int_test = partial(render_sersic_2d, xc = pos[0],yc = pos[1], flux = 10, r_eff = re, n = n, ellip = ellip,theta = theta)
    to_int = lambda x,y: float(int_test(x,y))

    lo_fun = lambda x: 0.
    hi_fun = lambda x: 200. 
    flux_int,_ = dblquad(to_int, 0.,200., lo_fun,hi_fun,epsrel=5.e-3)

    im = renderer_test.render_sersic(pos[0],pos[1],flux,re,n,ellip,theta)
    total = float( jnp.sum(im) )
    assert pytest.approx(flux_int, rel = err_tol) == total

@pytest.mark.parametrize("renderer", [PixelRenderer,FourierRenderer,HybridRenderer])
@pytest.mark.parametrize("prof", ['exp','dev']) 
@pytest.mark.parametrize("pos", [(100.,100.),(100.5,100.5),]) #test half pixel interpolation
@pytest.mark.parametrize("re", [5.,10.]) 
@pytest.mark.parametrize("ellip", [0,0.5])
@pytest.mark.parametrize("theta", [0,3.14/4.]) 
def test_exp_dev(renderer,prof,pos,re,ellip,theta):
    flux = 10.
    renderer_test = renderer((200,200), psf)
    if prof == 'exp':
        im = renderer_test.render_exp(pos[0],pos[1],flux,re,ellip,theta)
        n=1.
    if prof == 'dev':
        im = renderer_test.render_dev(pos[0],pos[1],flux,re,ellip,theta)
        n=4.

    #Calcualte fraction of flux contained in image
    int_test = partial(render_sersic_2d, xc = pos[0],yc = pos[1], flux = 10, r_eff = re, n = n, ellip = ellip,theta = theta)
    to_int = lambda x,y: float(int_test(x,y))

    lo_fun = lambda x: 0.
    hi_fun = lambda x: 200. 
    flux_int,_ = dblquad(to_int, 0.,200., lo_fun,hi_fun,epsrel=5.e-3)
    total = float( jnp.sum(im) )
    assert pytest.approx(flux_int, rel = err_tol) == total


def test_sky_rendering():
    renderer_test = BaseRenderer((100,100), psf)

    assert renderer_test.render_sky(0, None) == 0

    assert renderer_test.render_sky(1., 'flat') == 1

    sky_1 = renderer_test.render_sky([1e-4,0.,0.], 'tilted-plane')
    assert sky_1.shape == (100,100)
    sky_sum_1 = float(jnp.sum(sky_1))
    assert pytest.approx(1e-4*100*100, rel = 1e-6) == sky_sum_1

    sky_2 = renderer_test.render_sky([0.,1e-3,0.], 'tilted-plane')
    assert sky_2.shape == (100,100)
    sky_sum_2 = float(jnp.sum(sky_2))
    assert pytest.approx(0, abs = 1e-4) == sky_sum_2

    sky_3 = renderer_test.render_sky([0.,0.,1e-3,], 'tilted-plane')
    assert sky_3.shape == (100,100)
    sky_sum_3 = float(jnp.sum(sky_3))
    assert pytest.approx(0, abs = 1e-4) == sky_sum_3