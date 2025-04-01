from pysersic.rendering import PixelRenderer,MoGFourierRenderer, EmulatorFourierRenderer,HybridRenderer
from pysersic.rendering import render_sersic_2d
from astropy.convolution import Gaussian2DKernel
import pytest
import jax.numpy as jnp
from scipy.integrate import dblquad
from functools import partial
import jax

render_sersic_2d = jax.jit(render_sersic_2d)
kern = Gaussian2DKernel(x_stddev= 1.5)
psf = jnp.array(kern.array)
err_tol = 0.015 # 1.5% error tolerence for total flux

pixel_renderer = PixelRenderer((150,150), psf)
mog_fourier_renderer = MoGFourierRenderer((150,150), psf)
emu_fourier_renderer = EmulatorFourierRenderer((150,150), psf)
hybrid_renderer = HybridRenderer((150,150), psf)

@partial(jax.jit, static_argnums = [1,])
def get_models(params, profile_type: str):
    im_px = pixel_renderer.render_source(params, profile_type=profile_type)
    im_mf = mog_fourier_renderer.render_source(params, profile_type=profile_type)
    im_ef = emu_fourier_renderer.render_source(params, profile_type = profile_type)
    im_hy = hybrid_renderer.render_source(params, profile_type=profile_type)
    return im_px, im_mf,im_ef, im_hy

@pytest.mark.parametrize("pos", [(75.,75.),(75.5,75.5),(75.25,75.25) ,(75.,75.5),(75.5,75.)]) 
def test_point_source(pos):
    flux = 10.
    params = dict(flux = flux, xc = pos[0], yc = pos[1] )
    ims = get_models(params, 'pointsource')
    assert pytest.approx(ims[0].sum(), rel = err_tol) == flux #pixel
    assert pytest.approx(ims[1].sum(), rel = err_tol) == flux #fourier
    assert pytest.approx(ims[2].sum(), rel = err_tol) == flux #hybrid

@pytest.mark.parametrize("pos", [(75.,75.),(75.5,75.5),])
@pytest.mark.parametrize("re", [3.,5.]) 
@pytest.mark.parametrize("n", [1.5,2.5]) 
@pytest.mark.parametrize("ellip", [0,0.5])
@pytest.mark.parametrize("theta", [0,3.14/4.]) 
def test_sersic(pos,re,n,ellip,theta):
    flux = 10
    params = dict(flux = flux, xc = pos[0], yc = pos[1], n = n, ellip = ellip, theta = theta, r_eff = re )
    #Calculate fraction of flux contained in image
    int_test = partial(render_sersic_2d, xc = pos[0],yc = pos[1], flux = 10, r_eff = re, n = n, ellip = ellip,theta = theta)
    to_int = lambda x,y: float(int_test(x,y))
    lo_fun = lambda x: 0.
    hi_fun = lambda x: 150. 
    flux_int,_ = dblquad(to_int, 0.,150., lo_fun,hi_fun,epsrel=5.e-3)
    
    ims = get_models(params, 'sersic')
    assert pytest.approx(float(ims[0].sum()), rel = err_tol) == flux_int #pixel
    assert pytest.approx(float(ims[1].sum()), rel = err_tol) == flux_int #fourier MoG
    assert pytest.approx(float(ims[2].sum()), rel = err_tol) == flux_int #fourier Emu
    assert pytest.approx(float(ims[3].sum()), rel = err_tol) == flux_int #hybrid

@pytest.mark.parametrize("prof", ['exp','dev'])
@pytest.mark.parametrize("pos", [(75.,75.),(75.5,75.5),])
@pytest.mark.parametrize("re", [3.,5.]) 
@pytest.mark.parametrize("ellip", [0,0.5])
@pytest.mark.parametrize("theta", [0,3.14/4.]) 
def test_exp_dev(prof,pos,re,ellip,theta):
    flux = 10.
    params = dict(flux = flux, xc = pos[0], yc = pos[1], ellip = ellip, theta = theta, r_eff = re )
    
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

    ims = get_models(params, prof)
    assert pytest.approx(float(ims[0].sum()), rel = err_tol) == flux_int #pixel
    assert pytest.approx(float(ims[1].sum()), rel = err_tol) == flux_int #fourier MoG
    assert pytest.approx(float(ims[2].sum()), rel = err_tol) == flux_int #fourier Emu
    assert pytest.approx(float(ims[3].sum()), rel = err_tol) == flux_int #hybrid


@pytest.mark.parametrize("pos", [(75.,75.),(75.5,75.5),])
@pytest.mark.parametrize("re", [3.,5.]) 
@pytest.mark.parametrize("nu_star", [1.,5.]) 
@pytest.mark.parametrize("ellip", [0,0.5])
@pytest.mark.parametrize("theta", [0,3.14/4.]) 
def test_spergel(pos,re,nu_star,ellip,theta):
    flux = 10
    params = dict(flux = flux, xc = pos[0], yc = pos[1], nu_star = nu_star, ellip = ellip, theta = theta, r_eff = re )
    #Calculate fraction of flux contained in image

    
    ims = get_models(params, 'spergel')
    assert pytest.approx(float(ims[0].sum()), rel = err_tol) == flux #pixel
    assert pytest.approx(float(ims[1].sum()), rel = err_tol) == flux #fourier MoG
    assert pytest.approx(float(ims[2].sum()), rel = err_tol) == flux #fourier Emu
    assert pytest.approx(float(ims[3].sum()), rel = err_tol) == flux #hybrid
