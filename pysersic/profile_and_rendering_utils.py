
import jax
import jax.numpy as jnp
from scipy.special import comb
from typing import Union, Tuple

def render_gaussian_fourier(FX: jax.numpy.array,
        FY: jax.numpy.array,
        amps: jax.numpy.array,
        sigmas: jax.numpy.array,
        xc: float,
        yc: float, 
        theta: float,
        q: float)-> jax.numpy.array:
    """Render Gaussian components in the Fourier domain

    Parameters
    ----------
    FX : jax.numpy.array
        X frequency positions to evaluate
    FY : jax.numpy.array
        Y frequency positions to evaluate
    amps : jax.numpy.array
        Amplitudes of each component
    sigmas : jax.numpy.array
        widths of each component
    xc : float
        Central x position
    yc : float
        Central y position
    theta : float
        position angle
    q : float
        Axis ratio

    Returns
    -------
    jax.numpy.array
        Sum of components evaluated at FX and FY
    """
    theta_use = theta[:,jnp.newaxis,jnp.newaxis]
    Ui = FX[jnp.newaxis,:,:]*jnp.cos(theta_use) + FY[jnp.newaxis,:,:]*jnp.sin(theta_use) 
    Vi = -1*FX[jnp.newaxis,:,:]*jnp.sin(theta_use) + FY[jnp.newaxis,:,:]*jnp.cos(theta_use) 

    in_exp = -1*(Ui*Ui + Vi*Vi*q[:,jnp.newaxis,jnp.newaxis]*q[:,jnp.newaxis,jnp.newaxis])*(2*jnp.pi*jnp.pi*sigmas*sigmas)[:,jnp.newaxis,jnp.newaxis] - 1j*2*jnp.pi*FX*xc[:,jnp.newaxis,jnp.newaxis] - 1j*2*jnp.pi*FY*yc[:,jnp.newaxis,jnp.newaxis]
    Fgal_comp = amps[:,jnp.newaxis,jnp.newaxis]*jnp.exp(in_exp)
    Fgal = jnp.sum(Fgal_comp, axis = 0)
    return Fgal

def render_pointsource_fourier(FX: jax.numpy.array,
        FY: jax.numpy.array,
        xc: float,
        yc: float, 
        flux: float)-> jax.numpy.array:
    """Render a point source in the Fourier domain

    Parameters
    ----------
    FX : jax.numpy.array
        X frequency positions to evaluate
    FY : jax.numpy.array
        Y frequency positions to evaluate
    xc : float
        Central x position
    yc : float
        Central y position
    flux : float
        Total flux of source

    Returns
    -------
    jax.numpy.array
        Point source evaluated at FX FY
    """
    in_exp = -1j*2*jnp.pi*FX*xc - 1j*2*jnp.pi*FY*yc
    F_im = flux*jnp.exp(in_exp)
    return F_im


def render_gaussian_pixel(X: jax.numpy.array,
        Y: jax.numpy.array,
        amps: jax.numpy.array,
        sigmas: jax.numpy.array,
        xc: float,
        yc: float, 
        theta: float,
        q: Union[float,jax.numpy.array])-> jax.numpy.array:
    """Render Gaussian components in pixel space

    Parameters
    ----------
    FX : jax.numpy.array
        X positions to evaluate
    FY : jax.numpy.array
        Y positions to evaluate
    amps : jax.numpy.array
        Amplitudes of each component
    sigmas : jax.numpy.array
        widths of each component
    xc : float
        Central x position
    yc : float
        Central y position
    theta : float
        position angle
    q : Union[float,jax.numpy.array]
        Axis ratio

    Returns
    -------
    jax.numpy.array
        Sum of components evaluated at X and Y
    """
    X_bar = X - xc
    Y_bar = Y - yc

    Xi = X_bar*jnp.cos(theta) + Y_bar*jnp.sin(theta) 
    Yi = -1*X_bar*jnp.sin(theta) + Y_bar*jnp.cos(theta) 

    in_exp = -1*(Xi*Xi + Yi*Yi/(q*q)[:,jnp.newaxis,jnp.newaxis] )/ (2*sigmas*sigmas)[:,jnp.newaxis,jnp.newaxis]
    im_comp = (amps/(2*jnp.pi*sigmas*sigmas*q))[:,jnp.newaxis,jnp.newaxis]*jnp.exp(in_exp)
    im = jnp.sum(im_comp, axis = 0)
    return im


def render_sersic_2d(X: jax.numpy.array,
    Y: jax.numpy.array,
    xc: float,
    yc: float, 
    flux: float, 
    r_eff: float,
    n: float,
    ellip: float, 
    theta: float)-> jax.numpy.array:
    """Evalulate a 2D Sersic distribution at given locations

    Parameters
    ----------
    X : jax.numpy.array
        x locations to evaluate at
    Y : jax.numpy.array
        y locations to evaluate at
    xc : float
        Central x position
    yc : float
        Central y position
    flux : float
        Total flux
    r_eff : float
        Effective radius
    n : float
        Sersic index
    ellip : float
        Ellipticity
    theta : float
        Position angle in radians


    Returns
    -------
    jax.numpy.array
        Sersic model evaluated at given locations
    """
    bn = 1.9992*n - 0.3271
    a, b = r_eff, (1 - ellip) * r_eff
    cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)
    x_maj = (X - xc) * cos_theta + (Y - yc) * sin_theta
    x_min = -(X - xc) * sin_theta + (Y - yc) * cos_theta
    amplitude = flux*bn**(2*n) / ( jnp.exp(bn + jax.scipy.special.gammaln(2*n) ) *r_eff**2 *jnp.pi*2*n )
    z = jnp.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)
    out = amplitude * jnp.exp(-bn * (z ** (1 / n) - 1)) / (1.-ellip)
    return out

def calculate_etas_betas(precision: int)-> Tuple[jax.numpy.array, jax.numpy.array]:
    """Calculate the weights and nodes for the Gaussian decomposition described in Shajib (2019) (https://arxiv.org/abs/1906.08263)

    Parameters
    ----------
    precision : int
        Precision, higher number implies more precise decomposition but more nodes. Effective upper limit is 12 for 32 bit numbers, 27 for 64 bit numbers.

    Returns
    -------
    Tuple[jax.numpy.array, jax.numpy.array]
        etas and betas array to be use in gaussian decomposition
    """
    kes = jnp.arange(2 * precision + 1)
    betas = jnp.sqrt(2 * precision * jnp.log(10) / 3. + 2. * 1j * jnp.pi * kes)
    epsilons = jnp.zeros(2 * precision + 1)

    epsilons = epsilons.at[0].set(0.5)
    epsilons = epsilons.at[1:precision + 1].set(1.)
    epsilons = epsilons.at[-1].set(1 / 2. ** precision)

    for k in range(1, precision):
        epsilons = epsilons.at[2 * precision - k].set(epsilons[2 * precision - k + 1] + 1 / 2. ** precision * comb(precision, k) )

    etas = jnp.array( (-1.) ** kes * epsilons * 10. ** (precision / 3.) * 2. * jnp.sqrt(2*jnp.pi) )
    betas = jnp.array(betas)
    return etas,betas

def sersic_gauss_decomp(
        flux: float, 
        re: float, 
        n:float, 
        etas: jax.numpy.array, 
        betas:jax.numpy.array, 
        sigma_start: float, 
        sigma_end: float, 
        n_comp: int)-> Tuple[jax.numpy.array, jax.numpy.array]:
    """Calculate a gaussian decomposition of a given sersic profile, following Shajib (2019) (https://arxiv.org/abs/1906.08263)

    Parameters
    ----------
    flux : float
        Total flux
    re : float
        half light radius
    n : float
        Sersic index
    etas : jax.numpy.array
        Weights for decomposition, can be calcualted using pysersic.rendering_utils.calculate_etas_betas
    betas : jax.numpy.array
        Nodes for decomposition, can be calcualted using pysersic.rendering_utils.calculate_etas_betas
    sigma_start : float
        width for the smallest Gaussian component
    sigma_end : float
         width for the largest Gaussian component
    n_comp : int
        Number of Gaussian components

    Returns
    -------
    Tuple[jax.numpy.array, jax.numpy.array]
        Amplitudes and sigmas of Gaussian decomposition
    """
    sigmas = jnp.logspace(jnp.log10(sigma_start),jnp.log10(sigma_end),num = n_comp)

    f_sigmas = jnp.sum( etas * sersic1D(jnp.outer(sigmas,betas),flux,re,n).real,  axis=1)

    del_log_sigma = jnp.abs(jnp.diff(jnp.log(sigmas)).mean())

    amps = f_sigmas * del_log_sigma / jnp.sqrt(2*jnp.pi)

    amps = amps.at[0].multiply(0.5)
    amps = amps.at[-1].multiply(0.5)

    amps = amps*2*jnp.pi*sigmas*sigmas

    return amps,sigmas

def sersic1D(
        r: Union[float, jax.numpy.array],
        flux: float,
        re: float,
        n: float)-> Union[float, jax.numpy.array]:
    """Evaluate a 1D sersic profile

    Parameters
    ----------
    r : float
        radii to evaluate profile at
    flux : float
        Total flux
    re : float
        Effective radius
    n : float
        Sersic index

    Returns
    -------
    jax.numpy.array
        Sersic profile evaluated at r
    """
    bn = 1.9992*n - 0.3271
    Ie = flux / ( re*re* 2* jnp.pi*n * jnp.exp(bn + jax.scipy.special.gammaln(2*n) ) ) * bn**(2*n) 
    return Ie*jnp.exp ( -bn*( (r/re)**(1./n) - 1. ) )

