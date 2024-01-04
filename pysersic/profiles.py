import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod
from jax.numpy import array
from pysersic.rendering import HybridRenderer,PixelRenderer,FourierRenderer,BaseRenderer
from pysersic.profile_and_rendering_utils import render_gaussian_fourier, render_gaussian_pixel, render_pointsource_fourier, render_sersic_2d, calculate_etas_betas, sersic_gauss_decomp
from typing import Union, ClassVar
from interpax import interp1d
import warnings
from functools import partial

class BaseProfile(eqx.Module):
    param_names : eqx.AbstractClassVar[dict]
    supported_renderers : eqx.AbstractClassVar[list]
    render_intrinsic: eqx.AbstractClassVar[bool]


class PointSource(BaseProfile):
    param_names : ClassVar[list] = ['xc','yc','flux']
    supported_renderers : ClassVar[list] = [HybridRenderer,FourierRenderer,PixelRenderer]
    render_intrinsic : ClassVar[bool] = False

    def render_fourier(self,Fx: jnp.array, Fy: jnp.array, params: dict)-> jnp.array:
        F_im = render_pointsource_fourier(Fx,Fy, params['xc'], params['yc'],params['flux'])
        return F_im
    
    def render_pixel_obs(self, X: jnp.array, Y: jnp.array,PSF: jnp.array, params: dict):
        dx = params['xc'] - PSF.shape[0]/2.
        dy = params['yc'] - PSF.shape[1]/2.

        #TODO Currently only linear interpolation, could use interpax or other tool for higher order to make this more accurate
        shifted_psf = jax.scipy.ndimage.map_coordinates(PSF*params['flux'], [X-dx,Y-dy], order = 1, mode = 'constant')
    
        return shifted_psf

class BaseGaussianProfile(BaseProfile):
    """
    A Base class used to render profiles based on series of Gaussians, all of the sersic and MGE profile classes
    """
    param_names : eqx.AbstractClassVar[list]
    supported_renderers : ClassVar[list] = [HybridRenderer,FourierRenderer,PixelRenderer]
    render_intrinsic : ClassVar[bool] = True

    @abstractmethod
    def find_gaussians(self,params: dict)-> tuple[jnp.array,jnp.array,jnp.array,jnp.array,jnp.array, jnp.array]:
        """Abstract method to calculate Gaussians needed for the rest of the class

        Parameters
        ----------
        params : dict
            Input Parameters

        Returns
        -------
        tuple[jnp.array, jnp.array]
            xc,yc,theta,q,sigma and amplitude of Gaussians to render
        """
        return NotImplementedError
    
    def render_fourier(self,Fx: jnp.array, Fy: jnp.array, params: dict)-> jnp.array:
        """Render the series of Gaussians corresponding to `params` in Fourier space at Fx,Fy

        Parameters
        ----------
        Fx : jnp.array
            Frequencies corresponding to the x direction
        Fy : jnp.array
            Frequencies corresponding to the y direction
        params : dict
            Input parameters

        Returns
        -------
        jnp.array
            Fourier amplitudes of source evaluated at Fx, Fy
        """
        xc, yc, theta, q, sigmas, amps = self.find_gaussians(params)
        Fgal = render_gaussian_fourier(Fx,Fy,amps,sigmas, xc,yc,theta,q)
        return Fgal
    
    def render_pixel_intrinsic(self, X: jnp.array, Y: jnp.array, params):
        xc, yc, theta, q, sigmas, amps = self.find_gaussians(params)
        intr_im = render_gaussian_pixel(X,Y, amps,sigmas, xc,yc,theta,q)
        return intr_im

class SersicProfile(BaseGaussianProfile):
    param_names : ClassVar[list] = ['xc','yc','flux', 'n','r_eff','ellip','theta']
    
    etas: jax.numpy.array = eqx.field(static = True)
    betas: jax.numpy.array = eqx.field(static = True)
    n_ax: jax.numpy.array = eqx.field(static = True)
    amps_n_ax: jax.numpy.array = eqx.field(static = True)

    n_sigma: int = eqx.field(static=True)
    frac_end: float = eqx.field(static=True)
    frac_start: float = eqx.field(static=True)
    use_interp_amps: bool = eqx.field(static=True)

    def __init__(self, 
                 frac_start: float =1e-2, 
                 frac_end: float = 15., 
                 n_sigma: int = 15, 
                 precision: int = 10, 
                 use_interp_amps: bool = True):
        """        
        frac_start : Optional[float], optional
            Fraction of r_eff for the smallest Gaussian component, by default 0.01
        frac_end : Optional[float], optional
            Fraction of r_eff for the largest Gaussian component, by default 15.
        n_sigma : Optional[int], optional
            Number of Gaussian Components, by default 15
        precision : Optional[int], optional
            precision value used in calculating Gaussian components, see Shajib (2019) for more details, by default 10
        use_poly_fit_amps: Optional[bool]
            If True, instead of performing the direct calculation in Shajib (2019) at each iteration, a polynomial approximation is fit and used. The amplitudes of each gaussian component amplitudes as a function of Sersic index are fit with a polynomial. This smooth approximation is then used at each interval. While this adds a a little extra to the renderering error budget (roughly 1\%) but is much more numerically stable owing to the smooth gradients. If this matters for you then set this to False and make sure to enable jax's 64 bit capabilities which we find helps the stability.
        """
        self.etas,self.betas = calculate_etas_betas(precision)
        self.use_interp_amps = use_interp_amps
        if not use_interp_amps and not jax.config.x64_enabled:
            warnings.warn(" Gaussian decomposition can be numerically unstable when using jax's default 32 bit. Please either enable jax 64 bit or set 'use_interp_amps' = True for more reliable gradients and inference")
        self.n_sigma = n_sigma
        self.frac_start = frac_start
        self.frac_end = frac_end

        self.n_ax = jnp.linspace(.65,8., num = 50)
        self.amps_n_ax = jax.vmap( lambda n: sersic_gauss_decomp(1.,1.,n,self.etas,self.betas,self.frac_start,self.frac_end,self.n_sigma)[0] ) (self.n_ax)
        super().__init__()


    def find_gaussians(self, params: dict) -> tuple:
        """Derrive gaussian approximation so a sersic function using the method descrined in Shajib et al. (2019).
        This is either done directly or using pre-computed values as a function of n and calculated using interpolation.

        Parameters
        ----------
        params : dict

        Returns
        -------
        tuple
            xc,yc,theta,q,sigma and amplitude of Gaussians to render
        """
        if self.use_interp_amps:
            amps = interp1d(params['n'],self.n_ax, self.amps_n_ax, method='cubic2')*params['flux']
            sigmas = jnp.logspace(jnp.log10(self.frac_start),jnp.log10(self.frac_end),num = self.n_sigma)*params['r_eff']
        
        else:
            amps,sigmas = sersic_gauss_decomp(params['flux'], params['r_eff'], params['n'], self.etas, self.betas, self.frac_start*params['r_eff'], self.frac_end*params['r_eff'], self.n_sigma)
        ones = jnp.ones(self.n_sigma)
        return ones*params['xc'],ones*params['yc'],ones*params['theta'], 1. - ones*params['ellip'], sigmas, amps
    
    def render_pixel_intrinsic(self, X: jnp.array, Y: jnp.array, params):
        intr_im = render_sersic_2d(X,Y,xc = params['xc'], yc = params['yc'], flux = params['flux'],
                                   r_eff=params['r_eff'],n = params['n'],ellip = params['ellip'],
                                   theta = params['theta'])
        return intr_im

class SersicFixedIndexProfile(BaseGaussianProfile):
    param_names : ClassVar[list] = ['xc','yc','flux', 'r_eff','ellip','theta']
    
    etas: jax.numpy.array = eqx.field(static = True)
    betas: jax.numpy.array = eqx.field(static = True)
    pre_calc_amps: jax.numpy.array = eqx.field(static = True)

    n: float = eqx.field(static=True)
    n_sigma: int = eqx.field(static=True)
    frac_end: float = eqx.field(static=True)
    frac_start: float = eqx.field(static=True)

    def __init__(self, 
                 n: float,
                 frac_start: float =1e-2, 
                 frac_end: float = 15., 
                 n_sigma: int = 15, 
                 precision: int = 10):
        """   
        n : float
            value at which to fixed sersic index, e.g. 1 for exp, 4 for dev.
        frac_start : Optional[float], optional
            Fraction of r_eff for the smallest Gaussian component, by default 0.01
        frac_end : Optional[float], optional
            Fraction of r_eff for the largest Gaussian component, by default 15.
        n_sigma : Optional[int], optional
            Number of Gaussian Components, by default 15
        precision : Optional[int], optional
            precision value used in calculating Gaussian components, see Shajib (2019) for more details, by default 10
        """
        self.n_sigma = n_sigma
        self.frac_start = frac_start
        self.frac_end = frac_end
        self.n = n
        self.etas,self.betas = calculate_etas_betas(precision)
        self.pre_calc_amps = sersic_gauss_decomp(1.,1.,self.n,self.etas,self.betas,self.frac_start,self.frac_end,self.n_sigma)
        super().__init__()

    def find_gaussians(self, params: dict) -> tuple:
        sigmas = jnp.logspace(jnp.log10(self.frac_start),jnp.log10(self.frac_end),num = self.n_sigma)*params['r_eff']
        ones = jnp.ones(self.n_sigma)
        return ones*params['xc'],ones*params['yc'],ones*params['theta'], 1. - ones*params['ellip'], sigmas, self.pre_calc_amps*params['flux']
    
    def render_pixel_intrinsic(self, X: jnp.array, Y: jnp.array, params):
        intr_im = render_sersic_2d(X,Y,xc = params['xc'], yc = params['yc'], flux = params['flux'],
                                   r_eff=params['r_eff'],n = self.n,ellip = params['ellip'],
                                   theta = params['theta'])
        return intr_im
ExpProfile = partial(SersicFixedIndexProfile, n = 1.)
DevProfile = partial(SersicFixedIndexProfile, n = 4.)