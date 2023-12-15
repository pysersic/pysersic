import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod
from pysersic.rendering import HybridRenderer,PixelRenderer,FourierRenderer,BaseRenderer
from pysersic.profile_and_rendering_utils import render_gaussian_fourier, render_gaussian_pixel, render_pointsource_fourier, render_sersic_2d
from typing import Union, ClassVar
from interpax import interp1d

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

        #TODO Use interpax to make this more accurate
        shifted_psf = jax.scipy.ndimage.map_coordinates(PSF*params['flux'], [X-dx,Y-dy], order = 1, mode = 'constant')
    
        return shifted_psf

class BaseGaussianProfile(BaseProfile):
    """
    A Base class used to render profiles based on series of Gaussians, all of the sersic and MGE profile classes
    """
    param_names : eqx.AbstractClassVar[dict]
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
