from abc import ABC, abstractmethod
from typing import Iterable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from scipy.special import comb
from functools import partial
from .exceptions import * 

base_profile_types = ['sersic','doublesersic','pointsource','exp','dev']
base_profile_params =dict(
    zip(base_profile_types,
    [ ['xc','yc','flux','r_eff','n','ellip','theta'],
    ['xc','yc','flux','f_1', 'r_eff_1','n_1','ellip_1', 'r_eff_2','n_2','ellip_2','theta'],
    ['xc','yc','flux'],
    ['xc','yc','flux','r_eff','ellip','theta'],
    ['xc','yc','flux','r_eff','ellip','theta'],]
    )
)

class BaseRenderer(object):
    def __init__(self, 
            im_shape: Iterable, 
            pixel_PSF: jax.numpy.array
            )-> None:
        """Base class for different Renderers

        Parameters
        ----------
        im_shape : Iterable
            Tuple or list containing the shape of the desired output
        pixel_PSF : jax.numpy.array
            Pixelized version of the PSF
        """
        self.im_shape = im_shape
        self.pixel_PSF = pixel_PSF
        if not jnp.isclose(jnp.sum(self.pixel_PSF),1.0,0.1):
            raise PSFNormalizationWarning('PSF does not appear to be appropriately normalized; Sum(psf) is more than 0.1 away from 1.')
        self.psf_shape = jnp.shape(self.pixel_PSF)
        if jnp.all(self.im_shape<self.psf_shape):
            raise KernelError('PSF pixel image size must be smaller than science image.')
        self.x = jnp.arange(self.im_shape[0])
        self.y = jnp.arange(self.im_shape[1])
        self.X,self.Y = jnp.meshgrid(self.x,self.y)
        self.x_mid = self.im_shape[0]/2. - 0.5
        self.y_mid = self.im_shape[1]/2. - 0.5

        # Set up pre-FFTed PSF
        f1d1 = jnp.fft.rfftfreq(self.im_shape[0])
        f1d2 = jnp.fft.fftfreq(self.im_shape[1])
        self.FX,self.FY = jnp.meshgrid(f1d1,f1d2)
        self.fft_shape = self.FX.shape
        fft_shift_arr_x = jnp.exp(jax.lax.complex(0.,-1.)*2.*3.1415*-1*(self.psf_shape[0]/2.-0.5)*self.FX)
        fft_shift_arr_y = jnp.exp(jax.lax.complex(0.,-1.)*2.*3.1415*-1*(self.psf_shape[1]/2.-0.5)*self.FY)
        self.PSF_fft = jnp.fft.rfft2(self.pixel_PSF, s = self.im_shape)*fft_shift_arr_x*fft_shift_arr_y

        #All the  renderers here these profile types
        self.profile_types = base_profile_types
        self.profile_params = base_profile_params
        self.profile_func_dict = {}
        for profile_type in self.profile_types:
            self.profile_func_dict[profile_type] = getattr(self, f'render_{profile_type}') 

        self.fft_zeros = jnp.zeros(self.fft_shape)
        self.img_zeros = jnp.zeros(self.im_shape)

        def conv_img(image):
            img_fft = jnp.fft.rfft2(image)
            conv_fft = img_fft*self.PSF_fft
            conv_im = jnp.fft.irfft2(conv_fft, s= self.im_shape)
            return conv_im
        #self.conv_img = jit(conv_img)

        def conv_fft(F_im):
            im = jnp.fft.irfft2(F_im*self.PSF_fft, s= self.im_shape) 
            return im
        #self.conv_fft = jit(conv_fft)

        def combine_scene(F_im, int_im, obs_im):
            return conv_fft(F_im)+  conv_img(int_im) + obs_im
        self.combine_scene = jit(combine_scene)
    
    @abstractmethod
    def render_sersic(self,params: dict):
        return NotImplementedError

    def render_doublesersic(self,params:dict):
        dict_1 = {'xc':params['xc'], 'yc':params['yc'], 'flux': params['flux']*params['f_1'], 'n':params['n_1'],
                    'ellip': params['ellip_1'], 'theta': params['theta'], 'r_eff': params['r_eff_1']}
        dict_2 = {'xc':params['xc'], 'yc':params['yc'], 'flux': params['flux']*(1.-params['f_1']), 'n':params['n_2'],
                     'ellip': params['ellip_2'], 'theta': params['theta'], 'r_eff': params['r_eff_2']}
        F1, im_int_1, im_obs_1 =  self.render_sersic(dict_1)
        F2, im_int_2, im_obs_2 =  self.render_sersic(dict_2)

        return F1+F2, im_int_1+im_int_2, im_obs_1+im_obs_2

    @abstractmethod
    def render_pointsource(self,params: dict):
        return NotImplementedError

    def render_exp(self, params: dict)-> jax.numpy.array:
        """Thin wrapper for an exponential profile based on render_sersic

        Parameters
        ----------
        xc : float
            Central x position
        yc : float
            Central y position
        flux : float
            Total flux
        r_eff : float
            Effective radius
        ellip : float
            Ellipticity
        theta : float
            Position angle in radians

        Returns
        -------
        jax.numpy.array
            Rendered Exponential model
        """
        to_sersic = dict(params, n = 1.)
        return self.render_sersic(to_sersic)
    
    def render_dev(self, params: dict)-> jax.numpy.array:
        """Thin wrapper for a De Vaucouleurs profile based on render_sersic

        Parameters
        ----------
        xc : float
            Central x position
        yc : float
            Central y position
        flux : float
            Total flux
        r_eff : float
            Effective radius
        ellip : float
            Ellipticity
        theta : float
            Position angle in radians

        Returns
        -------
        jax.numpy.array
            Rendered Exponential model
        """
        to_sersic = dict(params, n = 4.)
        return self.render_sersic(to_sersic)
    
    def render_for_model(self,param_dict, types, suffix):

        F_tot = jnp.zeros(self.fft_shape)
        int_im_tot = jnp.zeros(self.im_shape)
        obs_im_tot = jnp.zeros(self.im_shape)

        for j,prof_type in enumerate(types):
            new_dict = {param:param_dict[param+f'_{j:d}{suffix}'] for param in base_profile_params[prof_type]}
            F_cur, int_im_cur, obs_im_cur = self.profile_func_dict[prof_type](new_dict) 
            F_tot = F_tot + F_cur
            int_im_tot = int_im_tot + int_im_cur
            obs_im_tot = obs_im_tot + obs_im_cur
        
        return self.combine_scene(F_tot,int_im_tot, obs_im_tot)
    
    def render_source(self,
            params: dict,
            profile_type: str,
            suffix: Optional[str] = '')->jax.numpy.array :
        """Render an observed source of a given type and parameters

        Parameters
        ----------
        params : jax.numpy.array
            Parameters specifying the source
        profile_type : str
            Type of profile to use

        Returns
        -------
        jax.numpy.array
            Rendered, observed model
        """
        to_func = {k.replace(suffix,""):v for k,v in params.items()}
        model_im = self.combine_scene( *self.profile_func_dict[profile_type](to_func) )
        return model_im


class PixelRenderer(BaseRenderer):
    """
    Render class based on rendering in pixel space and then convolving with the PSF
    """
    def __init__(self, 
            im_shape: Iterable, 
            pixel_PSF: jax.numpy.array,
            os_pixel_size: Optional[int]= 6, 
            num_os: Optional[int] = 12) -> None:
        """Initialize the PixelRenderer class

        Parameters
        ----------
        im_shape : Iterable
            Tuple or list containing the shape of the desired output
        pixel_PSF : jax.numpy.array
            Pixelized version of the PSF
        os_pixel_size : Optional[int], optional
            Size of box around the center of the image to perform pixel oversampling
        num_os : Optional[int], optional
            Number of points to oversample by in each direction, by default 8
        """
        super().__init__(im_shape, pixel_PSF)
        self.os_pixel_size = os_pixel_size
        self.num_os = num_os
        
        #Use Gauss-Legendre coefficents for better integration when oversampling
        dx,w = np.polynomial.legendre.leggauss(self.num_os)
        w = w/2. 
        dx = dx/2.
        
        #dx = np.linspace(-0.5,0.5, num= num_os,endpoint=True)
        #w = np.ones_like(dx)
        self.dx_os,self.dy_os = jnp.meshgrid(dx,dx)

        w1,w2 = jnp.meshgrid(w,w)
        self.w_os = w1*w2
        
        i_mid = int(self.im_shape[0]/2)
        j_mid = int(self.im_shape[1]/2)

        self.x_os_lo, self.x_os_hi = i_mid - self.os_pixel_size, i_mid + self.os_pixel_size
        self.y_os_lo, self.y_os_hi = j_mid - self.os_pixel_size, j_mid + self.os_pixel_size

        self.X_os = self.X[self.x_os_lo:self.x_os_hi ,self.y_os_lo:self.y_os_hi ,jnp.newaxis,jnp.newaxis] + self.dx_os
        self.Y_os = self.Y[self.x_os_lo:self.x_os_hi ,self.y_os_lo:self.y_os_hi ,jnp.newaxis,jnp.newaxis] + self.dy_os


        #Set up and jit intrinsic Sersic rendering with Oversampling
        def render_int_sersic(xc,yc, flux, r_eff, n,ellip, theta):
            im_no_os = render_sersic_2d(self.X,self.Y,xc,yc, flux, r_eff, n,ellip, theta)
            
            sub_im_os = render_sersic_2d(self.X_os,self.Y_os,xc,yc, flux, r_eff, n,ellip, theta)
            sub_im_os = jnp.sum(sub_im_os*self.w_os, axis = (2,3)) 
            
            im = im_no_os.at[self.x_os_lo:self.x_os_hi ,self.y_os_lo:self.y_os_hi].set(sub_im_os)
            return im
        self.render_int_sersic = jit(render_int_sersic)

    def render_sersic(self,params:dict)->jax.numpy.array:
        """Render a sersic profile

        Parameters
        ----------
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
            Rendered Sersic model
        """
        im_int = self.render_int_sersic(params['xc'],params['yc'], params['flux'],params['r_eff'], params['n'],params['ellip'],params['theta'])
        return self.fft_zeros, im_int, self.img_zeros
    

    def render_pointsource(self, params:dict)-> jax.numpy.array:
        """Render a Point source by interpolating given PSF into image. Currently jax only supports linear intepolation.

        Parameters
        ----------
        xc : float
            Central x position
        yc : float
            Central y position
        flux : float
            Total flux

        Returns
        -------
        jax.numpy.array
            rendered pointsource model
        """
        dx = params['xc'] - self.psf_shape[0]/2.
        dy = params['yc'] - self.psf_shape[1]/2.

        shifted_psf = jax.scipy.ndimage.map_coordinates(self.pixel_PSF*params['flux'], [self.X-dx,self.Y-dy], order = 1, mode = 'constant')
    
        return self.fft_zeros,self.img_zeros, shifted_psf
    

class FourierRenderer(BaseRenderer):
    """
    Class to render sources based on rendering them in Fourier space. Sersic profiles are modeled as a series of Gaussian following Shajib (2019) (https://arxiv.org/abs/1906.08263) and the implementation in lenstronomy (https://github.com/lenstronomy/lenstronomy/blob/main/lenstronomy/LensModel/Profiles/gauss_decomposition.py)
    """
    def __init__(self, 
            im_shape: Iterable, 
            pixel_PSF: jax.numpy.array,
            frac_start: Optional[float] = 1e-2,
            frac_end: Optional[float] = 15., 
            n_sigma: Optional[int] = 15, 
            precision: Optional[int] = 10,
            use_poly_fit_amps: Optional[bool] = True)-> None:
        """Initialize a Fourier renderer class

        Parameters
        ----------
        im_shape : Iterable
            Tuple or list containing the shape of the desired output
        pixel_PSF : jax.numpy.array
            Pixelized version of the PSF
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
        super().__init__(im_shape, pixel_PSF)
        self.frac_start = frac_start
        self.frac_end = frac_end
        self.n_sigma = n_sigma
        self.precision = precision

        self.etas,self.betas = calculate_etas_betas(self.precision)

        if use_poly_fit_amps:

            #Set up grid of amplitudes at different values of n
            log_n_ax = jnp.logspace(jnp.log10(.65),jnp.log10(8), num = 100)
            amps_log_n = jax.vmap( lambda n: sersic_gauss_decomp(1.,1.,n,self.etas,self.betas,self.frac_start,self.frac_end,self.n_sigma)[0] ) (log_n_ax)

            #Fit polynomial for smooth interpolation
            amps_log_n_pfits = jnp.polyfit(np.log10(log_n_ax),amps_log_n,10.)

            def get_amps_sigmas(flux,r_eff,n):
                amps_norm = jnp.polyval(amps_log_n_pfits, jnp.log10(n))
                amps = amps_norm*flux
                sigmas = jnp.logspace(jnp.log10(r_eff*self.frac_start),jnp.log10(r_eff*self.frac_end),num = self.n_sigma)
                return amps,sigmas
            self.get_amps_sigmas = jax.jit(get_amps_sigmas)
        else:
            if not jax.config.jax_enable_x64:
                print ("!! WARNING !! - FourierRenderer can be numerically unstable when using jax's default 32 bit. Please either enable jax 64 bit or set 'use_poly_amps' = True in the renderer kwargs")
            def get_amps_sigmas(flux,r_eff,n):
                return sersic_gauss_decomp(flux, r_eff, n, self.etas, self.betas, self.frac_start*r_eff, self.frac_end*r_eff, self.n_sigma)
            self.get_amps_sigmas = jax.jit(get_amps_sigmas)
            

        #Jit compile function to render sersic profile in Fourier space
        def render_sersic_mog_fourier(xc,yc, flux, r_eff, n,ellip, theta):
            amps,sigmas = self.get_amps_sigmas(flux, r_eff, n)
            q = 1.-ellip
            Fgal = render_gaussian_fourier(self.FX,self.FY, amps,sigmas,xc,yc, theta,q)
            return Fgal
        self.render_sersic_mog_fourier = jit(render_sersic_mog_fourier)

    def render_sersic(self,params: dict)->jax.numpy.array:
        """ Render a Sersic profile

        Parameters
        ----------
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
            Rendered Sersic model
        """
        F_im = self.render_sersic_mog_fourier(params['xc'],params['yc'], params['flux'],params['r_eff'], params['n'],params['ellip'],params['theta'])
        return F_im, self.img_zeros, self.img_zeros

    
    def render_pointsource(self, params:dict)-> jax.numpy.array:
        """Render a Point source

        Parameters
        ----------
        xc : float
            Central x position
        yc : float
            Central y position
        flux : float
            Total flux

        Returns
        -------
        jax.numpy.array
            rendered pointsource model
        """
        F_im = render_pointsource_fourier(self.FX,self.FY,params['xc'],params['yc'],params['flux'])
        return F_im, self.img_zeros, self.img_zeros

class HybridRenderer(BaseRenderer):
    """
    Class to render sources based on the hybrid rendering scheme introduced in Lang (2020). This avoids some of the artifacts introduced by rendering sources purely in Fourier space. Sersic profiles are modeled as a series of Gaussian following Shajib (2019) (https://arxiv.org/abs/1906.08263) and the implementation in lenstronomy (https://github.com/lenstronomy/lenstronomy/blob/main/lenstronomy/LensModel/Profiles/gauss_decomposition.py).

    Our scheme is implemented slightly differently than Lang (2020), specifically in how it chooses which gaussian components to render in Fourier vs. Real space. Lang (2020) employs a cutoff based on distance to the edge of the image. However given some of jax's limitation with dynamic shapes (see more here -> https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#dynamic-shapes), we have not implemented that specific criterion. Instead we use a simpler version where the user must decide how many components to render in real space, starting from the largest ones. While this is not ideal in all circumstances it still overcomes many of the issues of rendering purely in fourier space discussed in Lang (2020).
    """
    def __init__(self, 
            im_shape: Iterable, 
            pixel_PSF: jax.numpy.array,
            frac_start: Optional[float] = 1e-2,
            frac_end: Optional[float] = 15., 
            n_sigma: Optional[int] = 15, 
            num_pixel_render: Optional[int] = 3,
            precision: Optional[int] = 10,
            use_poly_fit_amps: Optional[bool] = True)-> None:
        """Initialize a  HybridRenderer class

        Parameters
        ----------
        im_shape : Iterable
            Tuple or list containing the shape of the desired output
        pixel_PSF : jax.numpy.array
            Pixelized version of the PSF
        frac_start : Optional[float], optional
            Fraction of r_eff for the smallest Gaussian component, by default 0.02
        frac_end : Optional[float], optional
            Fraction of r_eff for the largest Gaussian component, by default 15.
        n_sigma : Optional[int], optional
            Number of Gaussian Components, by default 13
        num_pixel_Render : Optional[int], optional
            Number of components to render in pixel space, counts back from largest component
        precision : Optional[int], optional
            precision value used in calculating Gaussian components, see Shajib (2019) for more details, by default 10
        use_poly_fit_amps: Optional[bool]
            If True, instead of performing the direct calculation in Shajib (2019) at each iteration, a polynomial approximation is fit and used. The amplitudes of each gaussian component amplitudes as a function of Sersic index are fit with a polynomial. This smooth approximation is then used at each interval. While this adds a a little extra to the renderering error budget (roughly 1\%) but is much more numerically stable owing to the smooth gradients. If this matters for you then set this to False and make sure to enable jax's 64 bit capabilities which we find helps the stability.
        """
        super().__init__(im_shape, pixel_PSF)
        self.frac_start = frac_start
        self.frac_end = frac_end
        self.n_sigma = n_sigma
        self.precision = precision
        self.num_pixel_render = num_pixel_render
        self.w_real = jnp.arange(self.n_sigma - self.num_pixel_render, self.n_sigma, dtype=jnp.int32)
        self.w_fourier = jnp.arange(self.n_sigma - self.num_pixel_render, dtype=jnp.int32)

        psf_X,psf_Y = jnp.meshgrid(jnp.arange(self.psf_shape[0]),jnp.arange(self.psf_shape[1]))
        sig_x = jnp.sqrt( (self.pixel_PSF*(psf_X - psf_X.mean())**2).sum()/self.pixel_PSF.sum() )
        sig_y = jnp.sqrt( (self.pixel_PSF*(psf_Y - psf_Y.mean())**2).sum()/self.pixel_PSF.sum() )
        self.sig_psf_approx = 0.5*(sig_x + sig_y)

        self.etas,self.betas = calculate_etas_betas(self.precision)

        if use_poly_fit_amps:

            #Set up grid of amplitudes at different values of n
            log_n_ax = jnp.logspace(jnp.log10(.65),jnp.log10(8), num = 100)
            amps_log_n = jax.vmap( lambda n: sersic_gauss_decomp(1.,1.,n,self.etas,self.betas,self.frac_start,self.frac_end,self.n_sigma)[0] ) (log_n_ax)

            #Fit polynomial for smooth interpolation
            amps_log_n_pfits = jnp.polyfit(np.log10(log_n_ax),amps_log_n,10.)

            def get_amps_sigmas(flux,r_eff,n):
                amps_norm = jnp.polyval(amps_log_n_pfits, jnp.log10(n))
                amps = amps_norm*flux
                sigmas = jnp.logspace(jnp.log10(r_eff*self.frac_start),jnp.log10(r_eff*self.frac_end),num = self.n_sigma)
                return amps,sigmas
            self.get_amps_sigmas = jax.jit(get_amps_sigmas)
        else:
            if not jax.config.jax_enable_x64:
                print ("!! WARNING !! - HybridRenderer can be numerically unstable when using jax's default 32 bit. Please either enable jax 64 bit or set 'use_poly_amps' = True in the renderer kwargs")
            def get_amps_sigmas(flux,r_eff,n):
                return sersic_gauss_decomp(flux, r_eff, n, self.etas, self.betas, self.frac_start*r_eff, self.frac_end*r_eff, self.n_sigma)
            self.get_amps_sigmas = jax.jit(get_amps_sigmas)
            

        def render_sersic_hybrid(xc,yc, flux, r_eff, n,ellip, theta):
            amps,sigmas = self.get_amps_sigmas(flux, r_eff, n)
            
            q = 1.-ellip

            sigmas_obs = jnp.sqrt(sigmas**2 + self.sig_psf_approx**2)
            q_obs = jnp.sqrt( (q*q*sigmas**2 + self.sig_psf_approx**2)/ sigmas_obs**2 )

            Fgal = render_gaussian_fourier(self.FX,self.FY, amps[self.w_fourier],sigmas[self.w_fourier],xc,yc, theta,q)

            im_gal = render_gaussian_pixel(self.X,self.Y, amps[self.w_real],sigmas_obs[self.w_real],xc,yc, theta,q_obs[self.w_real])

            return Fgal,im_gal
        self.render_sersic_hyrbid = jax.jit(render_sersic_hybrid)

    def render_sersic(self,params:dict)->jax.numpy.array:
        """ Render a Sersic profile

        Parameters
        ----------
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
            Rendered Sersic model
        """
        F, im  = self.render_sersic_hyrbid(params['xc'],params['yc'], params['flux'],params['r_eff'], params['n'],params['ellip'],params['theta'])
        return F, self.img_zeros, im


    def render_pointsource(self, params: dict)-> jax.numpy.array:
        """Render a Point source

        Parameters
        ----------
        xc : float
            Central x position
        yc : float
            Central y position
        flux : float
            Total flux

        Returns
        -------
        jax.numpy.array
            rendered pointsource model
        """
        F_im = render_pointsource_fourier(self.FX,self.FY,params['xc'],params['yc'],params['flux'])
        return F_im, self.img_zeros, self.img_zeros
