import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
from scipy.special import comb

from abc import abstractmethod
from typing import Union, Optional, Iterable

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
        self.psf_shape = jnp.shape(self.pixel_PSF)

        self.x = jnp.arange(self.im_shape[0])
        self.y = jnp.arange(self.im_shape[1])
        self.X,self.Y = jnp.meshgrid(self.x,self.y)
        self.x_mid = self.im_shape[0]/2. - 0.5
        self.y_mid = self.im_shape[1]/2. - 0.5

        # Set up pre-FFTed PSF
        f1d1 = jnp.fft.rfftfreq(self.im_shape[0])
        f1d2 = jnp.fft.fftfreq(self.im_shape[1])
        self.FX,self.FY = jnp.meshgrid(f1d1,f1d2)
        fft_shift_arr_x = jnp.exp(jax.lax.complex(0.,-1.)*2.*3.1415*-1*(self.psf_shape[0]/2.-0.5)*self.FX)
        fft_shift_arr_y = jnp.exp(jax.lax.complex(0.,-1.)*2.*3.1415*-1*(self.psf_shape[1]/2.-0.5)*self.FY)
        self.PSF_fft = jnp.fft.rfft2(self.pixel_PSF, s = self.im_shape)*fft_shift_arr_x*fft_shift_arr_y

        #All the  renderers here these profile types
        self.profile_types = base_profile_types
        self.profile_params = base_profile_params

    def flat_sky(self,x:float)-> float:
        """A constant sky background

        Parameters
        ----------
        x : float
            Background level
        Returns
        -------
        float
            Background level
        """
        return x

    def tilted_plane_sky(self, x:jax.numpy.array)->jax.numpy.array:
        """Render a tilted plane sky background

        Parameters
        ----------
        x : jax.numpy.array
            An array containing three parameters specifying the background

        Returns
        -------
        jax.numpy.array
            The model of the sky
        """
        return x[0] + (self.X -  self.x_mid)*x[1] + (self.Y - self.y_mid)*x[2]
    
    def render_sky(self,
            x: Union[None,float, jax.numpy.array],
            sky_type: str
            ) -> Union[float, jax.numpy.array]:
        """Render a sky background

        Parameters
        ----------
        x : Union[None,float, jax.numpy.array]
            Parameters of the given sky
        sky_type : str
            Type of sky model

        Returns
        -------
        Union[float, jax.numpy.array]
            Rendered sky background
        """
        if sky_type is None:
            return 0.
        elif sky_type == 'flat':
            return self.flat_sky(x)
        else:
            return self.tilted_plane_sky(x)

    @abstractmethod
    def render_sersic(self, xc,yc, flux, r_eff, n,ellip, theta):
        return NotImplementedError

    @abstractmethod
    def render_doublesersic(self,xc,yc, flux, f_1, r_eff_1,n_1, ellip_1, r_eff_2,n_2, ellip_2,theta):
        return NotImplementedError

    @abstractmethod
    def render_pointsource(self,xc,yc, flux):
        return NotImplementedError

    def render_exp(self, 
                xc: float,
                yc: float, 
                flux: float, 
                r_eff: float,
                ellip: float, 
                theta: float)-> jax.numpy.array:
        """Thin wrapper of an exponential profile based on render_sersic

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
        return self.render_sersic(xc,yc, flux, r_eff, 1.,ellip, theta)
    
    def render_dev(self, 
                xc: float,
                yc: float, 
                flux: float, 
                r_eff: float,
                ellip: float, 
                theta: float)-> jax.numpy.array:
        """Thin wrapper of a De Vaucouleurs profile based on render_sersic

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
        return self.render_sersic(xc,yc, flux, r_eff, 4.,ellip, theta)
    
    def render_source(self,
            params: jax.numpy.array,
            profile_type: str)->jax.numpy.array :
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
        render_func = getattr(self, f'render_{profile_type}')
        im = render_func(*params)
        return im


@jax.jit
def calc_sersic(X: jax.numpy.array,
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

class PixelRenderer(BaseRenderer):
    """
    Render class based on rendering in pixel space and then convolving with the PSF
    """
    #Basic implementation of Sersic renderering without any oversampling
    def __init__(self, 
            im_shape: Iterable, 
            pixel_PSF: jax.numpy.array,
            os_pixel_size: Optional[int]= 8, 
            num_os: Optional[int] = 8) -> None:
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

        #Set up quantities for injecting point sources
        self.X_psf,self.Y_psf = jnp.meshgrid(jnp.arange(self.psf_shape[0]), jnp.arange(self.psf_shape[1]))
        if self.psf_shape[0]%2 == 0:
            self.dx_ins_lo = (self.psf_shape[0]/2).astype(jnp.int32)
            self.dx_ins_hi = (self.psf_shape[0]/2).astype(jnp.int32)
        else:
            self.dx_ins_lo = jnp.floor(self.psf_shape[0]/2).astype(jnp.int32)
            self.dx_ins_hi = jnp.ceil(self.psf_shape[0]/2).astype(jnp.int32)

        if self.psf_shape[1]%2 == 0:
            self.dy_ins_lo = (self.psf_shape[1]/2).astype(jnp.int32)
            self.dy_ins_hi = (self.psf_shape[1]/2).astype(jnp.int32)
        else:
            self.dy_ins_lo = jnp.floor(self.psf_shape[1]/2).astype(jnp.int32)
            self.dy_ins_hi = jnp.ceil(self.psf_shape[1]/2).astype(jnp.int32)

        def conv(image):
            img_fft = jnp.fft.rfft2(image)
            conv_fft = img_fft*self.PSF_fft
            conv_im = jnp.fft.irfft2(conv_fft, s= self.im_shape)
            return jnp.abs(conv_im)
        self.conv = jit(conv)

        dx,w = np.polynomial.legendre.leggauss(self.num_os)
        dx = dx/2.
        self.dx_os,self.dy_os = jnp.meshgrid(dx,dx)

        w1,w2 = jnp.meshgrid(w,w)
        self.w_os = w1*w2/4.
        
        i_mid = int(self.im_shape[0]/2)
        j_mid = int(self.im_shape[1]/2)

        self.x_os_lo, self.x_os_hi = i_mid - self.os_pixel_size, i_mid + self.os_pixel_size
        self.y_os_lo, self.y_os_hi = j_mid - self.os_pixel_size, j_mid + self.os_pixel_size

        self.X_os = self.X[self.x_os_lo:self.x_os_hi ,self.y_os_lo:self.y_os_hi ,jnp.newaxis,jnp.newaxis] + self.dx_os
        self.Y_os = self.Y[self.x_os_lo:self.x_os_hi ,self.y_os_lo:self.y_os_hi ,jnp.newaxis,jnp.newaxis] + self.dy_os

        def render_int_sersic(xc,yc, flux, r_eff, n,ellip, theta):
            im_no_os = calc_sersic(self.X,self.Y,xc,yc, flux, r_eff, n,ellip, theta)
            
            sub_im_os = calc_sersic(self.X_os,self.Y_os,xc,yc, flux, r_eff, n,ellip, theta)
            sub_im_os = jnp.sum(sub_im_os*self.w_os, axis = (2,3))
            
            im = im_no_os.at[self.x_os_lo:self.x_os_hi ,self.y_os_lo:self.y_os_hi].set(sub_im_os)
            return im
        self.render_int_sersic = jit(render_int_sersic)

    def render_sersic(self,
            xc: float,
            yc: float,
            flux: float, 
            r_eff: float, 
            n: float,
            ellip: float, 
            theta: float)->jax.numpy.array:
        """_summary_

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
        im_int = self.render_int_sersic(xc,yc, flux, r_eff, n,ellip, theta)
        im = self.conv(im_int)
        return im
    
    def render_doublesersic(self,
            xc: float, 
            yc: float, 
            flux: float, 
            f_1: float, 
            r_eff_1: float, 
            n_1: float, 
            ellip_1: float, 
            r_eff_2: float, 
            n_2: float, 
            ellip_2: float, 
            theta: float
            ) -> jax.numpy.array:
        """Render a double Sersic profile with a common center and position angle

        Parameters
        ----------
        xc : float
            Central x position
        yc : float
            Central y position
        flux : float
            Total flux
        f_1 : float
            Fraction of flux in first component
        r_eff_1 : float
            Effective radius of first component
        n_1 : float
            Sersic index of first component
        ellip_1 : float
            Ellipticity of first component
        r_eff_2 : float
            Effective radius of second component
        n_2 : float
            Sersic index of second component
        ellip_2 : float
            Ellipticity of second component
        theta : float
            Position angle in radians

        Returns
        -------
        jax.numpy.array
            Rendered double Sersic model
        """
        im_int = self.render_int_sersic(xc,yc, flux*f_1, r_eff_1, n_1,ellip_1, theta) + self.render_int_sersic(xc,yc, flux*(1.-f_1), r_eff_2, n_2,ellip_2, theta)
        im = self.conv(im_int)
        return im
    
    def render_pointsource(self, 
            xc: float, 
            yc: float, 
            flux: float)-> jax.numpy.array:
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
        dx = xc - self.psf_shape[0]/2.
        dy = yc - self.psf_shape[1]/2.

        shifted_psf = jax.scipy.ndimage.map_coordinates(self.pixel_PSF*flux, [self.X-dx,self.Y-dy], order = 1, mode = 'constant')
    
        return shifted_psf

@jit
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


@jax.jit
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
    Ui = FX*jnp.cos(theta) + FY*jnp.sin(theta) 
    Vi = -1*FX*jnp.sin(theta) + FY*jnp.cos(theta) 

    in_exp = -1*(Ui*Ui + Vi*Vi*q*q)*(2*jnp.pi*jnp.pi*sigmas*sigmas)[:,jnp.newaxis,jnp.newaxis] - 1j*2*jnp.pi*FX*xc - 1j*2*jnp.pi*FY*yc
    Fgal_comp = amps[:,jnp.newaxis,jnp.newaxis]*jnp.exp(in_exp)
    Fgal = jnp.sum(Fgal_comp, axis = 0)
    return Fgal

@jax.jit
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

class FourierRenderer(BaseRenderer):
    """
    Class to render sources based on rendering them in Fourier space. Sersic profiles are modeled as a series of Gaussian following Shajib (2019)  and the implementation in lenstrometry
    """
    def __init__(self, 
            im_shape: Iterable, 
            pixel_PSF: jax.numpy.array,
            frac_start: Optional[float] = 0.02,
            frac_end: Optional[float] = 10., 
            n_sigma: Optional[int] = 11, 
            percision: Optional[int] = 5)-> None:
        """Initialize a Fourier renderer class

        Parameters
        ----------
        im_shape : Iterable
            Tuple or list containing the shape of the desired output
        pixel_PSF : jax.numpy.array
            Pixelized version of the PSF
        frac_start : Optional[float], optional
            Fraction of r_eff for the smallest Gaussian component, by default 0.02
        frac_end : Optional[float], optional
            Fraction of r_eff for the largest Gaussian component, by default 12.
        n_sigma : Optional[int], optional
            Number of Gaussian Components, by default 11
        percision : Optional[int], optional
            percision value used in calculating Gaussian components, see Shajib (2019) for more details, by default 5
        """
        super().__init__(im_shape, pixel_PSF)
        self.frac_start = frac_start
        self.frac_end = frac_end
        self.n_sigma = n_sigma
        self.percision = percision

        # Calculation of MoG representation of Sersic Profiles based on lenstrometry implementation and Shajib (2019)
        kes = np.arange(2 * self.percision + 1)
        betas = np.sqrt(2 * self.percision * np.log(10) / 3. + 2. * 1j * np.pi * kes)
        epsilons = np.zeros(2 * self.percision + 1)

        epsilons[0] = 0.5
        epsilons[1:self.percision + 1] = 1.
        epsilons[-1] = 1 / 2. ** self.percision

        for k in range(1, self.percision):
            epsilons[2 * self.percision - k] = epsilons[2 * self.percision - k + 1] + 1 / 2. ** self.percision * comb(
                self.percision, k)

        self.etas = jnp.array( (-1.) ** kes * epsilons * 10. ** (self.percision / 3.) * 2. * np.sqrt(2*np.pi) )
        self.betas = jnp.array(betas)

        #Define and JIT compile some functions
        def get_sersic_mog(flux,re,n):
            sigma_start = re*self.frac_start
            sigma_end = re*self.frac_end
            sigmas = jnp.logspace(jnp.log10(sigma_start),jnp.log10(sigma_end),num = self.n_sigma)

            f_sigmas = jnp.sum(self.etas * sersic1D(jnp.outer(sigmas,self.betas), flux,re,n).real,  axis=1)

            del_log_sigma = jnp.abs(jnp.diff(jnp.log(sigmas)).mean())

            amps = f_sigmas * del_log_sigma / jnp.sqrt(2*np.pi)

            # weighting for trapezoid method integral
            amps = amps.at[0].multiply(0.5)
            amps = amps.at[-1].multiply(0.5)

            amps = amps*2*np.pi*sigmas*sigmas
            return amps,sigmas
        self.get_sersic_mog = jit(get_sersic_mog)

        def render_sersic_mog_fourier(xc,yc, flux, r_eff, n,ellip, theta):
            amps,sigmas = self.get_sersic_mog(flux,r_eff,n)
            q = 1.-ellip
            Fgal = render_gaussian_fourier(self.FX,self.FY, amps,sigmas,xc,yc, theta,q)
            return Fgal
        self.render_sersic_mog_fourier = jit(render_sersic_mog_fourier)

        def conv_and_inv_FFT(F_im):
            im = jnp.abs( jnp.fft.irfft2(F_im*self.PSF_fft, s= self.im_shape) )
            return im
        self.conv_and_inv_FFT = jit(conv_and_inv_FFT)

    #Slower than pixel version, not sure exactly the cause, maybe try lax.scan instead of newaxis
    def render_sersic(self,
            xc: float,
            yc: float,
            flux: float, 
            r_eff: float, 
            n: float,
            ellip: float, 
            theta: float)->jax.numpy.array:
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
        F_im = self.render_sersic_mog_fourier(xc,yc, flux, r_eff, n,ellip, theta)
        im = self.conv_and_inv_FFT(F_im)
        return im

    def render_doublesersic(self,
            xc: float, 
            yc: float, 
            flux: float, 
            f_1: float, 
            r_eff_1: float, 
            n_1: float, 
            ellip_1: float, 
            r_eff_2: float, 
            n_2: float, 
            ellip_2: float, 
            theta: float
            ) -> jax.numpy.array:
        """Render a double Sersic profile with a common center and position angle

        Parameters
        ----------
        xc : float
            Central x position
        yc : float
            Central y position
        flux : float
            Total flux
        f_1 : float
            Fraction of flux in first component
        r_eff_1 : float
            Effective radius of first component
        n_1 : float
            Sersic index of first component
        ellip_1 : float
            Ellipticity of first component
        r_eff_2 : float
            Effective radius of second component
        n_2 : float
            Sersic index of second component
        ellip_2 : float
            Ellipticity of second component
        theta : float
            Position angle in radians

        Returns
        -------
        jax.numpy.array
            Rendered double Sersic model
        """
        F_im_1 = self.render_sersic_mog_fourier(xc,yc, flux*f_1, r_eff_1, n_1,ellip_1, theta)
        F_im_2 = self.render_sersic_mog_fourier(xc,yc, flux*(1.-f_1), r_eff_2, n_2,ellip_2, theta)
        im = self.conv_and_inv_FFT(F_im_1 + F_im_2)
        return im
    
    def render_pointsource(self, 
            xc: float, 
            yc: float, 
            flux: float)-> jax.numpy.array:
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
        F_im = render_pointsource_fourier(self.FX,self.FY,xc,yc,flux)
        im = self.conv_and_inv_FFT(F_im)
        return im

    def render_multi(self, 
            type_list: Iterable, 
            var_list: Iterable)-> jax.numpy.array:
        """Function to render multiple sources in the same image

        Parameters
        ----------
        type_list : Iterable
            List of strings containing the types of sources
        var_list : Iterable
            List of arrays contiaining the variables for each profile

        Returns
        -------
        jax.numpy.array
            Rendered image
        """
        
        F_tot = jnp.zeros_like(self.FX)

        for ind in range(len(type_list)):
            if type_list[ind] == 'pointsource':
                F_tot = F_tot + render_pointsource_fourier(self.FX,self.FY,*var_list[ind])
            elif type_list[ind] == 'sersic':
                F_tot = F_tot + self.render_sersic_mog_fourier(*var_list[ind])
            elif type_list[ind] == 'exp':
                xc,yc, flux, r_eff,ellip, theta = var_list[ind]
                F_tot = F_tot + self.render_sersic_mog_fourier(xc,yc, flux, r_eff,1.,ellip, theta)
            elif type_list[ind] == 'dev':
                xc,yc, flux, r_eff,ellip, theta = var_list[ind]
                F_tot = F_tot + self.render_sersic_mog_fourier(xc,yc, flux, r_eff,4.,ellip, theta)
            elif type_list[ind] == 'doublesersic':
                xc, yc, flux, f_1, r_eff_1, n_1, ellip_1, r_eff_2, n_2, ellip_2, theta = var_list[ind]
                F_im_1 = self.render_sersic_mog_fourier(xc,yc, flux*f_1, r_eff_1, n_1,ellip_1, theta)
                F_im_2 = self.render_sersic_mog_fourier(xc,yc, flux*(1.-f_1), r_eff_2, n_2,ellip_2, theta)
                F_tot = F_tot + F_im_1 + F_im_2

        im = self.conv_and_inv_FFT(F_tot)
        return im

@jax.jit
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


class HybridRenderer(BaseRenderer):
    """
    Class to render sources based on the hybrid rendering scheme introduced in Lang (2020). Sersic profiles are modeled as a series of Gaussian following Shajib (2019) and the implementation in lenstrometry
    """
    def __init__(self, 
            im_shape: Iterable, 
            pixel_PSF: jax.numpy.array,
            frac_start: Optional[float] = 0.02,
            frac_end: Optional[float] = 10., 
            n_sigma: Optional[int] = 11, 
            num_pixel_render: Optional[int] = 4,
            percision: Optional[int] = 5)-> None:
        """Initialize a  HyrbridRenderer class

        Parameters
        ----------
        im_shape : Iterable
            Tuple or list containing the shape of the desired output
        pixel_PSF : jax.numpy.array
            Pixelized version of the PSF
        frac_start : Optional[float], optional
            Fraction of r_eff for the smallest Gaussian component, by default 0.02
        frac_end : Optional[float], optional
            Fraction of r_eff for the largest Gaussian component, by default 12.
        n_sigma : Optional[int], optional
            Number of Gaussian Components, by default 11
        num_pixel_Render : Optional[int], optional
            Numeber of components to render in pixel space, counts back from largest component
        percision : Optional[int], optional
            percision value used in calculating Gaussian components, see Shajib (2019) for more details, by default 5
        """
        super().__init__(im_shape, pixel_PSF)
        self.frac_start = frac_start
        self.frac_end = frac_end
        self.n_sigma = n_sigma
        self.percision = percision
        self.num_pixel_render = num_pixel_render
        self.w_real = jnp.arange(self.n_sigma - self.num_pixel_render, self.n_sigma)
        self.w_fourier = jnp.arange(self.n_sigma - self.num_pixel_render)

        psf_X,psf_Y = jnp.meshgrid(jnp.arange(self.psf_shape[0]),jnp.arange(self.psf_shape[1]))
        sig_x = jnp.sqrt( (self.pixel_PSF*(psf_X)**2).sum()/self.pixel_PSF.sum() - psf_X.mean()**2 )
        sig_y = jnp.sqrt( (self.pixel_PSF*(psf_Y)**2).sum()/self.pixel_PSF.sum() - psf_Y.mean()**2 )
        self.sig_psf_approx = 0.5*(sig_x + sig_y)

        # Calculation of MoG representation of Sersic Profiles based on lenstrometry implementation and Shajib (2019)
        kes = np.arange(2 * self.percision + 1)
        betas = np.sqrt(2 * self.percision * np.log(10) / 3. + 2. * 1j * np.pi * kes)
        epsilons = np.zeros(2 * self.percision + 1)

        epsilons[0] = 0.5
        epsilons[1:self.percision + 1] = 1.
        epsilons[-1] = 1 / 2. ** self.percision

        for k in range(1, self.percision):
            epsilons[2 * self.percision - k] = epsilons[2 * self.percision - k + 1] + 1 / 2. ** self.percision * comb(
                self.percision, k)

        self.etas = jnp.array( (-1.) ** kes * epsilons * 10. ** (self.percision / 3.) * 2. * np.sqrt(2*np.pi) )
        self.betas = jnp.array(betas)

        #Define and JIT compile some functions
        def get_sersic_mog(flux,re,n):
            sigma_start = re*self.frac_start
            sigma_end = re*self.frac_end
            sigmas = jnp.logspace(jnp.log10(sigma_start),jnp.log10(sigma_end),num = self.n_sigma)

            f_sigmas = jnp.sum(self.etas * sersic1D(jnp.outer(sigmas,self.betas), flux,re,n).real,  axis=1)

            del_log_sigma = jnp.abs(jnp.diff(jnp.log(sigmas)).mean())

            amps = f_sigmas * del_log_sigma / jnp.sqrt(2*np.pi)

            # weighting for trapezoid method integral
            amps = amps.at[0].multiply(0.5)
            amps = amps.at[-1].multiply(0.5)

            amps = amps*2*np.pi*sigmas*sigmas
            return amps,sigmas
        self.get_sersic_mog = jit(get_sersic_mog)


        def render_sersic_hybrid(xc,yc, flux, r_eff, n,ellip, theta):
            amps,sigmas = self.get_sersic_mog(flux,r_eff,n)
            q = 1.-ellip

            sigmas_obs = jnp.sqrt(sigmas**2 + self.sig_psf_approx**2)
            q_obs = jnp.sqrt( (q*q*sigmas**2 + self.sig_psf_approx**2)/ sigmas_obs**2 )

            Fgal = render_gaussian_fourier(self.FX,self.FY, amps[self.w_fourier],sigmas[self.w_fourier],xc,yc, theta,q)

            im_gal = render_gaussian_pixel(self.X,self.Y, amps[self.w_real],sigmas_obs[self.w_real],xc,yc, theta,q_obs[self.w_real])

            return Fgal,im_gal
        self.render_sersic_hyrbid = jax.jit(render_sersic_hybrid)

        def conv_and_inv_FFT(F_im):
            im = jnp.abs( jnp.fft.irfft2(F_im*self.PSF_fft, s= self.im_shape) )
            return im
        self.conv_and_inv_FFT = jit(conv_and_inv_FFT)

    def render_sersic(self,
            xc: float,
            yc: float,
            flux: float, 
            r_eff: float, 
            n: float,
            ellip: float, 
            theta: float)->jax.numpy.array:
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
        F, im  = self.render_sersic_hyrbid(xc,yc, flux, r_eff, n,ellip, theta)
        out = im + self.conv_and_inv_FFT(F)
        return out

    def render_doublesersic(self,
            xc: float, 
            yc: float, 
            flux: float, 
            f_1: float, 
            r_eff_1: float, 
            n_1: float, 
            ellip_1: float, 
            r_eff_2: float, 
            n_2: float, 
            ellip_2: float, 
            theta: float
            ) -> jax.numpy.array:
        """Render a double Sersic profile with a common center and position angle

        Parameters
        ----------
        xc : float
            Central x position
        yc : float
            Central y position
        flux : float
            Total flux
        f_1 : float
            Fraction of flux in first component
        r_eff_1 : float
            Effective radius of first component
        n_1 : float
            Sersic index of first component
        ellip_1 : float
            Ellipticity of first component
        r_eff_2 : float
            Effective radius of second component
        n_2 : float
            Sersic index of second component
        ellip_2 : float
            Ellipticity of second component
        theta : float
            Position angle in radians

        Returns
        -------
        jax.numpy.array
            Rendered double Sersic model
        """
        F_1, im_1 = self.render_sersic_hyrbid(xc,yc, flux*f_1, r_eff_1, n_1,ellip_1, theta)
        F_2, im_2 = self.render_sersic_hyrbid(xc,yc, flux*(1.-f_1), r_eff_2, n_2,ellip_2, theta)
        im = im_1 + im_2  + self.conv_and_inv_FFT(F_1 + F_2)
        return im

    def render_pointsource(self, 
            xc: float, 
            yc: float, 
            flux: float)-> jax.numpy.array:
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
        F_im = render_pointsource_fourier(self.FX,self.FY,xc,yc,flux)
        im = self.conv_and_inv_FFT(F_im)
        return im


    def render_multi(self, 
            type_list: Iterable, 
            var_list: Iterable)-> jax.numpy.array:
        """Function to render multiple sources in the same image

        Parameters
        ----------
        type_list : Iterable
            List of strings containing the types of sources
        var_list : Iterable
            List of arrays contiaining the variables for each profile

        Returns
        -------
        jax.numpy.array
            Rendered image
        """
        
        F_tot = jnp.zeros_like(self.FX)
        im_tot = np.zeros_like(self.X)

        for ind in range(len(type_list)):
            if type_list[ind] == 'pointsource':
                F_tot = F_tot + render_pointsource_fourier(self.FX,self.FY,*var_list[ind])
            elif type_list[ind] == 'sersic':
                F_cur, im_cur  = self.render_sersic_hyrbid(*var_list[ind])
                F_tot = F_tot + F_cur
                im_tot = im_tot + im_cur
            elif type_list[ind] == 'exp':
                xc,yc, flux, r_eff,ellip, theta = var_list[ind]
                F_cur, im_cur  = self.render_sersic_hyrbid(xc,yc, flux, r_eff,1.,ellip, theta)
                F_tot = F_tot + F_cur
                im_tot = im_tot + im_cur
            elif type_list[ind] == 'dev':
                xc,yc, flux, r_eff,ellip, theta = var_list[ind]
                F_cur, im_cur  = self.render_sersic_hyrbid(xc,yc, flux, r_eff,4.,ellip, theta)
                F_tot = F_tot + F_cur
            elif type_list[ind] == 'doublesersic':
                xc, yc, flux, f_1, r_eff_1, n_1, ellip_1, r_eff_2, n_2, ellip_2, theta = var_list[ind]
                F_1, im_1 = self.render_sersic_hyrbid(xc,yc, flux*f_1, r_eff_1, n_1,ellip_1, theta)
                F_2, im_2 = self.render_sersic_hyrbid(xc,yc, flux*(1.-f_1), r_eff_2, n_2,ellip_2, theta)
                F_tot = F_tot + F_1 + F_2
                im_tot = im_tot + im_1 + im_2

        im = self.conv_and_inv_FFT(F_tot) + im_tot
        return im
