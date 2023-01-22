import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
from scipy.special import comb
from abc import abstractmethod


class BaseRenderer(object):
    def __init__(self, im_shape, pixel_PSF):
        self.im_shape = im_shape
        self.pixel_PSF = pixel_PSF
        self.psf_shape = jnp.shape(self.pixel_PSF)

        self.x = jnp.arange(self.im_shape[0])
        self.y = jnp.arange(self.im_shape[1])
        self.X,self.Y = jnp.meshgrid(self.x,self.y)
        
        # Set up pre-FFTed PSF
        f1d1 = jnp.fft.rfftfreq(self.im_shape[0])
        f1d2 = jnp.fft.fftfreq(self.im_shape[1])
        self.FX,self.FY = jnp.meshgrid(f1d1,f1d2)
        fft_shift_arr_x = jnp.exp(jax.lax.complex(0.,-1.)*2.*3.1415*-1*self.psf_shape[0]/2.*self.FX)
        fft_shift_arr_y = jnp.exp(jax.lax.complex(0.,-1.)*2.*3.1415*-1*self.psf_shape[1]/2.*self.FY)
        self.PSF_fft = jnp.fft.rfft2(self.pixel_PSF, s = self.im_shape)*fft_shift_arr_x*fft_shift_arr_y

    def flat_sky(self,x):
        return x

    def tilted_plane_sky(self, x):
        return x[0] + (self.X -  self.im_shape[0]/2.)*x[1] + (self.Y - self.im_shape[1]/2.)*x[2]
    
    def render_sky(self,x, sky_type):
        if sky_type is None:
            return 0
        elif sky_type == 'flat':
            return self.flat_sky(x)
        else:
            return self.tilted_plane_sky(x)

    @abstractmethod
    def render_sersic(self, x_0,y_0, flux, r_eff, n,ellip, theta):
        return NotImplementedError

    @abstractmethod
    def render_doublesersic(self,x_0,y_0, flux, f_1, r_eff_1,n_1, ellip_1, r_eff_2,n_2, ellip_2,theta):
        return NotImplementedError

    @abstractmethod
    def render_pointsource(self,x_0,y_0, flux):
        return NotImplementedError

    def render_exp(self, x_0,y_0, flux, r_eff,ellip, theta):
        return self.render_sersic( x_0,y_0, flux, r_eff, 1.,ellip, theta)
    
    def render_dev(self, x_0,y_0, flux, r_eff,ellip, theta):
        return self.render_sersic( x_0,y_0, flux, r_eff, 4.,ellip, theta)
    
    def render_source(self,params,profile_type):
        if profile_type == 'sersic':
            im = self.render_sersic(*params)
        elif profile_type == 'doublesersic':
            im = self.render_doublesersic(*params)
        elif profile_type == 'exp':
            im = self.render_exp(*params)
        elif profile_type == 'dev':
            im = self.render_dev(*params)
        else:
            im = self.render_pointsource(*params)
        return im

class PixelRenderer(BaseRenderer):
    #Basic implementation of Sersic renderering without any oversampling
    def __init__(self, im_shape, pixel_PSF):
        super().__init__(im_shape, pixel_PSF)
        
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

        def render_int_sersic(x_0,y_0, flux, r_eff, n,ellip, theta):
            bn = 1.9992*n - 0.3271
            a, b = r_eff, (1 - ellip) * r_eff
            cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)
            x_maj = (self.X - x_0) * cos_theta + (self.Y - y_0) * sin_theta
            x_min = -(self.X - x_0) * sin_theta + (self.Y - y_0) * cos_theta
            amplitude = flux*bn**(2*n) / ( jnp.exp(bn + jax.scipy.special.gammaln(2*n) ) *r_eff**2 *jnp.pi*2*n )
            z = jnp.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)
            out = amplitude * jnp.exp(-bn * (z ** (1 / n) - 1))
            return out
        self.render_int_sersic = jit(render_int_sersic)

        def conv(image):
            img_fft = jnp.fft.rfft2(image)
            conv_fft = img_fft*self.PSF_fft
            conv_im = jnp.fft.irfft2(conv_fft)
            return jnp.abs(conv_im)
        self.conv = jit(conv)

    def render_sersic(self,x_0,y_0, flux, r_eff, n,ellip, theta):
        im_int = self.render_int_sersic(x_0,y_0, flux, r_eff, n,ellip, theta)
        im = self.conv(im_int)
        return im
    
    #Currently not optimized for multiple sources as FFT is done every time, can change later.
    def render_doublesersic(self, x_0, y_0, flux, f_1, r_eff_1, n_1, ellip_1, r_eff_2, n_2, ellip_2, theta):
        im_int = self.render_int_sersic(x_0,y_0, flux*f_1, r_eff_1, n_1,ellip_1, theta) + self.render_int_sersic(x_0,y_0, flux*(1.-f_1), r_eff_2, n_2,ellip_2, theta)
        im = self.conv(im_int)
        return im
    
    #Cannot jit due to dynamic array addition
    def render_pointsource(self, x_0, y_0, flux):
        x_int = jnp.round(x_0).astype(jnp.int32)
        y_int = jnp.round(y_0).astype(jnp.int32)
        dx = x_0 - x_int
        dy = y_0 - y_int

        shifted_psf = jax.scipy.ndimage.map_coordinates(self.pixel_PSF, [self.X_psf+dx,self.Y_psf+dy], order = 1, mode = 'constant')
        
        im = jnp.zeros(self.im_shape)
        im = im.at[x_int - self.dx_ins_lo:x_int + self.dx_ins_hi, y_int - self.dy_ins_lo:y_int + self.dy_ins_hi].add(flux*shifted_psf)
        return im

@jit
def sersic1D(r,flux,re,n):
    bn = 1.9992*n - 0.3271
    Ie = flux / ( re*re* 2* jnp.pi*n * jnp.exp(bn + jax.scipy.special.gammaln(2*n) ) ) * bn**(2*n) 
    return Ie*jnp.exp ( -bn*( (r/re)**(1./n) - 1. ) )

class FourierRenderer(BaseRenderer):
    def __init__(self, im_shape, pixel_PSF,frac_start = 0.02,frac_end = 12., n_sigma = 11, percision = 5):
        super().__init__(im_shape, pixel_PSF)
        self.frac_start = frac_start
        self.frac_end = frac_end
        self.n_sigma = n_sigma
        self.percision = percision

        # Calculation of MoG representation of SersicProfiles based on lenstrometry implementation and Shajib (2019)
        # nodes and weights based on Fourier-Euler method
        # for details Abate & Whitt (2006)
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

        def render_sersic_mog_fourier(x_0,y_0, flux, r_eff, n,ellip, theta):
            amps,sigmas = self.get_sersic_mog(flux,r_eff,n)

            q = 1.-ellip
            Ui = self.FX*jnp.cos(theta) + self.FY*jnp.sin(theta) 
            Vi = -1*self.FX*jnp.sin(theta) + self.FY*jnp.cos(theta) 

            in_exp = -1*(Ui*Ui + Vi*Vi*q*q)*(2*jnp.pi*jnp.pi*sigmas*sigmas)[:,jnp.newaxis,jnp.newaxis] - 1j*2*jnp.pi*self.FX*x_0 - 1j*2*jnp.pi*self.FY*y_0
            Fgal_comp = amps[:,jnp.newaxis,jnp.newaxis]*jnp.exp(in_exp)
            Fgal = jnp.sum(Fgal_comp, axis = 0)
            return Fgal
        self.render_sersic_mog_fourier = jit(render_sersic_mog_fourier)

        def render_pointsource_fourier(x_0, y_0, flux):
            in_exp = -1j*2*jnp.pi*self.FX*x_0 - 1j*2*jnp.pi*self.FY*y_0
            F_im = flux*jnp.exp(in_exp)
            return F_im
        self.render_pointsource_fourier = jit(render_pointsource_fourier)

        def conv_and_inv_FFT(F_im):
            im = jnp.abs( jnp.fft.irfft2(F_im*self.PSF_fft) )
            return im
        self.conv_and_inv_FFT = jit(conv_and_inv_FFT)

    #Slower than pixel version, not sure exactly the cause, maybe try lax.scan instead of newaxis
    def render_sersic(self,x_0,y_0, flux, r_eff, n,ellip, theta):
        F_im = self.render_sersic_mog_fourier(x_0,y_0, flux, r_eff, n,ellip, theta)
        im = self.conv_and_inv_FFT(F_im)
        return im

    def render_doublesersic(self, x_0, y_0, flux, f_1, r_eff_1, n_1, ellip_1, r_eff_2, n_2, ellip_2, theta):
        F_im_1 = self.render_sersic_mog_fourier(x_0,y_0, flux*f_1, r_eff_1, n_1,ellip_1, theta)
        F_im_2 = self.render_sersic_mog_fourier(x_0,y_0, flux*(1.-f_1), r_eff_2, n_2,ellip_2, theta)
        im = self.conv_and_inv_FFT(F_im_1 + F_im_2)
        return im
    
    def render_pointsource(self, x_0, y_0, flux):
        F_im = self.render_pointsource_fourier(x_0,y_0,flux)
        im = self.conv_and_inv_FFT(F_im)
        return im