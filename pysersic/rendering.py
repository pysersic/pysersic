import jax
from jax import jit
from functools import partial
import jax.numpy as jnp
import numpy as np
from scipy.special import comb
from abc import abstractmethod

@jax.jit
def conv_fft(image,psf_fft):
    img_fft = jnp.fft.rfft2(image)
    conv_fft = img_fft*psf_fft
    conv_im = jnp.fft.irfft2(conv_fft)
    return jnp.abs(conv_im)

class BaseRenderer(object):
    def __init__(self, im_shape, pixel_PSF):
        self.im_shape = im_shape
        self.pixel_PSF = jnp.array(pixel_PSF)
        self.psf_shape = self.pixel_PSF.shape

        x = jnp.arange(im_shape[0])
        y = jnp.arange(im_shape[1])
        self.X,self.Y = jnp.meshgrid(x,y)
        
        # Set up pre-FFTed PSF
        f1d1 = jnp.fft.rfftfreq(self.im_shape[0])
        f1d2 = jnp.fft.fftfreq(self.im_shape[1])
        self.FX,self.FY = jnp.meshgrid(f1d1,f1d2)
        fft_shift_arr_x = jnp.exp(jax.lax.complex(0.,-1.)*2.*3.1415*-1*self.psf_shape[0]/2.*self.FX)
        fft_shift_arr_y = jnp.exp(jax.lax.complex(0.,-1.)*2.*3.1415*-1*self.psf_shape[1]/2.*self.FY)
        self.PSF_fft = jnp.fft.rfft2(self.pixel_PSF, s = self.im_shape)*fft_shift_arr_x*fft_shift_arr_y

    @abstractmethod
    def render_sersic(self, x_0,y_0, flux, r_eff, n,ellip, theta):
        return NotImplementedError

    @abstractmethod
    def render_doublesersic(self,x_0,y_0, flux, f_1, r_eff_1,n_1, ellip_1, r_eff_2,n_2, ellip_2,theta):
        return NotImplementedError

    @abstractmethod
    def render_pointsource(self,x_0,y_0, flux):
        return NotImplementedError

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

        def Sersic2D(x_0,y_0, flux, r_eff, n,ellip, theta):
            bn = 1.9992*n - 0.3271
            a, b = r_eff, (1 - ellip) * r_eff
            cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)
            x_maj = (self.X - x_0) * cos_theta + (self.Y - y_0) * sin_theta
            x_min = -(self.X - x_0) * sin_theta + (self.Y - y_0) * cos_theta
            amplitude = flux*bn**(2*n) / ( jnp.exp(bn + jax.scipy.special.gammaln(2*n) ) *r_eff**2 *jnp.pi*2*n )
            z = jnp.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)
            out = amplitude * jnp.exp(-bn * (z ** (1 / n) - 1))
            out = conv_fft(out,self.PSF_fft)
            return out

        self.Sersic2D = jax.jit(Sersic2D)

    def render_sersic(self,x_0,y_0, flux, r_eff, n,ellip, theta):
         return self.Sersic2D(x_0,y_0, flux, r_eff, n,ellip, theta)
    
    #Currently not optimized for multiple sources as FFT is done every time, can change later.
    def render_doublesersic(self, x_0, y_0, flux, f_1, r_eff_1, n_1, ellip_1, r_eff_2, n_2, ellip_2, theta):
        return self.render_sersic(x_0,y_0, flux*f_1, r_eff_1, n_1,ellip_1, theta) + self.render_sersic(x_0,y_0, flux*(1.-f_1), r_eff_2, n_2,ellip_2, theta)
    
    
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

class FourierRenderer(BaseRenderer):
    def __init__(self, im_shape, pixel_PSF,frac_start = 0.02,frac_end = 12., n_sigma = 11, precision = 5):
        super().__init__(im_shape, pixel_PSF)
        self.frac_start = frac_start
        self.frac_end = frac_end
        self.n_sigma = n_sigma
        self.percision = precision

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
    
    @partial(jit, static_argnums=0)
    def sersic1D(self, r,flux,re,n):
        bn = 1.9992*n - 0.3271
        Ie = flux / ( re*re* 2* jnp.pi*n * jnp.exp(bn + jax.scipy.special.gammaln(2*n) ) ) * bn**(2*n) 
        return Ie*jnp.exp ( -bn*( (r/re)**(1./n) - 1. ) )
    
    @partial(jit, static_argnums=0)
    def get_sersic_mog(self,flux,re,n):
        sigma_start = re*self.frac_start
        sigma_end = re*self.frac_end
        sigmas = jnp.logspace(jnp.log10(sigma_start),jnp.log10(sigma_end),num = self.n_sigma)

        f_sigmas = jnp.sum(self.etas * self.sersic1D(jnp.outer(sigmas,self.betas), flux,re,n).real,  axis=1)

        del_log_sigma = jnp.abs(jnp.diff(jnp.log(sigmas)).mean())

        amps = f_sigmas * del_log_sigma / jnp.sqrt(2*np.pi)

        # weighting for trapezoid method integral
        amps = amps.at[0].multiply(0.5)
        amps = amps.at[-1].multiply(0.5)

        amps = amps*2*np.pi*sigmas*sigmas
        return amps,sigmas

    #Slower than pixel version, not sure exactly the cause, maybe try lax.scan instead of newaxis
    @partial(jit, static_argnums=0)
    def render_sersic(self,x_0,y_0, flux, r_eff, n,ellip, theta):
        amps,sigmas = self.get_sersic_mog(flux,r_eff,n)

        q = 1.-ellip
        Ui = self.FX*jnp.cos(theta) + self.FY*jnp.sin(theta) 
        Vi = -1*self.FX*jnp.sin(theta) + self.FY*jnp.cos(theta) 

        in_exp = -1*(Ui*Ui + Vi*Vi*q*q)*(2*jnp.pi*jnp.pi*sigmas*sigmas)[:,jnp.newaxis,jnp.newaxis] - 1j*2*jnp.pi*self.FX*x_0 - 1j*2*jnp.pi*self.FY*y_0
        Fgal = amps[:,jnp.newaxis,jnp.newaxis]*jnp.exp(in_exp)

        Fgal = jnp.sum(Fgal, axis = 0)
        gal_im = jnp.abs( jnp.fft.irfft2(Fgal*self.PSF_fft) )

        return gal_im

    @partial(jit, static_argnums=0)
    def render_doublesersic(self, x_0, y_0, flux, f_1, r_eff_1, n_1, ellip_1, r_eff_2, n_2, ellip_2, theta):
        Ui = self.FX*jnp.cos(theta) + self.FY*jnp.sin(theta) 
        Vi = -1*self.FX*jnp.sin(theta) + self.FY*jnp.cos(theta) 
        F_shift = - 1j*2*jnp.pi*self.FX*x_0 - 1j*2*jnp.pi*self.FY*y_0
        Ui_sq = Ui * Ui
        Vi_sq = Vi * Vi

        amps_1,sigmas_1 = self.get_sersic_mog(flux*f_1,r_eff_1,n_1)
        q_1 = 1-ellip_1
        in_exp_1 = -1*(Ui_sq + Vi_sq*q_1*q_1)*(2*jnp.pi*jnp.pi*sigmas_1*sigmas_1)[:,jnp.newaxis,jnp.newaxis] + F_shift
        Fgal_1 = amps_1[:,jnp.newaxis,jnp.newaxis]*jnp.exp(in_exp_1)
        Fgal_1 = jnp.sum(Fgal_1, axis = 0)

        amps_2,sigmas_2 = self.get_sersic_mog(flux*(1-f_1),r_eff_2,n_2)
        q_2 = 1-ellip_2
        in_exp_2 = -1*(Ui_sq+ Vi_sq*q_2*q_2)*(2*jnp.pi*jnp.pi*sigmas_2*sigmas_2)[:,jnp.newaxis,jnp.newaxis] + F_shift
        Fgal_2 = amps_2[:,jnp.newaxis,jnp.newaxis]*jnp.exp(in_exp_2)
        Fgal_2 = jnp.sum(Fgal_2, axis = 0)

        gal_im = jnp.abs( jnp.fft.irfft2((Fgal_1 + Fgal_2)*self.PSF_fft) )
        return gal_im
    
    @partial(jit, static_argnums=0)
    def render_pointsource(self, x_0, y_0, flux):
        j_jax = jax.lax.complex(0.,1.)
        in_exp = -1*j_jax*2*jnp.pi*self.FX*x_0 - 1*j_jax*2*jnp.pi*self.FY*y_0
        F_im = flux*jnp.exp(in_exp)
        return jnp.abs( jnp.fft.irfft2(F_im*self.PSF_fft) )

