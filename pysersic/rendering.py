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
        self.x_mid = self.im_shape[0]/2. - 0.5
        self.y_mid = self.im_shape[1]/2. - 0.5

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
        return x[0] + (self.X -  self.x_mid)*x[1] + (self.Y - self.y_mid)*x[2]
    
    def render_sky(self,x, sky_type):
        if sky_type is None:
            return 0
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

    def render_exp(self, xc,yc, flux, r_eff,ellip, theta):
        return self.render_sersic(xc,yc, flux, r_eff, 1.,ellip, theta)
    
    def render_dev(self, xc,yc, flux, r_eff,ellip, theta):
        return self.render_sersic(xc,yc, flux, r_eff, 4.,ellip, theta)
    
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


@jax.jit
def calc_sersic(X,Y,xc,yc, flux, r_eff, n,ellip, theta):
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
    #Basic implementation of Sersic renderering without any oversampling
    def __init__(self, im_shape, pixel_PSF, os_pixel_size = 5, num_os = 8):
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

    def render_sersic(self,xc,yc, flux, r_eff, n,ellip, theta):
        im_int = self.render_int_sersic(xc,yc, flux, r_eff, n,ellip, theta)
        im = self.conv(im_int)
        return im
    
    #Currently not optimized for multiple sources as FFT is done every time, can change later.
    def render_doublesersic(self, xc, yc, flux, f_1, r_eff_1, n_1, ellip_1, r_eff_2, n_2, ellip_2, theta):
        im_int = self.render_int_sersic(xc,yc, flux*f_1, r_eff_1, n_1,ellip_1, theta) + self.render_int_sersic(xc,yc, flux*(1.-f_1), r_eff_2, n_2,ellip_2, theta)
        im = self.conv(im_int)
        return im
    
    def render_pointsource(self, xc, yc, flux):
        dx = xc - self.psf_shape[0]/2.
        dy = yc - self.psf_shape[1]/2.

        shifted_psf = jax.scipy.ndimage.map_coordinates(self.pixel_PSF*flux, [self.X-dx,self.Y-dy], order = 1, mode = 'constant')
    
        return shifted_psf

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

        def render_sersic_mog_fourier(xc,yc, flux, r_eff, n,ellip, theta):
            amps,sigmas = self.get_sersic_mog(flux,r_eff,n)

            q = 1.-ellip
            Ui = self.FX*jnp.cos(theta) + self.FY*jnp.sin(theta) 
            Vi = -1*self.FX*jnp.sin(theta) + self.FY*jnp.cos(theta) 

            in_exp = -1*(Ui*Ui + Vi*Vi*q*q)*(2*jnp.pi*jnp.pi*sigmas*sigmas)[:,jnp.newaxis,jnp.newaxis] - 1j*2*jnp.pi*self.FX*xc - 1j*2*jnp.pi*self.FY*yc
            Fgal_comp = amps[:,jnp.newaxis,jnp.newaxis]*jnp.exp(in_exp)
            Fgal = jnp.sum(Fgal_comp, axis = 0)
            return Fgal
        self.render_sersic_mog_fourier = jit(render_sersic_mog_fourier)

        def render_pointsource_fourier(xc, yc, flux):
            in_exp = -1j*2*jnp.pi*self.FX*xc - 1j*2*jnp.pi*self.FY*yc
            F_im = flux*jnp.exp(in_exp)
            return F_im
        self.render_pointsource_fourier = jit(render_pointsource_fourier)

        def conv_and_inv_FFT(F_im):
            im = jnp.abs( jnp.fft.irfft2(F_im*self.PSF_fft, s= self.im_shape) )
            return im
        self.conv_and_inv_FFT = jit(conv_and_inv_FFT)

    #Slower than pixel version, not sure exactly the cause, maybe try lax.scan instead of newaxis
    def render_sersic(self,xc,yc, flux, r_eff, n,ellip, theta):
        F_im = self.render_sersic_mog_fourier(xc,yc, flux, r_eff, n,ellip, theta)
        im = self.conv_and_inv_FFT(F_im)
        return im

    def render_doublesersic(self, xc, yc, flux, f_1, r_eff_1, n_1, ellip_1, r_eff_2, n_2, ellip_2, theta):
        F_im_1 = self.render_sersic_mog_fourier(xc,yc, flux*f_1, r_eff_1, n_1,ellip_1, theta)
        F_im_2 = self.render_sersic_mog_fourier(xc,yc, flux*(1.-f_1), r_eff_2, n_2,ellip_2, theta)
        im = self.conv_and_inv_FFT(F_im_1 + F_im_2)
        return im
    
    def render_pointsource(self, xc, yc, flux):
        F_im = self.render_pointsource_fourier(xc,yc,flux)
        im = self.conv_and_inv_FFT(F_im)
        return im

    def render_multi(self, type_list, var_list):
        
        F_tot = jnp.zeros_like(self.FX)

        for ind in range(len(type_list)):
            if type_list[ind] == 'pointsource':
                F_tot = F_tot + self.render_pointsource_fourier(*var_list[ind])
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
