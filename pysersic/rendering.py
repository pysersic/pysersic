import warnings
from abc import abstractmethod
from typing import Iterable, Optional, Tuple, Union,Literal
from numpyro import deterministic
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from interpax import interp1d
from scipy.special import comb

from .exceptions import *

base_profile_types = [
    "sersic",
    "doublesersic",
    "sersic_exp",
    "sersic_pointsource",
    "pointsource",
    "exp",
    "dev",
    "spergel"
]
base_profile_params = dict(
    zip(
        base_profile_types,
        [
            ["xc", "yc", "flux", "r_eff", "n", "ellip", "theta"],
            [
                "xc",
                "yc",
                "flux",
                "f_1",
                "r_eff_1",
                "n_1",
                "ellip_1",
                "r_eff_2",
                "n_2",
                "ellip_2",
                "theta",
            ],
            [
                "xc",
                "yc",
                "flux",
                "f_1",
                "r_eff_1",
                "ellip_1",
                "r_eff_2",
                "n",
                "ellip_2",
                "theta",
            ],
            ["xc", "yc", "flux", "f_ps", "r_eff", "n", "ellip", "theta"],
            ["xc", "yc", "flux"],
            ["xc", "yc", "flux", "r_eff", "ellip", "theta"],
            ["xc", "yc", "flux", "r_eff", "ellip", "theta"],
            ["xc", "yc", "flux", "r_eff", "nu", "ellip", "theta"]
        ],
    )
)


class BaseRenderer(eqx.Module):
    im_shape: tuple = eqx.field(static=True)
    psf_shape: tuple = eqx.field(static=True)
    fft_shape: tuple = eqx.field(static=True)
    profile_func_dict: dict = eqx.field(static=True)

    pixel_PSF: jax.numpy.array
    PSF_fft: jax.numpy.array
    X: jax.numpy.array
    Y: jax.numpy.array
    FX: jax.numpy.array
    FY: jax.numpy.array
    x_mid: float
    y_mid: float
    fft_zeros: jnp.array
    img_zeros: jnp.array

    def __init__(self, im_shape: Iterable, pixel_PSF: jax.numpy.array) -> None:
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
        if not jnp.isclose(jnp.sum(self.pixel_PSF), 1.0, 0.1):
            warnings.warn(
                "PSF does not appear to be appropriately normalized; Sum(psf) is more than 0.1 away from 1."
            )
        self.psf_shape = jnp.shape(self.pixel_PSF)
        if jnp.any(self.im_shape < self.psf_shape):
            raise KernelError(
                "PSF pixel image size must be smaller than science image."
            )
        x = jnp.arange(self.im_shape[0])
        y = jnp.arange(self.im_shape[1])
        self.X, self.Y = jnp.meshgrid(x, y)
        self.x_mid = self.im_shape[0] / 2.0 - 0.5
        self.y_mid = self.im_shape[1] / 2.0 - 0.5

        # Set up pre-FFTed PSF
        f1d1 = jnp.fft.rfftfreq(self.im_shape[0])
        f1d2 = jnp.fft.fftfreq(self.im_shape[1])
        self.FX, self.FY = jnp.meshgrid(f1d1, f1d2)
        self.fft_shape = self.FX.shape
        fft_shift_arr_x = jnp.exp(
            jax.lax.complex(0.0, -1.0)
            * 2.0
            * 3.1415
            * -1
            * (self.psf_shape[0] / 2.0 - 0.5)
            * self.FX
        )
        fft_shift_arr_y = jnp.exp(
            jax.lax.complex(0.0, -1.0)
            * 2.0
            * 3.1415
            * -1
            * (self.psf_shape[1] / 2.0 - 0.5)
            * self.FY
        )
        self.PSF_fft = (
            jnp.fft.rfft2(self.pixel_PSF, s=self.im_shape)
            * fft_shift_arr_x
            * fft_shift_arr_y
        )

        # All the  renderers here these profile types
        self.profile_func_dict = {}
        for profile_type in base_profile_types:
            self.profile_func_dict[profile_type] = getattr(
                self, f"render_{profile_type}"
            )

        self.fft_zeros = jnp.zeros(self.fft_shape)
        self.img_zeros = jnp.zeros(self.im_shape)

    def conv_img_and_fft(self, image,F_im):
        img_fft = jnp.fft.rfft2(image)
        conv_fft = (img_fft+F_im) * self.PSF_fft
        conv_im = jnp.fft.irfft2(conv_fft, s=self.im_shape)
        return conv_im

    def conv_fft(self, F_im):
        im = jnp.fft.irfft2(F_im * self.PSF_fft, s=self.im_shape)
        return im

    def combine_scene(self, F_im, int_im, obs_im):
        """Default implementation that naively combines all types of images even if they are known to be zero

        Parameters
        ----------
        F_im : Fourier image
            Sum of sources rendered in Fourier space
        int_im : _type_
            Sum of sources rendered in Intrinsic space
        obs_im : _type_
            Sum of sources rendered in observed space

        Returns
        -------
        Model image
            Combination of all sources to be compared to observations
        """
        return self.conv_img_and_fft(int_im,F_im) + obs_im

    @abstractmethod
    def render_sersic(self, params: dict):
        return NotImplementedError

    def render_doublesersic(self, params: dict):
        dict_1 = {
            "xc": params["xc"],
            "yc": params["yc"],
            "flux": params["flux"] * params["f_1"],
            "n": params["n_1"],
            "ellip": params["ellip_1"],
            "theta": params["theta"],
            "r_eff": params["r_eff_1"],
        }
        dict_2 = {
            "xc": params["xc"],
            "yc": params["yc"],
            "flux": params["flux"] * (1.0 - params["f_1"]),
            "n": params["n_2"],
            "ellip": params["ellip_2"],
            "theta": params["theta"],
            "r_eff": params["r_eff_2"],
        }
        F1, im_int_1, im_obs_1 = self.render_sersic(dict_1)
        F2, im_int_2, im_obs_2 = self.render_sersic(dict_2)

        return F1 + F2, im_int_1 + im_int_2, im_obs_1 + im_obs_2

    def render_sersic_exp(self, params: dict):
        dict_1 = {
            "xc": params["xc"],
            "yc": params["yc"],
            "flux": params["flux"] * params["f_1"],
            "n": params["n"],
            "ellip": params["ellip_1"],
            "theta": params["theta"],
            "r_eff": params["r_eff_1"],
        }
        dict_2 = {
            "xc": params["xc"],
            "yc": params["yc"],
            "flux": params["flux"] * (1.0 - params["f_1"]),
            "ellip": params["ellip_2"],
            "theta": params["theta"],
            "r_eff": params["r_eff_2"],
        }
        F1, im_int_1, im_obs_1 = self.render_sersic(dict_1)
        F2, im_int_2, im_obs_2 = self.render_exp(dict_2)

        return F1 + F2, im_int_1 + im_int_2, im_obs_1 + im_obs_2

    def render_sersic_pointsource(self, params: dict):
        pointsource_dict = {}
        sersic_dict = params.copy()

        sersic_dict["flux"] = (1.0 - params["f_ps"]) * params["flux"]
        pointsource_dict["flux"] = sersic_dict.pop("f_ps") * params["flux"]
        pointsource_dict["xc"] = sersic_dict["xc"]
        pointsource_dict["yc"] = sersic_dict["yc"]

        F1, im_int_1, im_obs_1 = self.render_sersic(sersic_dict)
        F2, im_int_2, im_obs_2 = self.render_pointsource(pointsource_dict)
        return F1 + F2, im_int_1 + im_int_2, im_obs_1 + im_obs_2

    @abstractmethod
    def render_pointsource(self, params: dict):
        return NotImplementedError

    def render_exp(self, params: dict) -> jax.numpy.array:
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
        to_sersic = dict(params, n=1.0)
        return self.render_sersic(to_sersic)

    def render_dev(self, params: dict) -> jax.numpy.array:
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
        to_sersic = dict(params, n=4.0)
        return self.render_sersic(to_sersic)

    def render_spergel(self, params: dict):
        
        F_im = render_spergel_fourier(
            self.FX, 
            self.FY,
            params['r_eff'], 
            params['flux'], 
            params['nu_star'],
            params['xc'],
            params['yc'], 
            params['theta'], 
            1. - params['ellip']
        )
        return F_im, self.img_zeros, self.img_zeros
    
    def render_for_model(self, param_dict, types, suffix):
        F_tot = jnp.zeros(self.fft_shape)
        int_im_tot = jnp.zeros(self.im_shape)
        obs_im_tot = jnp.zeros(self.im_shape)

        for j, prof_type in enumerate(types):
            new_dict = {
                param: param_dict[param + f"_{j:d}{suffix}"]
                for param in base_profile_params[prof_type]
            }
            F_cur, int_im_cur, obs_im_cur = self.profile_func_dict[prof_type](new_dict)
            F_tot = F_tot + F_cur
            int_im_tot = int_im_tot + int_im_cur
            obs_im_tot = obs_im_tot + obs_im_cur

        return self.combine_scene(F_tot, int_im_tot, obs_im_tot)

    def render_source(
        self, params: dict, profile_type: str, suffix: Optional[str] = ""
    ) -> jax.numpy.array:
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
        to_func = {k.replace(suffix, ""): v for k, v in params.items()}
        model_im = self.combine_scene(*self.profile_func_dict[profile_type](to_func))
        return model_im


class PixelRenderer(BaseRenderer):
    """
    Render class based on rendering in pixel space and then convolving with the PSF
    """

    os_pixel_size: int = eqx.field(static=True)
    num_os: int = eqx.field(static=True)

    w_os: jnp.array
    x_os_lo: int
    x_os_hi: int
    y_os_lo: int
    y_os_hi: int

    X_os: jnp.array
    Y_os: jnp.array

    def __init__(
        self,
        im_shape: Iterable,
        pixel_PSF: jax.numpy.array,
        os_pixel_size: Optional[int] = 6,
        num_os: Optional[int] = 12,
    ) -> None:
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

        # Use Gauss-Legendre coefficents for better integration when oversampling
        dx, w = jnp.array(np.polynomial.legendre.leggauss(self.num_os))
        w = w / 2.0
        dx = dx / 2.0

        # dx = np.linspace(-0.5,0.5, num= num_os,endpoint=True)
        # w = np.ones_like(dx)
        dx_os, dy_os = jnp.meshgrid(dx, dx)

        w1, w2 = jnp.meshgrid(w, w)
        self.w_os = w1 * w2

        i_mid = int(self.im_shape[0] / 2)
        j_mid = int(self.im_shape[1] / 2)

        self.x_os_lo, self.x_os_hi = (
            i_mid - self.os_pixel_size,
            i_mid + self.os_pixel_size,
        )
        self.y_os_lo, self.y_os_hi = (
            j_mid - self.os_pixel_size,
            j_mid + self.os_pixel_size,
        )

        self.X_os = (
            self.X[
                self.x_os_lo : self.x_os_hi,
                self.y_os_lo : self.y_os_hi,
                jnp.newaxis,
                jnp.newaxis,
            ]
            + dx_os
        )
        self.Y_os = (
            self.Y[
                self.x_os_lo : self.x_os_hi,
                self.y_os_lo : self.y_os_hi,
                jnp.newaxis,
                jnp.newaxis,
            ]
            + dy_os
        )

    def render_int_sersic(self, xc, yc, flux, r_eff, n, ellip, theta):
        im_no_os = render_sersic_2d(
            self.X, self.Y, xc, yc, flux, r_eff, n, ellip, theta
        )

        sub_im_os = render_sersic_2d(
            self.X_os, self.Y_os, xc, yc, flux, r_eff, n, ellip, theta
        )
        sub_im_os = jnp.sum(sub_im_os * self.w_os, axis=(2, 3))

        im = im_no_os.at[self.x_os_lo : self.x_os_hi, self.y_os_lo : self.y_os_hi].set(
            sub_im_os
        )
        return im

    def render_sersic(self, params: dict) -> jax.numpy.array:
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
        im_int = self.render_int_sersic(
            params["xc"],
            params["yc"],
            params["flux"],
            params["r_eff"],
            params["n"],
            params["ellip"],
            params["theta"],
        )
        return self.fft_zeros, im_int, 0.

    def render_pointsource(self, params: dict) -> jax.numpy.array:
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
        dx = params["xc"] - self.psf_shape[0] / 2.0
        dy = params["yc"] - self.psf_shape[1] / 2.0

        shifted_psf = jax.scipy.ndimage.map_coordinates(
            self.pixel_PSF * params["flux"],
            [self.X - dx, self.Y - dy],
            order=1,
            mode="constant",
        )

        return self.fft_zeros, 0., shifted_psf
    
    def combine_scene(self, F_im, int_im, obs_im):
        """Combine scene for PixelRenderer when nothing is rendered in Fourier space

        Parameters
        ----------
        F_im : Fourier image
            Sum of sources rendered in Fourier space
        int_im : _type_
            Sum of sources rendered in Intrinsic space
        obs_im : _type_
            Sum of sources rendered in observed space

        Returns
        -------
        Model image
            Combination of all sources to be compared to observations
        """
        return self.conv_img_and_fft(int_im+self.img_zeros, F_im + self.fft_zeros) + obs_im

def sersic_hankel_emul_func(nu_in,n):
    c = [-1.6021528 ,  0.33155227,  0.2694067 ,  0.73505086]
    nu =  nu_in + 1e-8
    numerator = c[0]*jnp.log(nu) + c[1]/nu + c[2]
    return jnp.square( jax.nn.sigmoid(numerator/ jnp.sqrt(n) + c[3]) )

class MoGFourierRenderer(BaseRenderer):
    """
    Class to render sources based on rendering them in Fourier space. Sersic profiles are modeled as a series of Gaussian following Shajib (2019) (https://arxiv.org/abs/1906.08263) and the implementation in lenstronomy (https://github.com/lenstronomy/lenstronomy/blob/main/lenstronomy/LensModel/Profiles/gauss_decomposition.py)
    """

    frac_start: float = eqx.field(static=True)
    frac_end: float = eqx.field(static=True)
    n_sigma: int = eqx.field(static=True)
    precision: int = eqx.field(static=True)
    use_interp_amps: bool = eqx.field(static=True)
    etas: jax.numpy.array
    betas: jax.numpy.array
    n_ax: jax.numpy.array
    amps_n_ax: jax.numpy.array

    def __init__(
        self,
        im_shape: Iterable,
        pixel_PSF: jax.numpy.array,
        frac_start: Optional[float] = 1e-2,
        frac_end: Optional[float] = 15.0,
        n_sigma: Optional[int] = 15,
        precision: Optional[int] = 10,
        use_interp_amps: Optional[bool] = True,
    ) -> None:
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
        use_interp_amps: Optional[bool]
            If True, instead of performing the direct calculation in Shajib (2019) at each iteration, a polynomial approximation is fit and used. The amplitudes of each gaussian component amplitudes as a function of Sersic index are interpolated based on a computed grid. This is much more numerically stable owing to the smooth gradients. If this matters for you then set this to False and make sure to enable jax's 64 bit capabilities which we find helps the stability.
        """
        super().__init__(im_shape, pixel_PSF)
        self.frac_start = frac_start
        self.frac_end = frac_end
        self.n_sigma = n_sigma
        self.precision = precision
        self.use_interp_amps = use_interp_amps
        self.etas, self.betas = calculate_etas_betas(self.precision)
        if not use_interp_amps and not jax.config.x64_enabled:
            warnings.warn(
                "!! WARNING !! - Gaussian decomposition can be numerically unstable when using jax's default 32 bit. Please either enable jax 64 bit or set 'use_interp_amps' = True in the renderer kwargs"
            )

        # Fit polynomial for smooth interpolation
        self.n_ax = jnp.linspace(0.65, 8.0, num=50)
        self.amps_n_ax = jax.vmap(
            lambda n: sersic_gauss_decomp(
                1.0,
                1.0,
                n,
                self.etas,
                self.betas,
                self.frac_start,
                self.frac_end,
                self.n_sigma,
            )[0]
        )(self.n_ax)



    def get_amps_sigmas(self, flux, r_eff, n):
        if self.use_interp_amps:
            amps_norm = interp1d(n, self.n_ax, self.amps_n_ax, method="cubic2")
            amps = amps_norm * flux
            sigmas = jnp.logspace(
                jnp.log10(r_eff * self.frac_start),
                jnp.log10(r_eff * self.frac_end),
                num=self.n_sigma,
            )
        else:
            amps, sigmas = sersic_gauss_decomp(
                flux,
                r_eff,
                n,
                self.etas,
                self.betas,
                self.frac_start * r_eff,
                self.frac_end * r_eff,
                self.n_sigma,
            )

        return amps, sigmas

    def render_sersic_mog_fourier(self, xc, yc, flux, r_eff, n, ellip, theta):
        amps, sigmas = self.get_amps_sigmas(flux, r_eff, n)
        q = 1.0 - ellip
        Fgal = render_gaussian_fourier(self.FX, self.FY, amps, sigmas, xc, yc, theta, q)
        return Fgal
    
    def render_sersic(self, params: dict) -> jax.numpy.array:
        """Render a Sersic profile

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
        F_im = self.render_sersic_mog_fourier(
            params["xc"],
            params["yc"],
            params["flux"],
            params["r_eff"],
            params["n"],
            params["ellip"],
            params["theta"],
        )

        return F_im, 0.,0.

    def render_pointsource(self, params: dict) -> jax.numpy.array:
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
        F_im = render_pointsource_fourier(
            self.FX, self.FY, params["xc"], params["yc"], params["flux"]
        )
        return F_im, 0., 0.
    
    def combine_scene(self, F_im, int_im, obs_im):
        """Combine scene for FourierRenderer where everything is rendered in Fourier space

        Parameters
        ----------
        F_im : Fourier image
            Sum of sources rendered in Fourier space
        int_im : _type_
            Sum of sources rendered in Intrinsic space
        obs_im : _type_
            Sum of sources rendered in observed space

        Returns
        -------
        Model image
            Combination of all sources to be compared to observations
        """
        return self.conv_fft(F_im)

class FourierRenderer(MoGFourierRenderer):
    def __init__(
        self,
        im_shape: Iterable,
        pixel_PSF: jax.numpy.array,
        frac_start: Optional[float] = 1e-2,
        frac_end: Optional[float] = 15.0,
        n_sigma: Optional[int] = 15,
        precision: Optional[int] = 10,
        use_interp_amps: Optional[bool] = True,
    ) -> None:
        super.__init__(
            im_shape=im_shape,
            pixel_PSF=pixel_PSF,
            frac_start=frac_start,
            frac_end=frac_end,
            n_sigma=n_sigma,
            precision=precision,
            use_interp_amps=use_interp_amps,
        )
        warnings.warn("The original 'FourierRenderer' has been rename 'MoGFourierRenderer. Note in future releases this name will be deprecated'",DeprecationWarning)

class EmulatorFourierRenderer(MoGFourierRenderer):
    emul_func: callable = eqx.field(static=True)

    def __init__(
        self,
        im_shape: Iterable,
        pixel_PSF: jax.numpy.array,
        emul_func: Optional[Union[callable,str]] = 'F'
    ) -> None:
        super().__init__(im_shape = im_shape, pixel_PSF= pixel_PSF)

        if isinstance(emul_func, str):
            if emul_func == 'F':
                self.emul_func = F_tilde
            elif emul_func == 'F_A':
                self.emul_func = F_tilde_A
            else:
                raise ValueError("Only 'F' and 'F_A' are pre-computed functions, please see documentiaion")
        elif isinstance(emul_func,callable):
            warnings.warn('You are using a user-specified emulator function, if this is not accurate the results will be unreliable. Be sure you know what you are doing and double check with other methods')
            self.emul_func = emul_func
        else:
            raise TypeError("Argument 'emul_func' must be either a string or callable")

    
    def render_sersic_emul_fourier(self, xc, yc, flux, r_eff, n, ellip, theta):
        theta = theta + (jnp.pi / 2.0)
        Ui = self.FX * jnp.cos(theta) + self.FY* jnp.sin(theta)
        Vi = -1 * self.FX * jnp.sin(theta) + self.FY * jnp.cos(theta)

        k_tilde = jnp.hypot(Ui, Vi*(1.-ellip) ) * r_eff*2*np.pi
        Fgal = self.emul_func(k_tilde, n)
        in_exp = (
            - 1j * 2 * jnp.pi * self.FX * xc
            - 1j * 2 * jnp.pi * self.FY * yc
        )
        return Fgal * jnp.exp(in_exp) * flux
    
    def render_sersic(self, params: dict) -> jax.numpy.array:
        """Render a Sersic profile

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
        F_im = self.render_sersic_emul_fourier(
            params["xc"],
            params["yc"],
            params["flux"],
            params["r_eff"],
            params["n"],
            params["ellip"],
            params["theta"],
        )

        return F_im, 0.,0.

class HybridRenderer(MoGFourierRenderer):
    """
    Class to render sources based on the hybrid rendering scheme introduced in Lang (2020). This avoids some of the artifacts introduced by rendering sources purely in Fourier space. Sersic profiles are modeled as a series of Gaussian following Shajib (2019) (https://arxiv.org/abs/1906.08263) and the implementation in lenstronomy (https://github.com/lenstronomy/lenstronomy/blob/main/lenstronomy/LensModel/Profiles/gauss_decomposition.py).

    Our scheme is implemented slightly differently than Lang (2020), specifically in how it chooses which gaussian components to render in Fourier vs. Real space. Lang (2020) employs a cutoff based on distance to the edge of the image. However given some of jax's limitation with dynamic shapes (see more here -> https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#dynamic-shapes), we have not implemented that specific criterion. Instead we use a simpler version where the user must decide how many components to render in real space, starting from the largest ones. While this is not ideal in all circumstances it still overcomes many of the issues of rendering purely in fourier space discussed in Lang (2020).
    """

    num_pixel_render: int = eqx.field(static=True)
    w_real: jax.numpy.array = eqx.field(static=True)
    w_fourier: jax.numpy.array = eqx.field(static=True)
    sig_psf_approx: float = eqx.field(static=True)

    def __init__(
        self,
        im_shape: Iterable,
        pixel_PSF: jax.numpy.array,
        frac_start: Optional[float] = 1e-2,
        frac_end: Optional[float] = 15.0,
        n_sigma: Optional[int] = 15,
        num_pixel_render: Optional[int] = 3,
        precision: Optional[int] = 10,
        use_interp_amps: Optional[bool] = True,
    ) -> None:
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
        use_interp_amps: Optional[bool]
            If True, instead of performing the direct calculation in Shajib (2019) at each iteration, a polynomial approximation is fit and used. The amplitudes of each gaussian component amplitudes as a function of Sersic index are interpolated based on a computed grid. This is much more numerically stable owing to the smooth gradients. If this matters for you then set this to False and make sure to enable jax's 64 bit capabilities which we find helps the stability.
        """
        super().__init__(
            im_shape,
            pixel_PSF,
            frac_start,
            frac_end,
            n_sigma,
            precision,
            use_interp_amps,
        )

        self.num_pixel_render = num_pixel_render
        self.w_real = jnp.arange(
            self.n_sigma - self.num_pixel_render, self.n_sigma, dtype=jnp.int32
        )
        self.w_fourier = jnp.arange(
            self.n_sigma - self.num_pixel_render, dtype=jnp.int32
        )

        psf_X, psf_Y = jnp.meshgrid(
            jnp.arange(self.psf_shape[0]), jnp.arange(self.psf_shape[1])
        )
        sig_x = jnp.sqrt(
            (self.pixel_PSF * (psf_X - psf_X.mean()) ** 2).sum() / self.pixel_PSF.sum()
        )
        sig_y = jnp.sqrt(
            (self.pixel_PSF * (psf_Y - psf_Y.mean()) ** 2).sum() / self.pixel_PSF.sum()
        )
        self.sig_psf_approx = 0.5 * (sig_x + sig_y)

    def render_sersic_hybrid(self, xc, yc, flux, r_eff, n, ellip, theta):
        amps, sigmas = self.get_amps_sigmas(flux, r_eff, n)

        q = 1.0 - ellip

        sigmas_obs = jnp.sqrt(sigmas**2 + self.sig_psf_approx**2)
        q_obs = jnp.sqrt((q * q * sigmas**2 + self.sig_psf_approx**2) / sigmas_obs**2)

        Fgal = render_gaussian_fourier(
            self.FX,
            self.FY,
            amps[self.w_fourier],
            sigmas[self.w_fourier],
            xc,
            yc,
            theta,
            q,
        )

        im_gal = render_gaussian_pixel(
            self.X,
            self.Y,
            amps[self.w_real],
            sigmas_obs[self.w_real],
            xc,
            yc,
            theta,
            q_obs[self.w_real],
        )

        return Fgal, im_gal

    def render_sersic(self, params: dict) -> jax.numpy.array:
        """Render a Sersic profile

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
        F, im = self.render_sersic_hybrid(
            params["xc"],
            params["yc"],
            params["flux"],
            params["r_eff"],
            params["n"],
            params["ellip"],
            params["theta"],
        )
        return F, 0., im

    def render_pointsource(self, params: dict) -> jax.numpy.array:
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
        F_im = render_pointsource_fourier(
            self.FX, self.FY, params["xc"], params["yc"], params["flux"]
        )
        return F_im, 0., 0.

    def combine_scene(self, F_im, int_im, obs_im):
        """Combine scene for FourierRenderer where nothing is rendered in Intrinsic space

        Parameters
        ----------
        F_im : Fourier image
            Sum of sources rendered in Fourier space
        int_im : _type_
            Sum of sources rendered in Intrinsic space
        obs_im : _type_
            Sum of sources rendered in observed space

        Returns
        -------
        Model image
            Combination of all sources to be compared to observations
        """
        return self.conv_fft(F_im) + obs_im

def sersic1D(
    r: Union[float, jax.numpy.array], flux: float, re: float, n: float
) -> Union[float, jax.numpy.array]:
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
    bn = 1.9992 * n - 0.3271
    Ie = (
        flux
        / (re * re * 2 * jnp.pi * n * jnp.exp(bn + jax.scipy.special.gammaln(2 * n)))
        * bn ** (2 * n)
    )
    return Ie * jnp.exp(-bn * ((r / re) ** (1.0 / n) - 1.0))

def render_spergel_fourier(
    FX: jax.numpy.array,
    FY: jax.numpy.array,
    r_eff: float,
    flux: float,
    nu_star: float,
    xc: float,
    yc: float,
    theta: float,
    q: float,
) -> jax.numpy.array:
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
    theta = theta + (jnp.pi / 2.0)
    Ui = FX * jnp.cos(theta) + FY * jnp.sin(theta)
    Vi = -1 * FX * jnp.sin(theta) + FY * jnp.cos(theta) 
    nu = deterministic('nu', 1./nu_star - 1.)

    in_exp = - 1j * 2 * jnp.pi * FX * xc - 1j * 2 * jnp.pi * FY * yc
    
    r_maj = 2*np.pi*r_eff
    r_min = 2*np.pi*r_eff*q
    c_nu = c_nu_approx(nu)
    Fgal = jnp.exp(in_exp)* flux * jnp.power(1. + (r_maj**2 * Ui**2 + r_min**2 * Vi**2)/c_nu**2  , -(1.+nu) )
    return Fgal

def render_gaussian_fourier(
    FX: jax.numpy.array,
    FY: jax.numpy.array,
    amps: jax.numpy.array,
    sigmas: jax.numpy.array,
    xc: float,
    yc: float,
    theta: float,
    q: float,
) -> jax.numpy.array:
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
    theta = theta + (jnp.pi / 2.0)
    Ui = FX * jnp.cos(theta) + FY * jnp.sin(theta)
    Vi = -1 * FX * jnp.sin(theta) + FY * jnp.cos(theta)

    in_exp = (
        -1
        * (Ui * Ui + Vi * Vi * q * q)
        * (2 * jnp.pi * jnp.pi * sigmas * sigmas)[:, jnp.newaxis, jnp.newaxis]
        - 1j * 2 * jnp.pi * FX * xc
        - 1j * 2 * jnp.pi * FY * yc
    )
    Fgal_comp = amps[:, jnp.newaxis, jnp.newaxis] * jnp.exp(in_exp)
    Fgal = jnp.sum(Fgal_comp, axis=0)
    return Fgal


def render_pointsource_fourier(
    FX: jax.numpy.array, FY: jax.numpy.array, xc: float, yc: float, flux: float
) -> jax.numpy.array:
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
    in_exp = -1j * 2 * jnp.pi * FX * xc - 1j * 2 * jnp.pi * FY * yc
    F_im = flux * jnp.exp(in_exp)
    return F_im


def render_gaussian_pixel(
    X: jax.numpy.array,
    Y: jax.numpy.array,
    amps: jax.numpy.array,
    sigmas: jax.numpy.array,
    xc: float,
    yc: float,
    theta: float,
    q: Union[float, jax.numpy.array],
) -> jax.numpy.array:
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
    theta = theta + (jnp.pi / 2.0)
    Xi = X_bar * jnp.cos(theta) + Y_bar * jnp.sin(theta)
    Yi = -1 * X_bar * jnp.sin(theta) + Y_bar * jnp.cos(theta)

    in_exp = (
        -1
        * (Xi * Xi + Yi * Yi / (q * q)[:, jnp.newaxis, jnp.newaxis])
        / (2 * sigmas * sigmas)[:, jnp.newaxis, jnp.newaxis]
    )
    im_comp = (amps / (2 * jnp.pi * sigmas * sigmas * q))[
        :, jnp.newaxis, jnp.newaxis
    ] * jnp.exp(in_exp)
    im = jnp.sum(im_comp, axis=0)
    return im


def render_sersic_2d(
    X: jax.numpy.array,
    Y: jax.numpy.array,
    xc: float,
    yc: float,
    flux: float,
    r_eff: float,
    n: float,
    ellip: float,
    theta: float,
) -> jax.numpy.array:
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
        Position angle in radians [now measured from north]

    Returns
    -------
    jax.numpy.array
        Sersic model evaluated at given locations
    """
    bn = 1.9992 * n - 0.3271
    a, b = r_eff, (1 - ellip) * r_eff
    theta = theta + (jnp.pi / 2.0)
    cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)
    x_maj = (X - xc) * cos_theta + (Y - yc) * sin_theta
    x_min = -(X - xc) * sin_theta + (Y - yc) * cos_theta
    amplitude = (
        flux
        * bn ** (2 * n)
        / (jnp.exp(bn + jax.scipy.special.gammaln(2 * n)) * r_eff**2 * jnp.pi * 2 * n)
    )
    z = jnp.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)
    out = amplitude * jnp.exp(-bn * (z ** (1 / n) - 1)) / (1.0 - ellip)
    return out

def c_nu_approx(nu):
    params = {'a': 0.80882865,
    'b': 0.18703458,
    'd': 1.1386762,
    'e': 1.4785621}
    return params['a'] + params['b']*nu + params['d']*jnp.log(params['e']+nu)

def calculate_etas_betas(precision: int) -> Tuple[jax.numpy.array, jax.numpy.array]:
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
    betas = jnp.sqrt(2 * precision * jnp.log(10) / 3.0 + 2.0 * 1j * jnp.pi * kes)
    epsilons = jnp.zeros(2 * precision + 1)

    epsilons = epsilons.at[0].set(0.5)
    epsilons = epsilons.at[1 : precision + 1].set(1.0)
    epsilons = epsilons.at[-1].set(1 / 2.0**precision)

    for k in range(1, precision):
        epsilons = epsilons.at[2 * precision - k].set(
            epsilons[2 * precision - k + 1] + 1 / 2.0**precision * comb(precision, k)
        )

    etas = jnp.array(
        (-1.0) ** kes
        * epsilons
        * 10.0 ** (precision / 3.0)
        * 2.0
        * jnp.sqrt(2 * jnp.pi)
    )
    betas = jnp.array(betas)
    return etas, betas


def sersic_gauss_decomp(
    flux: float,
    re: float,
    n: float,
    etas: jax.numpy.array,
    betas: jax.numpy.array,
    sigma_start: float,
    sigma_end: float,
    n_comp: int,
) -> Tuple[jax.numpy.array, jax.numpy.array]:
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
    sigmas = jnp.logspace(jnp.log10(sigma_start), jnp.log10(sigma_end), num=n_comp)

    f_sigmas = jnp.sum(
        etas * sersic1D(jnp.outer(sigmas, betas), flux, re, n).real, axis=1
    )

    del_log_sigma = jnp.abs(jnp.diff(jnp.log(sigmas)).mean())

    amps = f_sigmas * del_log_sigma / jnp.sqrt(2 * jnp.pi)

    amps = amps.at[0].multiply(0.5)
    amps = amps.at[-1].multiply(0.5)

    amps = amps * 2 * jnp.pi * sigmas * sigmas

    return amps, sigmas

def G(k_in, n):
    k = k_in + 1e-4
    a = jnp.array(
        [
            2.27361328549901,
            0.0795856,
            0.054102138,
            0.13979608,
            0.10258421077129,
            0.925636,
            0.439828534772359,
            1.4859663,
            0.015870415,
            0.00511146791581249,
            0.745477235501201,
        ]
    )
    sqrt_n = jnp.sqrt(n)
    h = (
        -a[6] * sqrt_n
        + (k - a[7]) / (n - a[8])
        + a[9] * jnp.square(a[10] * jnp.sqrt(k) * n - 1.0)
    )
    return (
        a[0]
        / sqrt_n
        * (jnp.log(k) - a[1] / n - a[2] / k - a[3] + a[4] / (k + a[5]) * jnp.square(h))
    )


def F_tilde(k, n):
    return 1.0 / (1.0 + jnp.exp(G(k, n)))


def G_A(k_in, n):
    k = k_in + 1e-4
    a = jnp.array([-0.105035484, 2.46661648594762, 0.3465032])
    return a[0] + a[1] * (-a[2] * k / (k + n) + jnp.log(k)) / jnp.sqrt(n)


def F_tilde_A(k, n):
    return 1./(1. + jnp.exp( G_A(k,n) ) )