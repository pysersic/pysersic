import warnings
from typing import Iterable, Optional

import jax
import jax.numpy as jnp
import jax.scipy.ndimage
from jaxopt import ScipyBoundedMinimize

from .priors import SourceProperties


class FastRenderer:
    def __init__(self, im_shape: Iterable, pixel_PSF: jnp.array, profile_type: str):
        self.im_shape = im_shape
        self.pixel_PSF = pixel_PSF
        self.profile_type = profile_type

        if not jnp.isclose(jnp.sum(self.pixel_PSF), 1.0, 0.1):
            warnings.warn(
                "PSF does not appear to be appropriately normalized; Sum(psf) is more than 0.1 away from 1."
            )
        self.psf_shape = jnp.shape(self.pixel_PSF)
        if jnp.any(self.im_shape < self.psf_shape):
            raise AssertionError(
                "PSF pixel image size must be smaller than science image."
            )

        x = jnp.arange(self.im_shape[0])
        y = jnp.arange(self.im_shape[1])
        self.X, self.Y = jnp.meshgrid(x, y)

        # Set up pre-FFTed PSF
        f1d1 = jnp.fft.rfftfreq(self.im_shape[0])
        f1d2 = jnp.fft.fftfreq(self.im_shape[1])
        self.FX, self.FY = jnp.meshgrid(f1d1, f1d2)
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

        self.fft_zeros = jnp.zeros(self.FX.shape)
        self.img_zeros = jnp.zeros(self.im_shape)

        # Compile the rendering functions
        self._compile_renderer()

    def _compile_renderer(self):
        self.render_source = jax.jit(self._render_source)

    def _render_source(self, params):
        if self.profile_type == "sersic":
            return self._render_sersic(params)
        elif self.profile_type == "pointsource":
            return self._render_pointsource(params)
        else:
            raise ValueError(f"Unsupported profile type: {self.profile_type}")

    def _render_sersic(self, params):
        xc, yc, flux, r_eff, e1, e2, n = params
        r_eff, ellip, phi = self.convert_params(r_eff, e1, e2)
        bn = 1.9992 * n - 0.3271
        a, b = r_eff, (1 - ellip) * r_eff
        phi = phi + (jnp.pi / 2.0)  # This is the theta derived from e1, e2
        cos_phi, sin_phi = jnp.cos(phi), jnp.sin(phi)
        x_maj = (self.X - xc) * cos_phi + (self.Y - yc) * sin_phi
        x_min = -(self.X - xc) * sin_phi + (self.Y - yc) * cos_phi
        amplitude = (
            flux
            * bn ** (2 * n)
            / (
                jnp.exp(bn + jax.scipy.special.gammaln(2 * n))
                * r_eff**2
                * jnp.pi
                * 2
                * n
            )
        )
        z = jnp.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)
        im_int = amplitude * jnp.exp(-bn * (z ** (1 / n) - 1)) / (1.0 - ellip)
        return self.fft_zeros, im_int, self.img_zeros

    def _render_pointsource(self, params):
        xc, yc, flux = params
        dx = xc - self.psf_shape[0] / 2.0
        dy = yc - self.psf_shape[1] / 2.0
        shifted_psf = jax.scipy.ndimage.map_coordinates(
            self.pixel_PSF * flux,
            [self.X - dx, self.Y - dy],
            order=1,
            mode="constant",
        )
        return self.fft_zeros, self.img_zeros, shifted_psf

    def render(self, params):
        return self.render_source(params)

    def conv_img(self, image):
        img_fft = jnp.fft.rfft2(image)
        conv_fft = img_fft * self.PSF_fft
        conv_im = jnp.fft.irfft2(conv_fft, s=self.im_shape)
        return conv_im

    def conv_fft(self, F_im):
        im = jnp.fft.irfft2(F_im * self.PSF_fft, s=self.im_shape)
        return im

    def combine_scene(self, F_im, int_im, obs_im):
        return self.conv_fft(F_im) + self.conv_img(int_im) + obs_im

    @staticmethod
    def convert_params(r, e1, e2):
        rout = jnp.exp(r)
        epnorm = jnp.hypot(e1, e2)
        enorm = 2 / jnp.pi * jnp.arctan(epnorm)
        phi = 0.5 * jnp.arctan2(e1, e2)
        ba = (1 - enorm) / (1 + enorm)
        ellip = jnp.sqrt(1 - ba**2)
        return rout, ellip, phi


class FitSingleMAP:
    def __init__(
        self,
        renderer: FastRenderer,
        data: jnp.array,
        sig: jnp.array,
        psf: jnp.array,
        mask: Optional[jnp.array] = None,
        profile_type="sersic",
    ):
        self.renderer = renderer(
            data.shape,
            pixel_PSF=jnp.array(psf).astype(float),
            profile_type=profile_type,
        )
        self.data = data
        self.sig = sig
        self.mask = self.parse_mask(mask, data)

    @staticmethod
    def parse_mask(mask: Optional[jnp.array], data: jnp.array) -> jnp.array:
        if mask is None:
            return jnp.logical_not(jnp.ones_like(data).astype(jnp.bool_))
        else:
            return jnp.logical_not(jnp.array(mask).astype(jnp.bool_))

    def chi2(self, params) -> float:
        F, int_im, obs_im = self.renderer.render(params)
        model = self.renderer.combine_scene(F, int_im, obs_im)

        # Apply the mask to both the data and model
        masked_residuals = ((self.data - model) * self.mask) / self.sig
        chi2_val = jnp.sum(masked_residuals**2)
        return chi2_val

    def fit(self) -> jnp.array:
        sp = SourceProperties(self.data, mask=jnp.logical_not(self.mask).astype(float))
        xc = sp.xc_guess
        yc = sp.yc_guess
        flux = sp.flux_guess
        r_eff = sp.r_eff_guess
        e1 = 0.3
        e2 = 0.1

        n = 1
        params_init = jnp.array([xc, yc, flux, r_eff, e1, e2, n])
        bounds = (
            jnp.array(
                [
                    xc - 10,
                    yc - 10,
                    0,
                    0,
                    -1,
                    -1,
                    0.5,
                ]
            ),
            jnp.array(
                [
                    xc + 10,
                    yc + 10,
                    5 * flux,
                    5 * r_eff,
                    1,
                    1,
                    7,
                ]
            ),
        )
        solver = ScipyBoundedMinimize(fun=jax.jit(self.chi2), method="L-BFGS-B")
        result = solver.run(params_init, bounds=bounds)
        return result.params

    def create_model(self, params: jnp.array) -> jnp.array:
        F, int_im, obs_im = self.renderer.render(params)
        model = self.renderer.combine_scene(F, int_im, obs_im)
        return model
