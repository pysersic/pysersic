Rendering
==========

The foundation of any Sersic fitting method is an efficient and accurate way to render the model images. In pysersic this performed using the ``rendering.py`` module. This is kept seperate from the inference module to facilitate easy testing of different implementations and to hopefully encourage innovation. With the initial release we have implemented three rendering methods which will be described below: ``PixelRenderer``, ``FourierRenderer``, and ``HybridRenderer``. All of these are implemented in classes which are based on the ``BaseRenderer`` class, upon which new renderers can be built. These classes are all initialized with the desired image shape and PSF cutout. Each algorithm has additional optional arguments described below. 

We set ``HybridRenderer`` as the default as it provides accurate results for the center of sources, where most of the signal is, while avoiding the aliasing issues of ``FourierRenderer``. However it is not as accurate at very large radii (at greater that 10 time the effective radius) and at low sersic index, n $<$ 0.8. In these cases it is better to use ``PixelRenderer``.

``PixelRenderer``
------------------

The first rendering method is the most straightforward and is similar to many other Sersic fitting codes. It involves rendering the profile on the pixel grid "above the sky", i.e. the intrinsic profile. This intrinsic image is then convolved with the PSF using a FFT to create the model image to be compared with the data. When using this approach, we have to be careful to ensure accurate rendering for high index profiles. Near the center of such a profile, the brightness of the profile changes so quickly that the center of a pixel is no longer an accurate approximation for the average flux, as is implicitly assumed when rendering on a pixel grid. As is common, we implement an over-sampling scheme to better predict the flux for these profiles. To conserve computational resources oversampling is performed in a square grid with a fixed size and location in the center of the image where these effects are most prevalent.  


There are two optional arguments:

* ``os_pixel_size`` - The number of pixels in each direction of the center to perform oversampling. So the square is 2x ``os_pixel_size`` by 2x ``os_pixel_size``.

* ``num_os`` - Number of sub-pixels in each direction to oversample by.

``FourierRenderer`` and ``HybridRenderer``
-------------------------------------------

An increasingly common method for rendering profiles is to instead render the profile in Fourier space. This offers several benefits, including faster convolution, only involving a single inverse FFT, and accurate rendering of the centers of profiles without the need for over sampling. In our implementation, it has the additional benefit of providing accurate results for all sources, not just those at the center.

An immediate problem is that the Sersic profile has a pretty nasty Fourier transform. Instead, following `Hogg & Lang (2013) <https://arxiv.org/abs/1210.6563>`_ we implement a method to model a Sersic profile using a series of Gaussians. Gaussians are much better-behaved numerically, and the Fourier transform is simply a Gaussian. We implement a recent innovation presented in `Shajib (2019) <https://arxiv.org/abs/1906.08263>`_ which presents an analytic method for deriving a mixture of Gaussian approximation to any given profile. We base our implementation off of the implementation of this algorithm in ``lenstronomy``.

However, we noticed that this process leads to some to several numerical instabilities, specifically when tracing the gradient through the Shajib et al. formalism. This causes issues with convergence during inference. This may be due to ``jax``'s default 32 bit implementation. Adjusting the precision value or enabling the 64 bit implementation appears to help alleviate these issues, but these solutions have not been fully vetted. We noticed, however, that the amplitudes of each Gaussian component do vary smoothly with Sersic index, and can thus be easily re-scaled to any flux and radius. Now, ``pySersic`` pre-calculates a set of amplitudes at a grid of Sersic indices and then fits a polynomial as a function of log(n). Empirically this leads to a better fit than as a function of n. These polynomials are then used to decompose a profile for a given n during inference ensuring smooth gradients at only a minor cost in accuracy (less than roughly 1\%). This method is enabled by default but can easily be turned off, going back to the direct algorithm, using the argument ``use_poly_fit_amps``.

Both ``FourierRenderer`` and ``HybridRenderer`` use this Gaussian decomposition and render (at least some) of these components in Fourier space. Both have additional optional arguments:

* ``frac_start`` - fraction of the effective radius for the width of the smallest Gaussian component
* ``frac_end`` -  fraction of the effective radius for the width of the largest Gaussian component
* ``n_sigma`` - number of Gaussian components to use
* ``precision`` - Precision value to use in the decomposition described in `Shajib (2019) <https://arxiv.org/abs/1906.08263>`_
* ``use_poly_fit_amps`` - Whether to use a polynomial fit to the amplitudes as a function of n, see above for details.

``FourierRenderer``, as the name implies, renders sources solely in Fourier space. However this can lead to some artifacts, specifically, aliasing if the source is near the edge. This is because the inverse FFT assumes the image is periodic so part of the source that should lie outside the image appears opposite. To help combat this we also implement a version of the hybrid real-Fourier algorithm described in `Lang (2020) <https://arxiv.org/abs/2012.15797>`_ in ``HybridRenderer``. The innovation is to render some of the largest Gaussian components in real space to help avoid the aliasing while maintaining the benefits of rendering in Fourier space. This has one additional argument beyond those described above:

* ``num_pixel_render`` - Number of Gaussian components to render in real space, beginning with the largest and counting backwards.
