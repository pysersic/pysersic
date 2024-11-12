Tips and Tricks 
===============

Based on common questions and usage questions we receive, we have a few tips and tricks or guidance that may help you get ``pysersic`` working to the place you need it. 


Image Cutout Size 
-----------------

It is important to have (roughly) the right sized image cutout to ensure a converged fit, especially if your sky level is not already 0 and flat (i.e., you are trying to fit the sky).
Above a certain size, the fitting will start slowing down due to creating models with many pixels, but too small, and your sky estimate will be contaminated by light from the galaxy. 
A "safe" cutout size is in the range of ~10x the half light radius of the galaxy. How do you estimate this without having yet measured the effective radius?
A good rule of thumb is that the "optical extent" of a galaxy that you can see in survey imaging (something like the 26th mag isophote) is very roughly at 3 times the effective radius. 
So based on that, if you make your cutouts roughly 3-4x larger than the visible galaxy (assuming standard survey imaging and the use of semi major axis), you should have enough sky to well constrain both the galaxy and sky while not having too huge of a cutout.

How Important is Masking?
-------------------------

When fitting galaxies, the presence of objects in the field can in principle impact your fit. This is especially a problem when trying to batch fit very large samples, for which manual/visual inspection of inputs and outputs is not as feasible.
If there are sources in your cutout that are very far from the center, and are relatively small, they will likely not affect the fit much, given the priors which mostly lock the fit to the object in the center of the cutout. 

On the other hand, objects quite close to the target galaxy are a larger concern. But in this case, it may be more beneficial to simply fit all the sources in the cutout using ``FitMulti`` (or at least those whose light may overlap the target).

Also, note that the initial sky estimate comes from pixels around the edge of the cutout, so bright objects that "overlap the edge" may influence the starting sky guess, and may be worth masking. 

Which renderer should I use?
----------------------------

By default, ``pysersic`` will use a ``HybridRenderer`` that mixes fourier based rendering and real-space rendering.
For most ground-based imaging of galaxies with sizes bigger than the PSF, this should work fine. On rare occasions, the model for very elliptical (e.g., edge on) distributions displays some "fringing" like effects emanating from the center. This usually doesn't affect the model parameters being trustworthy. 

When dealing with higher Sérsic indices, or galaxies that are quite smal in pixel space (or both), the light becomes very concentrated in the center, and oversampling those pixels is necessary to ensure reasonable modeling (this is actually a known issue with the ``astropy`` Sersic2D modeling as well).
In this case, we recommend importing and passing the ``PixelRenderer`` to the fitter, which has an implementation for oversampling the central region of generated models to ensure the shape of the profile is faithfully represented. 

