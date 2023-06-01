Inference
=========

Inference in pysersic is performed using the `numpyro package <https://github.com/pyro-ppl/numpyro>`_ . Priors are expressed as numpyro distributions, see the example notebook for more details. We have a set of default inference methods, that are applied to both ``FitSingle`` and ``FitMulti``, which are described below. They roughly increase in both accuracy and computational time.

Maximium a posteriori or finding the 'best fit'
------------------------------------------------

The quickest method is to find the best fit values using ``.find_MAP()``. This produces a point estimate that corresponds to the maximum a-posteriori or the "best fit" parameter values.  This can be useful for certain applications but the real power of ``pysersic`` lies in it's ability to produce full posterior distributions, using the methods described below.


Laplace Approximation
---------------------

The quickest method to produce a full posterior distribution is to use the Laplace approximation, which is called by using ``.estimate_posterior(method = 'laplace')``. The `Laplace Approximation <https://en.wikipedia.org/wiki/Laplace%27s_approximation>`_ assumes the posterior is Gaussian and is calculated by finding the maximum a-posteriori (MAP) and then calculating the curvature of the posterior at the MAP estimate. This method is inherently limited by the Gaussian approximation. In our testing a Gaussian approximation for the posterior is usually pretty safe however this may not be the case for all data sets. This method usually takes only 5-10 seconds longer than `.find_MAP()` and is a great place to start!


Variational Inference
---------------------

The next set of methods are differing implementations of Variational inference. This is an alternative to traditional sampling methods, like MCMC or nested sampling, which aims to estimate the posterior by choosing a parameterized form and finding the best approximation. For example, assuming that each variable is independent and Gaussian and optimizing the mean and standard deviation for each. In this way the traditional sampling methods are recast as an optimization problems, which are often much quicker but may not be as accurate if the parameterized form is not a good approximation to the posterior. For a nice introductory blog post see `here <https://towardsdatascience.com/bayesian-inference-problem-mcmc-and-variational-inference-25a8aa9bce29>`_, or for a more technical introduction see `here <https://arxiv.org/abs/2108.13083>`_.

We current have implemented two Variational inference methods. The first ``svi-mvn`` which uses a multi-variate normal distribution, optimizing the means and covariance matrix. In practice this is often very similar to the Laplace approximation described above. The second is ``svi-flow`` which utilizes a normalizing flow, specifically a `block neural autoregressive flow (BNAF) <https://arxiv.org/abs/1904.04676>`_. This is a flexible description of the posterior based on a series of transforms of a simple distribution. This method can capture non-Gaussian distributions but is more computationally expensive.


MCMC
----

Good old MCMC but now with gradients! The default MCMC method implemented in ``pysersic`` is the No U-turn Sampler (NUTS) described in `Hoffman & Gelman (2011) <https://arxiv.org/abs/1111.4246>`_. This is a gradient based sampler which greatly improves efficiency, especially for high dimensional datasets, like fitting multiple Sersic sources at once. This is implemented in the ``.sample()`` function. This method is usually the most robust and accurate but may also be the most computationally intensive. The default is to sample two chains with 1000 steps for warmup and 1000 steps for sampling, but these can be adapted easily.


Customization
-------------

The default inference methods we have implemented should work well in most cases but may not be suitable for all. If you are having trouble or seeing strange results feel free to reach out to us. For any experts in Bayesian statistics or numpyro, every fitter instance contains a method called  ``.build_model()`` which returns a callable function specifying the numpyro model that can easily be slotted into any of the inference algorithms implemented in the library. Then it is still possible to take full advantage of the analysis programs we have written by instantiating a ``PySersicResults`` class and using the ``.injest_data()`` function.