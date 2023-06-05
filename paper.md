---
title: 'pysersic: A Python package for determining galaxy structural properties via Bayesian inference, accelerated with jax'
tags:
  - Python
  - astronomy
  - galaxies
  - model fitting
authors:
  - name: Imad Pasha
    orcid: 0000-0002-7075-9931
    equal-contrib: true
    affiliation: "1, 2"
  - name: Tim B. Miller
    orcid: 0000-0001-8367-6265
    equal-contrib: true
    affiliation: 1
affiliations:
 - name: Department of Astronomy, Yale University, USA
   index: 1
 - name: National Science Foundation Graduate Research Fellow
   index: 2
date: 5 June 2023
bibliography: paper.bib

---

# Summary

The modern standard for measuring structural parameters of galaxies involves a forward-modeling procedure in which parametric models are fit directly to images while accounting for the effect of the point-spread function (PSF). This is an integral step in many analyses. The most common parametric form is a Sérsic profile [@Sersic:1968], described by a radial profile following


$$
I(R) ∝ F_{\rm total} \exp \left[\left(\frac{R}{R_e}\right)^{1/n}-1\right],
$$


where the total flux, $F_{\rm total}$, half-light radius, $R_e$ and Sérsic index, $n$ are the parameters of interest to be fit and subsequently used to characterize a galaxy's morphology.


Here we present `pysersic`, a Bayesian framework created to facilitate the inference of structural parameters from galaxy images. It is written in pure `Python`, and built using the `jax` framework [@Bradbury:2018] allowing for just-in-time (JIT) compilation, auto-differentiation and seamless execution on CPUs, GPUs or TPUs. Inference is performed with the `numpyro` [@Phan:2019;@Bingham:2019] package utilizing gradient based methods, e.g., No U-Turn Sampling (NUTS) [@Hoffman:2014], for efficient and robust posterior estimation. `pysersic` was designed to have a user-friendly interface, allowing users to fit single or multiple sources in a few lines of code. It was also designed to scale to many images, such that it can be seamlessly integrated into current and future analysis pipelines.

# Statement of need

Parametric profile fitting has become a ubiquitous and essential tool for numerous applications including measuring the photometry --- or total flux --- of galaxies, as well as the investigation of the structural evolution of galaxies over cosmic time [@Lange:2015;@Mowla:2019;@Kawinwanichakij:2021]. This approach allows one to both extrapolate galaxy surface brightness profiles beyond the noise limit of images, as well as account for the PSF to accurately measure the structure of galaxies near the resolution limit of those images. The empirically derived Sérsic profile is the most common parametric form for the surface-brightness profile as it provides a reasonable approximation to nearly all galaxies, given the additional freedom of the Sérsic index, $n$, over fixed-index profiles.


Given the long history of Sérsic fitting codes with many available tools, the development of `pysersic` was largely motivated by two related factors, first and foremost of which was the desire to implement Sérsic fitting in a fully Bayesian context *at speed*. The *ability* to place the typical Sérsic fitting problem into a Bayesian context with runtimes that are not prohibitive (the traditional drawback of MCMC methods) has recently been unlocked by the second motivation: to leverage the `jax` library. `jax` utilizes JIT compilation to decrease computational runtimes, provides seamless integration with hardware accelerators such as GPUs and TPUs for further improvements in performance, and enables automatic differentiation, facilitating gradient based optimization and sampling methods. Together, these features greatly increase speed and efficiency, especially when sampling or optimizing a large number of parameters.

Inference in `pysersic` is implemented using the `numpyro` probabilistic programming language (PPL). This allows for total control over the priors and sampling methods used while sampling. The `numpyro` package utilizes `jax`'s auto-differentiation capabilities for gradient based samplers such as Hamiltonian Monte Carlo (HMC) and No-U-Turn-Samplin (NUTS). In addition, there are recently-developed techniques for posterior estimation, including variational inference [@Ranganath:2014] utilizing normalizing flows [@DeCao:2020]. These techniques dramatically reduce the number of likelihood calls required to provide accurate estimates of the posterior relative to gradient-free methods. Combined with the `jax`'s JIT compilation, posteriors can now be generated in a few minutes or less on modern laptops.

# Code Description

`pysersic` was designed to have a user-friendly API with sensible defaults. Tools are provided to automatically generate priors for all free parameters based on an initial characterization of a given image --- but can also easily be set manually. We provide default inference routines for NUTS MCMC and variational inference using neural flows. Users can access the underlying `numpyro` model if desired, to perform inference using any tools available within the `numpyro` ecosystem. The goal for `pysersic` is to provide a reasonable defaults for new users interested in a handful of galaxies, yet maintain the ability for advanced users to tweak options as necessary to perform inference for entire surveys.

A crucial component of any Sérsic fitting code is a efficient and accurate rendering algorithm. Sérsic profiles with high index, $n\gtrsim 3$ are notoriously difficult to render accurately given the steep increase in brightness as $r \rightarrow 0$ In `pysersic`, the `rendering` module is kept separate from the frontend API and inference modules, such that different algorithms can be interchanged and therefore easily tested (and hopefully encourage innovation as well). In this initial release, we provide three algorithms. The first is a traditional rendering algorithm in which the intrinsic profile is rendered in real space, with oversampling in the center to ensure accurate results for high index profiles. The second and third  methods render the profiles in Fourier space, providing accurate results even for strongly peaked profiles and avoiding artifacts due to pixelization. In `pysersic`, this is achieved by representing the profiles using a series of Gaussian following the algorithm presented in @Shajib:2019. We include one algorithm that is fully based in Fourier space, along with a version of the hybrid-real Fourier algorithm introduced in @Lang:2020 which helps avoid some of the aliasing present when rendering solely in Fourier space.


# Related Software


There is a long history and many software tools designed for Sérsic profile fitting. Some of the most popular libraries are listed below.

- `galfit` [@Peng:2002]
- `imfit` [@Erwin:2015]
- `profit` [@Robotham:2017]
- `galight` [@Ding:2021], which is built on top of `lenstronomy` [@Birrer:2021]
- `PetroFit` [@Geda:2022]
- `PyAutoGalaxy` [@Nightingale:2023]

# Software Citations

`pysersic` makes use of the following packages:

- arviz [@arviz:2019]
- asdf [@asdf]
- astropy [@astropy:2013;@astropy:2018;@astropy:2022]
- corner [@corner]
- jax [@Bradbury:2018] 
- matplotlib [@Hunter:2007]
- numpy [@Harris:2020]
- numpyro [@Phan:2019;@Bingham:2019]
- pandas [@reback:2020]
- photutils [@Bradley:2022]
- pytest [@pytest]
- scipy [@Scipy:2020]
- tqdm [@tqdm]


# Acknowledgements

We acknowledge Pieter van Dokkum for useful conversations surrounding the design and implementation of `pysersic`.

# References
