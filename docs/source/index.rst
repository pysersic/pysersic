.. pysersic documentation master file, created by
   sphinx-quickstart on Wed May 17 11:32:50 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
====================================
Pysersic
====================================

-----------------------------------------------------------------------------------------------------------
 A Python package for determining galaxy structural properties via Bayesian inference, accelerated with jax
-----------------------------------------------------------------------------------------------------------

.. image:: https://img.shields.io/github/actions/workflow/status/pysersic/pysersic/pytest.yml

.. image:: https://readthedocs.org/projects/pysersic/badge/?version=latest

.. image:: https://img.shields.io/github/v/release/pysersic/pysersic

.. image:: https://img.shields.io/pypi/v/pysersic?color=blue

.. image:: https://img.shields.io/badge/license-MIT-blue

pysersic is a Python package for fitting Sersic (an other) profiles to astronomical images using Bayesian inference. It is built using the ``jax`` framework with inference performed using the ``numpyro`` probabilistic programming library.

The code is hosted on `GitHub <https://github.com/pysersic/pysersic>`_ and is available open source under the MIT license. We will soon be submitting a paper describing pysersic, for now please cite Pasha & Miller in prep. and reference the github URL.

Statement of Need
=================
The empirically derived Sérsic profile is the most common parametric form for the surface-brightness profile as it provides a reasonable approximation to nearly all galaxies, given the additional freedom of the Sérsic index over fixed-index profiles. Given the long history of Sérsic fitting codes with many available tools, the development of ``pysersic`` was largely motivated by two related factors, first and foremost of which was the desire to implement Sérsic fitting in a fully Bayesian context at speed. The ability to place the typical Sérsic fitting problem into a Bayesian context with runtimes that are not prohibitive (the traditional drawback of MCMC methods) has recently been unlocked by the second motivation: to leverage the ``jax`` library. ``jax`` utilizes just-in-time compilation to decrease computational runtimes, provides seamless integration with hardware accelerators such as graphics processing units (GPUs) for further improvements in performance, and enables automatic differentiation, facilitating gradient based optimization and sampling methods. Together, these features greatly increase speed and efficiency, especially when sampling or optimizing a large number of parameters.



.. toctree::
   :maxdepth: 1
   :caption: Guide:

   install
   issues
   rendering
   inference

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   example-fit
   multi-source-fitting
   manual-priors

.. toctree::
   :maxdepth: 1
   :caption: API info:
   
   API-Summary

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
