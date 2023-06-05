.. pysersic documentation master file, created by
   sphinx-quickstart on Wed May 17 11:32:50 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Pysersic: A Python package for determining galaxy structural properties via Bayesian inference, accelerated with jax
====================================

.. | build | image:: https://img.shields.io/github/actions/workflow/status/pysersic/pysersic/pytest.yml

.. |Documentation Status| image:: https://readthedocs.org/projects/pysersic/badge/?version=latest
   :target: http://pysersic.readthedocs.io/?badge=latest

.. |GitHub release| image:: https://img.shields.io/github/v/release/pysersic/pysersic
   :target: https://github.com/pysersic/pysersic/releases/

.. |PyPI pyversions| image:: https://img.shields.io/pypi/v/pysersic?color=blue
   :target: https://pypi.python.org/pypi/pysersic/

.. |GitHub license| image:: https://img.shields.io/badge/license-MIT-blue
   :target: https://github.com/pysersic/pysersic/blob/main/LICENSE

pysersic is a Python package for fitting Sersic (an other) profiles to astronomical images using Bayesian inference. It is built using the ``jax`` framework with inference performed using the ``numpyro`` probabilistic programming library.

The code is hosted on `GitHub <https://github.com/pysersic/pysersic>`_ and is available open source under the MIT license. We will soon be submitting a paper describing pysersic, for now please cite Pasha & Miller in prep. and reference the github URL.

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
