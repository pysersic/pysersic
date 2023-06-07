![pysersic logo featuring a spiral s-galaxy as the S](misc/pysersic.png)
[![status](https://joss.theoj.org/papers/4214c6e588774490458e34630e8052c1/status.svg)](https://joss.theoj.org/papers/4214c6e588774490458e34630e8052c1)
[![PyPi version](https://img.shields.io/pypi/v/pysersic?color=blue)](https://pypi.org/project/pysersic)
[![GitHub release](https://img.shields.io/github/v/release/pysersic/pysersic)](https://github.com/pysersic/pysersic/releases/)
[![Documentation Status](https://readthedocs.org/projects/pysersic/badge/?version=latest)](https://pysersic.readthedocs.io/en/latest/?badge=latest)
![tests](https://github.com/pysersic/pysersic/actions/workflows/pytest.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue)

pysersic is a code for fitting sersic profiles to galaxy images using Bayesian Inference. It is written in python using [jax](https://github.com/google/jax) with inference performed using [numpyro](https://github.com/pyro-ppl/numpyro)

## Installation

First you should install jax. The easiest way to do this is to use ``` pip install 'jax[cpu]' ```, however if you are on windows or planning on running it on a GPU, the installation can be a little more complicated, please see the guide [here](https://github.com/google/jax#installation). The other package dependencies for ```pysersic``` are:
- arviz
- asdf
- astropy
- corner
- jax
- matplotlib
- numpy
- numpyro
- pandas
- photutils
- scipy
- tqdm

Next you can install pysersic! To install the latest stable release, you can simply pip install via 

```
pip install pysersic
```
If you would like the bleeding edge, development build (or would like to modify, add to, or otherwise access ```pysersic``` source files), you can also install from the github:

```
git clone https://github.com/pysersic/pysersic.git
cd pysersic
pip install -e .
```

## Basic usage

All you need to run pysersic is a cutout of you galaxy of interest, and cutout of the error map, pixelized version of the PSF, and optionally a mask specifying bad pixels or nearby sources. The most basic setup is shown below for many many more details and in depth examples please see the [documentation](https://pysersic.readthedocs.io/en/latest/). First we set up a prior.

```
from pysersic.priors import SourceProperties

props = SourceProperties(img_cutout,mask=mask) # Optional mask
prior = props.generate_prior('sersic', # Other profiles inclues 'exp', 'dev' and 'pointsource'
                          sky_type='none') # Can also use 'flat' or 'tilted-plane' to simultaneously fit a background
```
Then we initialize the fitting object:

```
from pysersic import FitSingle
from pysersic.loss import gaussian_loss

fitter = FitSingle(data=img_cutout, # Cutout of galaxy 
                  rms=sig_cutout, # Cutout of pixel errors
                  psf=psf, #Point spread function
                  prior=prior, # Prior that we initialized above
                  mask=mask, #Optional mask
                  loss_func=gaussian_loss) # Can specify loss function! See loss.py for many options or write your own!
```

Now we can fit!

```
map_dict = fitter.find_MAP() # Find the 'best-fit' parameters as the maximum-a-posteriori. Returns a dictionary containing the MAP parameters and model image

svi_res = fitter.estimate_posterior('svi-flow') #Train a normalizing flow to estimate the posterior, retruns a PysersicResults object containing the posterior and other helper methods
nuts_res = fitter.sample() #Sample the posterior using the No U-turn sampler (NUTS), retruns a PysersicResults object
```

## Citation
We will be submitting a paper soon describing pysersic for now please reference this webpage and cite 'Pasha & Miller in prep.' 
