from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Optional, Union

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas
from jax import random
from numpyro import infer
from numpyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from numpyro.optim import Adam, optax_to_numpyro
from optax import adamw

from pysersic.rendering import (
    BaseRenderer,
    FourierRenderer,
    HybridRenderer,
)
from pysersic.priors import PySersicSourcePrior, PySersicMultiPrior
from pysersic.utils import gaussian_loss, train_numpyro_svi_early_stop 
from pysersic.results import PySersicResults

ArrayLike = Union[np.array, jax.numpy.array]

class BaseFitter(ABC):
    """
    Bass class for Pysersic Fitters
    """
    def __init__(self,
        data: ArrayLike,
        rms: ArrayLike,
        psf: ArrayLike,
        mask: Optional[ArrayLike] = None,
        loss_func: Optional[Callable] = gaussian_loss,
        renderer: Optional[BaseRenderer] =  HybridRenderer, 
        renderer_kwargs: Optional[dict] = {}) -> None:
        """Initialze BaseFitter class

        Parameters
        ----------
        data : ArrayLike
            Science image to be fit
        weight_map : ArrayLike
            Weight map (one over the variance) corresponding to `data`, must be the same shape
        psf_map : ArrayLike
            Pixelized PSF
        mask : Optional[ArrayLike], optional
            Array specifying the mask, `True` or 1 signifies a pixel should be masked, must be same shape as `data`
        sky_model : Optional[str], optional
            One of None, 'flat' or 'tilted-plane' specifying how to model the sky background
        renderer : Optional[BaseRenderer], optional
            The renderer to be used to generate model images, by default HybridRenderer
        renderer_kwargs : Optional[dict], optional
            Any additional arguments to pass to the renderer, by default {}
        """

        
        self.loss_func = loss_func

        if data.shape != rms.shape:
            raise AssertionError('rms map ndims must match input data')
            
        self.data = jnp.array(data) 
        self.rms = jnp.array(rms)
        self.psf = jnp.array(rms)

        if mask is None:
            self.mask = jnp.ones_like(self.data).astype(jnp.bool_)
        else:
            self.mask = jnp.logical_not(jnp.array(mask)).astype(jnp.bool_)

        self.renderer = renderer(data.shape, jnp.array(psf), **renderer_kwargs)
    
        self.prior_dict = {}

    
    def set_loss_func(self, loss_func: Callable) -> None:
        """Set loss function to be used for inference

        Parameters
        ----------
        loss_func : Callable
            Functions which takes samples the loss function, see utils/loss.py for some examples.
        """
        self.loss_func = loss_func

    def set_prior(self,parameter: str,
        distribution: numpyro.distributions.Distribution) -> None:
        """Set the prior for a specific parameter

        Parameters
        ----------
        parameter : str
            Parameter to be set
        distribution : numpyro.distributions.Distribution
            Numpyro distribution object corresponding to the prior
        """
        self.prior_dict[parameter] = distribution
    
    


    def sample(self,
                sampler_kwargs: Optional[dict] ={},
                mcmc_kwargs: Optional[dict] = 
                dict(num_warmup=1000,
                num_samples=1000,
                num_chains=2,
                progress_bar=True),
                rkey: Optional[jax.random.PRNGKey] = jax.random.PRNGKey(3)     
        ) -> pandas.DataFrame:
        """ Perform inference using the NUTS sampler using default parameters

        Parameters
        ----------
        sampler_kwargs : Optional[dict], optional
            Arguments to pass to the numpyro NUTS kernel
        mcmc_kwargs : Optional[dict], optional
            Arguments to pass to the numpyro MCMC sampler
        rkey : Optional[jax.random.PRNGKey], optional
            PRNG key to use, by default jax.random.PRNGKey(3)

        Returns
        -------
        pandas.DataFrame
            ArviZ summary of posterior
        """
        model =  self.build_model()
        
        self.sampler =infer.MCMC(infer.NUTS(model, **sampler_kwargs),**mcmc_kwargs)
        self.sampler.run(rkey)
        self.sampling_results = PySersicResults(data=self.data,rms=self.rms,psf=self.psf,mask=self.mask,loss_func=self.loss_func,renderer=self.renderer)
        self.sampling_results.add_prior(self.prior)
        self.sampling_results.injest_data(sampler = self.sampler)
        return self.svi_results 
        


    def _train_SVI(self,
            autoguide: numpyro.infer.autoguide.AutoContinuous,
            SVI_kwargs: Optional[dict]= {},
            train_kwargs: Optional[dict] = {},
            rkey: Optional[jax.random.PRNGKey] = jax.random.PRNGKey(6),
            )-> pandas.DataFrame:
        """
        Performance inference using stochastic variational inference.

        Parameters
        ----------
        autoguide : numpyro.infer.autoguide.AutoContinuous
            Function to build guide
        SVI_kwargs : Optional[dict], optional
            Additional arguments to pass to numpyro.infer.SVI, by default {}
        train_kwargs : Optional[dict], optional
            Additional arguments to pass to utils.train_numpyro_svi_early_stop, by default {}
        rkey : Optional[jax.random.PRNGKey], optional
            PRNG key, by default jax.random.PRNGKey(6)

        Returns
        -------
        pandas.DataFrame
            ArviZ summary of posterior
        """
        model_cur = self.build_model()
        guide = autoguide(model_cur)

        svi_kernel = SVI(model_cur,guide, Adam(0.1), **SVI_kwargs)
        self.svi_result = train_numpyro_svi_early_stop(svi_kernel,rkey=rkey, **train_kwargs)

        svi_res_dict =  dict(guide = guide, model = model_cur, svi_result = self.svi_result)
        self.svi_results = PySersicResults(data=self.data,rms=self.rms,psf=self.psf,mask=self.mask,loss_func=self.loss_func,renderer=self.renderer)
        self.svi_results.injest_data(svi_res_dict=svi_res_dict)
        self.svi_results.add_prior(self.prior)
        return self.svi_results

    
    def get_MAP(self,rkey: Optional[jax.random.PRNGKey] = jax.random.PRNGKey(3),):
        pass 
    
    def estimate_posterior(self,
                           method:str='laplace',
                           rkey: Optional[jax.random.PRNGKey] = jax.random.PRNGKey(3),
                           ) -> pandas.DataFrame:
        """Estimate the posterior using one of several methods.
        Options include:
        - 'laplace'
        - 'flow'

        Parameters
        ----------
        method : str, optional
            method to use, by default 'laplace'
        """
        if method=='laplace':
            return self._laplace_fit(rkey=rkey)
        elif method=='flow':
            return self._train_flow(rkey=rkey)
    
    
    
    def _laplace_fit(self,
            rkey: Optional[jax.random.PRNGKey] = jax.random.PRNGKey(3),
            )-> pandas.DataFrame:
        """
        Perform inference by finding the Maximum a-posteriori (MAP) with uncertainties calculated using the Laplace Approximation. This is a good starting place to find a 'best fit' along with reasonable uncertainties.

        Parameters
        ----------
        rkey : Optional[jax.random.PRNGKey], optional
            PRNG key, by default jax.random.PRNGKey(3)

        Returns
        -------
        pandas.DataFrame
            ArviZ summary of posterior
        """

        train_kwargs = dict(lr_init = 0.1, num_round = 4,frac_lr_decrease  = 0.25, patience = 100, optimizer = Adam)
        svi_kwargs = dict(loss = Trace_ELBO(1))
        summary = self._train_SVI(infer.autoguide.AutoLaplaceApproximation, SVI_kwargs=svi_kwargs, train_kwargs=train_kwargs, rkey=rkey)

        return summary
    
    def _train_flow(self,
            rkey: Optional[jax.random.PRNGKey] = jax.random.PRNGKey(3),
        )-> pandas.DataFrame:
        """
        Perform inference using variational inference by fitting a Block Neural Autoregressive Flow (BNAF,https://arxiv.org/abs/1904.04676) to the posterior distribution. Usually faster than MCMC sampling but not guarenteed to converge accurately

        Parameters
        ----------
        rkey : Optional[jax.random.PRNGKey], optional
            PRNG key, by default jax.random.PRNGKey(3)

        Returns
        -------
        pandas.DataFrame
            ArviZ summary of posterior
        """
        def opt_func(lr):
            return optax_to_numpyro(adamw(lr, weight_decay=0.001))

        train_kwargs = dict(lr_init = 4e-3, num_round = 4,frac_lr_decrease  = 0.25, patience = 100, optimizer = opt_func)
        svi_kwargs = dict(loss = TraceMeanField_ELBO(16))
        guide_func = partial(infer.autoguide.AutoBNAFNormal, num_flows = 1)

        summary = self._train_SVI(guide_func, SVI_kwargs=svi_kwargs, train_kwargs=train_kwargs, rkey=rkey)

        return summary

    @abstractmethod
    def build_model(self,):
        raise NotImplementedError


class FitSingle(BaseFitter):
    """
    Class used to fit a single source
    """
    def __init__(self,
        data: ArrayLike,
        rms: ArrayLike,
        psf: ArrayLike,
        prior: PySersicSourcePrior,
        mask: Optional[ArrayLike] = None,
        loss_func: Optional[Callable] = gaussian_loss,
        renderer: Optional[BaseRenderer] =  HybridRenderer, 
        renderer_kwargs: Optional[dict] = {}) -> None:
        """Initialze FitSingle class

        Parameters
        ----------
        data : ArrayLike
            Science image to be fit
        weight_map : ArrayLike
            Weight map (one over the variance) corresponding to `data`, must be the same shape
        psf_map : ArrayLike
            Pixelized PSF
        mask : Optional[ArrayLike], optional
            Array specifying the mask, `True` or 1 signifies a pixel should be masked, must be same shape as `data`
        sky_model : Optional[str], optional
            One of None, 'flat' or 'tilted-plane' specifying how to model the sky background
        profile_type : Optional[str], optional
            Must be one of: ['sersic','doublesersic','pointsource','exp','dev'] specifying how to paramaterize the source, default 'sersic'
        renderer : Optional[BaseRenderer], optional
            The renderer to be used to generate model images, by default HybridRenderer
        renderer_kwargs : Optional[dict], optional
            Any additional arguments to pass to the renderer, by default {}
        """

        super().__init__(data,rms,psf,loss_func = loss_func, mask = mask, renderer = renderer, renderer_kwargs = renderer_kwargs)

        if prior.profile_type not in self.renderer.profile_types:
            raise AssertionError('Profile must be one of:', self.renderer.profile_types)
        self.prior = prior
        


    def build_model(self,) -> Callable:
        """ Generate Numpyro model for the specified image, profile and priors

        Returns
        -------
        model: Callable
            Function specifying the current model in Numpyro, can be passed to inference algorithms
        """
        # Make sure all variables have priors
        check_prior = self.prior.check_vars
        if not check_prior:
            raise AssertionError('Not all variables have priors, please run .autogenerate_priors')

        @numpyro.handlers.reparam(config = self.prior.reparam_dict)
        def model():
            params = self.prior()
            out = self.renderer.render_source(params, self.prior.profile_type)

            sky = self.prior.sample_sky(self.renderer.X, self.renderer.Y)

            obs = out + sky
            
            self.loss_func(obs, self.data, self.rms, self.mask)
        return model

    

class FitMulti(BaseFitter):
    """
    Class used to fit multiple sources within a single image
    """
    def __init__(self,
        data: ArrayLike,
        rms: ArrayLike,
        psf: ArrayLike,
        prior: PySersicMultiPrior,
        mask: Optional[ArrayLike] = None,
        loss_func: Optional[Callable] = gaussian_loss,
        renderer: Optional[BaseRenderer] =  HybridRenderer, 
        renderer_kwargs: Optional[dict] = {}) -> None:
        """Initialze FitMulti class

        Parameters
        ----------
        data : ArrayLike
            Science image to be fit
        weight_map : ArrayLike
            Weight map (one over the variance) corresponding to `data`, must be the same shape
        psf_map : ArrayLike
            Pixelized PSF
        mask : Optional[ArrayLike], optional
            Array specifying the mask, `True` or 1 signifies a pixel should be masked, must be same shape as `data`
        sky_model : Optional[str], optional
            One of None, 'flat' or 'tilted-plane' specifying how to model the sky background
        profile_type : Optional[str], optional
            Must be one of: ['sersic','doublesersic','pointsource','exp','dev'] specifying how to paramaterize the source, default 'sersic'
        renderer : Optional[BaseRenderer], optional
            The renderer to be used to generate model images, by default HybridRenderer
        renderer_kwargs : Optional[dict], optional
            Any additional arguments to pass to the renderer, by default {}
        """
        super().__init__(data,rms,psf,mask = mask,loss_func = loss_func,renderer = renderer, renderer_kwargs = renderer_kwargs)
        self.prior = prior
        if type(self.renderer) not in [FourierRenderer,HybridRenderer]:
            raise AssertionError('Currently only FourierRenderer and HybridRenderer Supported for FitMulti')
    
    def build_model(self,) -> Callable:
        """Generate Numpyro model for the specified image, profile and priors

        Returns
        -------
        model: Callable
            Function specifying the current model in Numpyro, can be passed to inference algorithms
        """

        @numpyro.handlers.reparam(config = self.prior.reparam_dict)
        def model():
            source_variables = self.prior()

            out = self.renderer.render_multi(self.prior.catalog['type'],source_variables)
            sky = self.prior.sample_sky(self.renderer.X, self.renderer.Y)

            obs = out + sky
            
            loss = self.loss_func(obs, self.data, self.rms, self.mask)
            return loss

        return model
    
    def render_best_fit(self,):
        raise NotImplementedError