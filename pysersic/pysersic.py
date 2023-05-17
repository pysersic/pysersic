import copy
from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pandas
import tqdm
from jax.random import PRNGKey
from numpyro import deterministic, infer, optim
from numpyro.handlers import condition, trace
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.svi import SVIRunResult

from pysersic.priors import PySersicMultiPrior, PySersicSourcePrior
from pysersic.rendering import BaseRenderer, HybridRenderer
from pysersic.results import PySersicResults

from .loss import gaussian_loss

ArrayLike = Union[np.array, jax.numpy.array]

class BaseFitter(ABC):
    """
    Base class for Pysersic Fitters
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
        self.psf = jnp.array(psf)

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
                num_samples: int = 1000,
                num_warmup: int = 1000,
                num_chains: int = 2,
                init_strategy: Optional[Callable] = infer.init_to_sample,
                sampler_kwargs: Optional[dict] ={},
                mcmc_kwargs: Optional[dict] = {},
                return_model: Optional[bool] = True,
                rkey: Optional[jax.random.PRNGKey] = jax.random.PRNGKey(3)     
        ) -> pandas.DataFrame:
        """ Perform inference using a NUTS sampler

        Parameters
        ----------
        num_samples : int, optional
            Number of samples to draw, by default 1000
        num_warmup : int, optional
            Number of warmup samples, by default 1000
        num_chains : int, optional
            Number of chains to run, by default 2
        init_strategy : Optional[Callable], optional
            Initialization strategy for the sampler, by default infer.init_to_sample. See numpyro.infer.initialization for more options
        sampler_kwargs : Optional[dict], optional
            Arguments to pass to the numpyro NUTS kernel
        mcmc_kwargs : Optional[dict], optional
            Arguments to pass to the numpyro MCMC sampler
        return_model : Optional[bool]
            Whether to return the model images but adds a small memory/time overhead, by default True
        rkey : Optional[jax.random.PRNGKey], optional
            PRNG key to use, by default jax.random.PRNGKey(3)

        Returns
        -------
        pandas.DataFrame
            ArviZ summary of posterior
        """

        model =  self.build_model(return_model = return_model)
        self.sampler =infer.MCMC(infer.NUTS(model,init_strategy=init_strategy, **sampler_kwargs),num_chains=num_chains, num_samples=num_samples, num_warmup=num_warmup,  **mcmc_kwargs)
        self.sampler.run(rkey)
        self.sampling_results = PySersicResults(data=self.data,rms=self.rms,psf=self.psf,mask=self.mask,loss_func=self.loss_func,renderer=self.renderer)
        self.sampling_results.add_prior(self.prior)
        self.sampling_results.injest_data(sampler = self.sampler)
        return self.sampling_results 
        


    def _train_SVI(self,
            autoguide: numpyro.infer.autoguide.AutoContinuous,
            method:str,
            ELBO_loss: Optional[Callable] = infer.Trace_ELBO(1),
            lr_init: Optional[int] = 1e-2,
            num_round: Optional[int] = 3,
            SVI_kwargs: Optional[dict]= {},
            train_kwargs: Optional[dict] = {},
            return_model: Optional[bool] = True,
            rkey: Optional[jax.random.PRNGKey] = jax.random.PRNGKey(6),
            )-> pandas.DataFrame:
        """
        Internal function to perform inference using stochastic variational inference.

        Parameters
        ----------
        autoguide : numpyro.infer.autoguide.AutoContinuous
            Function to build guide
        method: str
            name of method being used; for saving results
        Elbo_loss : Optional[Callable], optional
            Loss function to use, by default infer.Trace_ELBO(1), see numpyro.infer.elbo for more options
        lr_init : Optional[int], optional
            Initial learning rate, by default 1e-2
        num_round : Optional[int], optional
            Number of rounds for training, lr decreases each round, by default 3
        SVI_kwargs : Optional[dict], optional
            Additional arguments to pass to numpyro.infer.SVI, by default {}
        train_kwargs : Optional[dict], optional
            Additional arguments to pass to utils.train_numpyro_svi_early_stop, by default {}
        return_model : Optional[bool]
            Whether to return the model images but adds a small memory/time overhead, by default True
        rkey : Optional[jax.random.PRNGKey], optional
            PRNG key, by default jax.random.PRNGKey(6)

        Returns
        -------
        pandas.DataFrame
            ArviZ summary of posterior
        """
        model_cur = self.build_model(return_model=return_model)
        guide = autoguide(model_cur)

        svi_kernel = SVI(model_cur,guide, optim.Adam(0.1), loss = ELBO_loss, **SVI_kwargs)
        numpyro_svi_result = train_numpyro_svi_early_stop(svi_kernel,rkey=rkey,lr_init=lr_init,
                                                        num_round=num_round, **train_kwargs)

        svi_res_dict =  dict(guide = guide, model = model_cur, svi_result = numpyro_svi_result)
        self.svi_results = PySersicResults(data=self.data,rms=self.rms,psf=self.psf,mask=self.mask,loss_func=self.loss_func,renderer=self.renderer)
        self.svi_results.injest_data(svi_res_dict=svi_res_dict,purge_extra=True)
        self.svi_results.add_prior(self.prior)
        self.svi_results.add_method_used(method)
        return self.svi_results


    
    def find_MAP(self,
                return_model: Optional[bool] = True,
                rkey: Optional[jax.random.PRNGKey] = jax.random.PRNGKey(3),):
        """Find the "best-fit" parameters as the maximum a-posteriori and return a dictionary with values for the parameters.

        Parameters
        ----------
        return_model : Optional[bool]
            Whether to return the model images but adds a small memory/time overhead, by default True
        rkey : Optional[jax.random.PRNGKey], optional
            rng key, by default jax.random.PRNGKey(3)

        Returns
        -------
        dict
            dictionary with fit parameters and their values.
        """
        model_cur = self.build_model(return_model=return_model)
        autoguide_map = infer.autoguide.AutoDelta(model_cur, init_loc_fn= infer.init_to_median)
        train_kwargs = dict(lr_init = 0.01, num_round = 3,frac_lr_decrease  = 0.1, patience = 250, optimizer = optim.Adam, max_train = int(1e4))
        svi_kernel = SVI(model_cur,autoguide_map, optim.Adam(0.01),loss=Trace_ELBO())
        
        res = train_numpyro_svi_early_stop(svi_kernel,rkey=rkey, **train_kwargs)

        use_dict = {}
        for key in res.params.keys():
            pref = key.split('_auto_loc')[0]
            use_dict[pref] = res.params[key]
        trace_out = trace(condition(model_cur, use_dict)).get_trace()
        real_out = {}
        for key in trace_out:
            if key == 'Loss':
                continue
            elif key == 'model':
                real_out[key] = np.asarray(trace_out[key]['value'])
            elif not ('base' in key or 'auto' in key or 'unwrapped' in key or 'factor' in key or 'loss' in key):
                real_out[key] = float('{:.5e}'.format(trace_out[key]['value']) )

        return real_out
    
    def estimate_posterior(self,
                        method : str='laplace',
                        return_model: bool = True,
                        rkey: Optional[jax.random.PRNGKey] = jax.random.PRNGKey(6),
                        ) -> pandas.DataFrame:
        """Estimate the posterior using a method other than MCMC sampling. Generally faster than MCMC, but could be less accurate.
        Current Options are:
        - 'laplace'
            - Uses the Laplace approximation, which finds the MAP and then uses a Gaussian approximation to the posterior. The covariance matrix is calculated using the Hessian of the log posterior at the MAP.
        - 'svi-flow'
            - Uses a normalizing flow (currently a BNAF, https://arxiv.org/abs/1904.04676) to approximate the posterior. This is more flexible than the Laplace approximation, but is slower to train. Optimization can be inconsistet so use and interpret with caution. Best to cross-reference with sample on tests cases.

        Parameters
        ----------
        method : str, optional
            method to use, by default 'laplace'
        return_model : Optional[bool]
            Whether to return the model images but adds a small memory/time overhead, by default True
        rkey : Optional[jax.random.PRNGKey], optional
            rng key, by default jax.random.PRNGKey(6)
        
        """
        assert method in ['laplace','svi-flow']
        if method=='laplace':
            train_kwargs = dict(patience = 250, max_train = 10000)
            guide_func = partial(infer.autoguide.AutoLaplaceApproximation, init_loc_fn = infer.init_to_median )
            results = self._train_SVI(guide_func,method=method, train_kwargs=train_kwargs, return_model = return_model, rkey=rkey)
        elif method=='svi-flow':
            train_kwargs = dict(patience = 500, max_train = 20000)
            guide_func = partial(infer.autoguide.AutoBNAFNormal, num_flows =4,hidden_factors = [5,], init_loc_fn = infer.init_to_median)
            results = self._train_SVI(guide_func,method='svi-flow',ELBO_loss= infer.Trace_ELBO(8),train_kwargs=train_kwargs,num_round=3,lr_init = 1e-2, rkey=rkey,return_model = return_model,)
        return results.summary()

    @abstractmethod
    def build_model(self,return_model: bool = True):
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
        


    def build_model(self, return_model: bool = True) -> Callable:
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
        def model(return_model: bool = return_model):
            params = self.prior()
            out = self.renderer.render_source(params, self.prior.profile_type)

            sky = self.prior.sample_sky(self.renderer.X, self.renderer.Y)

            obs = out + sky
            if return_model:
                obs = deterministic('model', obs)
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
        
    def build_model(self, return_model: bool = True) -> Callable:
        """Generate Numpyro model for the specified image, profile and priors

        Returns
        -------
        model: Callable
            Function specifying the current model in Numpyro, can be passed to inference algorithms
        """

        @numpyro.handlers.reparam(config = self.prior.reparam_dict)
        def model(return_model: bool = return_model):
            source_variables = self.prior()

            out = self.renderer.render_multi(self.prior.catalog['type'],source_variables)
            sky = self.prior.sample_sky(self.renderer.X, self.renderer.Y)

            obs = out + sky

            if return_model:
                obs = deterministic('model', obs)
            
            loss = self.loss_func(obs, self.data, self.rms, self.mask)
            return loss

        return model

    def find_MAP(self,
                return_model: Optional[bool] = True,
                rkey: Optional[jax.random.PRNGKey] = jax.random.PRNGKey(3),):
        """Find the "best-fit" parameters as the maximum a-posteriori and return a dictionary with values for the parameters.

        Parameters
        ----------
        return_model : Optional[bool], optional
            whether to return the model image, adds a small time and memory overhead, by default True
        rkey : Optional[jax.random.PRNGKey], optional
            rng key, by default jax.random.PRNGKey(3)

        Returns
        -------
        dict
            dictionary with fit parameters and their values.
        """
        raw_dict = super().find_MAP(return_model=return_model, rkey=rkey)
        results_dict = {}
        for i in range(self.prior.N_sources):
            results_dict[f'source_{i}'] = {}
            for pname in self.prior.all_priors[i].param_names:
                results_dict[f'source_{i}'][pname] = raw_dict.pop(pname + f'_{i:d}')
        results_dict.update(raw_dict)
        return results_dict




def train_numpyro_svi_early_stop(
        svi_class: SVI,
        num_round: Optional[int] = 3,
        max_train: Optional[int] = 5000,
        lr_init: Optional[float] = 0.01,
        frac_lr_decrease: Optional[float]  = 0.1,
        patience: Optional[int] = 100,
        optimizer: Optional[optim._NumPyroOptim] = optim.Adam,
        rkey: Optional[PRNGKey] = PRNGKey(10),
    )-> SVIRunResult:
    """Optimize a SVI model by training for multiple rounds with a deacreasing learning rate, and early stopping for each round

    Parameters
    ----------
    svi_class : SVI
        Initialized numpyo SVI class, note that the optimizer will be overwritten
    num_round : Optional[int], optional
        Number of training rounds, by default 3
    max_train : Optional[int], optional
        Max number of training epochs per ropund, by default 3000
    lr_init : Optional[float], optional
        Initial learning rate, by default 0.1
    frac_lr_decrease : Optional[float], optional
        Multiplicative factor to change learning rate each round, by default 0.1
    patience : Optional[int], optional
        Number of training epochs to wait for improvement, by default 100
    optimizer : Optional[optim._NumPyroOptim], optional
        Optimizer algorithm tro use, by default optim.Adam
    rkey : Optional[PRNGKey], optional
        Jax PRNG key, by default PRNGKey(10)

    Returns
    -------
    SVIRunResult
        SVI Result class containing trained model
    """
    optim_init = optimizer(lr_init)
    svi_class.__setattr__('optim', optim_init)

    init_state = svi_class.init(rkey)
    all_losses = []

    @partial(jax.jit, static_argnums = 1)
    def update_func(state,svi_class,lr):
        svi_class.__setattr__('optim', optimizer(lr))
        state,loss = svi_class.stable_update(state)
        return state,loss

    best_state, best_loss = update_func(init_state, svi_class, lr_init)
    
    for r in range(num_round):
        losses = []
        wait_counter = 0
        svi_state = copy.copy(best_state)

        if r>0:
            lr_cur = lr_init*frac_lr_decrease**r
            best_loss = jnp.inf
        else:
            lr_cur = lr_init

        with tqdm.trange(1, max_train + 1) as t:
            for j in t:
                svi_state, loss = update_func(svi_state, svi_class, lr_cur)
                if loss < best_loss:
                    best_loss = loss
                    best_state = copy.copy(svi_state)
                    wait_counter = 0
                elif wait_counter >= patience:
                    break
                else:
                    wait_counter += 1
                t.set_postfix_str(f'Round = {r:d},step_size = {lr_cur:.1e} loss: {best_loss:.3e}',refresh=False)
                losses.append(loss)
        
        all_losses.append(losses)

    return SVIRunResult(svi_class.get_params(best_state), svi_state,losses)
    
