import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from numpyro import distributions as dist, infer
import numpyro
import arviz as az
import pandas
import optax
from abc import abstractmethod,ABC

from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.initialization import init_to_median
from numpyro.infer.reparam import TransformReparam,CircularReparam
from jax import random


from pysersic.rendering import PixelRenderer,FourierRenderer,HybridRenderer,BaseRenderer, base_profile_params
from pysersic.utils import autoprior,multi_prior, sample_sky


from typing import Union, Optional, Callable
ArrayLike = Union[np.array, jax.numpy.array]

class BaseFitter(ABC):
    """
    Bass class for Pysersic Fitters
    """
    def __init__(self,
        data: ArrayLike,
        weight_map: ArrayLike,
        psf_map: ArrayLike,
        mask: Optional[ArrayLike] = None,
        sky_model: Optional[str] = None,
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
        renderer : Optional[BaseRenderer], optional
            The renderer to be used to generate model images, by default HybridRenderer
        renderer_kwargs : Optional[dict], optional
            Any additional arguments to pass to the renderer, by default {}
        """

        if data.shape != weight_map.shape:
            raise AssertionError('Weight map ndims must match input data')
            
        self.data = jnp.array(data) 
        self.weight_map = jnp.array(weight_map)
        self.rms_map = 1/jnp.sqrt(weight_map)
        if mask is None:
            self.mask = jnp.ones_like(self.data).astype(jnp.bool_)
        else:
            self.mask = jnp.logical_not(jnp.array(mask)).astype(jnp.bool_)

        self.renderer = renderer(data.shape, jnp.array(psf_map), **renderer_kwargs)
    
        if sky_model not in [None,'flat','tilted-plane']:
            raise AssertionError('Sky model must match one of: None,flat, tilted-plane')
        else:
            self.sky_model = sky_model
        
        self.prior_dict = {}

    
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
    
    def generate_sky_priors(self) -> None:
        """
        Generate default priors for the parameters controlling the sky background
        """
        if self.sky_model == 'flat':
            self.set_prior('sky0',dist.TransformedDistribution(
                                dist.Normal(),
                                dist.transforms.AffineTransform(0.0,1e-4),)
                            )
        elif self.sky_model == 'tilted-plane':
            self.set_prior('sky0',  dist.TransformedDistribution(
                                dist.Normal(),
                                dist.transforms.AffineTransform(0.0,1e-4),)
            )
            self.set_prior('sky1',  dist.TransformedDistribution(
                                dist.Normal(),
                                dist.transforms.AffineTransform(0.0,1e-5),)
            )

            self.set_prior('sky2',dist.TransformedDistribution(
                                dist.Normal(),
                                dist.transforms.AffineTransform(0.0,1e-5),)
            )
    
    def injest_data(self, 
                sampler: Optional[numpyro.infer.mcmc.MCMC] =  None, 
                svi_res_dict: Optional[dict] =  None,
                purge_extra: Optional[bool] = True,
                rkey: Optional[jax.random.PRNGKey] = random.PRNGKey(5)
        ) -> pandas.DataFrame:
        """Method to injest data from optimized SVI model or results of sampling. Sets the class attribute 'idata' with an Arviz InferenceData object.

        Parameters
        ----------
        sampler : Optional[numpyro.infer.mcmc.MCMC], optional
            numpyro sampler containing results
        svi_res_dict : Optional[dict], optional
            Dictionary containing 'guide', 'model' and 'svi_result' specifying a trained SVI model
        purge_extra : Optional[bool], optional
            Whether to purge variables containing 'auto', 'base' or 'unwrapped' often used in reparamaterization, by default True
        rkey : Optional[jax.random.PRNGKey], optional
            PRNG key to use, by default jax.random.PRNGKey(5)

        Returns
        -------
        pandas.DataFrame
            ArviZ Summary of results

        Raises
        ------
        AssertionError
            Must supply one of sampler or svi_dict
        """

        if sampler is None and (svi_res_dict is None):
            raise AssertionError("Must svi results dictionary or sampled sampler")

        elif not sampler is None:
            self.idata = az.from_numpyro(sampler)
        else:
            assert 'guide' in svi_res_dict.keys()
            assert 'model' in svi_res_dict.keys()
            assert 'svi_result' in svi_res_dict.keys()

            post_raw = svi_res_dict['guide'].sample_posterior(rkey, svi_res_dict['svi_result'].params, sample_shape = ((1000,)))
            #Convert to arviz
            post_dict = {}
            for key in post_raw:
                post_dict[key] = post_raw[key][jnp.newaxis,]
            self.idata = az.from_dict(post_dict)

        var_names = list(self.idata.posterior.to_dataframe().columns)
        
        for var in var_names:
            if 'theta' in var:
                new_theta = np.remainder(self.idata['posterior'][var]+np.pi, np.pi)
                self.idata['posterior'][var] = new_theta

        if purge_extra:
            to_drop = []
            for var in var_names:
                if ('base' in var) or ('auto' in var) or ('unwrapped' in var):
                    to_drop.append(var)

            self.idata.posterior = self.idata.posterior.drop_vars(to_drop)

        return az.summary(self.idata)


    def sample(self,
                sampler_kwargs: Optional[dict] = dict(init_strategy =  init_to_median),
                mcmc_kwargs: Optional[dict] = 
                dict(num_warmup=500,
                num_samples=500,
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

        summary = self.injest_data(sampler = self.sampler)

        return summary

    def optimize(self,
            Nrun:Optional[int] = 4000,
            rkey: Optional[jax.random.PRNGKey] = jax.random.PRNGKey(3) 
            )-> pandas.DataFrame:
        """ Perform inference by optimizing a Multivariate Normal SVI model. This is a good starting place to find a 'best fit' along with reasonable uncertainties.

        Parameters
        ----------
        Nrun : Optional[int], optional
            Number of training steps, by default 2000
        rkey : Optional[jax.random.PRNGKey], optional
            _description_, by default jax.random.PRNGKey(3)

        Returns
        -------
        pandas.DataFrame
            ArviZ summary of posterior
        """

        optax_optimizer = optax.adabelief(0.1)
        optimizer = numpyro.optim.optax_to_numpyro(optax_optimizer)
        
        model = self.build_model()
        guide = numpyro.infer.autoguide.AutoMultivariateNormal(model)
        
        svi = SVI(model, guide, optimizer, loss = Trace_ELBO(num_particles=2), )
        svi_result = svi.run(rkey, Nrun)
        
        self.svi_res_dict = dict(guide = guide, model = model, svi_result = svi_result)
        summary = self.injest_data(svi_res_dict= self.svi_res_dict)
        return summary
    

    @abstractmethod
    def build_model(self,):
        raise NotImplementedError

    @abstractmethod
    def autogenerate_priors(self,):
        raise NotImplementedError
    
    @abstractmethod
    def render_best_fit(self,):
        raise NotImplementedError

    # TODO Some analysis stuff here
    def plot_residuals(self):
        return None

class FitSingle(BaseFitter):
    """
    Class used to fit a single source
    """
    def __init__(self,
        data: ArrayLike,
        weight_map: ArrayLike,
        psf_map: ArrayLike,
        mask: Optional[ArrayLike] = None,
        sky_model: Optional[str] = None,
        profile_type: Optional[str] = 'sersic', 
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

        super().__init__(data,weight_map,psf_map,mask = mask,sky_model = sky_model, renderer = renderer, renderer_kwargs = renderer_kwargs)

        if profile_type in self.renderer.profile_types:
            self.profile_type = profile_type
            self.param_names = self.renderer.profile_params[self.profile_type]
        else:
            raise AssertionError('Profile must be one of:', renderer.profile_types)

    def autogenerate_priors(self) -> None:
        """Generate default priors based on image and profile type. Calls pysersic.utils.autoprior
        """
        prior_dict = autoprior(self.data*self.mask, self.profile_type)
        for i in prior_dict.keys():
            self.set_prior(i,prior_dict[i])
        
        #set sky priors
        self.generate_sky_priors()



    def build_model(self,) -> Callable:
        """ Generate Numpyro model for the specified image, profile and priors

        Returns
        -------
        model: Callable
            Function specifying the current model in Numpyro, can be passed to inference algorithms
        """
        # Make sure all variables have priors
        check_prior_dict = np.all([
            (param in self.param_names) or ('sky' in param) for param in self.prior_dict.keys()
        ])
        if not check_prior_dict:
            raise AssertionError('Not all variables have priors, please run .autogenerate_priors')

        #Set up and reparamaterization, 
        reparam_dict = {}
        for key in self.prior_dict.keys():
            if hasattr(self.prior_dict[key], 'transforms'):
                reparam_dict[key] = TransformReparam()
            elif type(self.prior_dict[key]) == numpyro.distributions.directional.VonMises:
                reparam_dict[key] = CircularReparam()

        @numpyro.handlers.reparam(config = reparam_dict)
        def model():
            params = jnp.array([numpyro.sample(name,self.prior_dict[name]) for name in self.param_names])
            out = self.renderer.render_source(params, self.profile_type)

            sky_params = sample_sky(self.prior_dict, self.sky_model)
            sky = self.renderer.render_sky(sky_params, self.sky_model)

            obs = out + sky
            
            with numpyro.handlers.mask(mask = self.mask):
                numpyro.sample("obs", dist.Normal(obs, self.rms_map), obs=self.data)

        return model

    def render_best_fit(self):
        assert hasattr(self, 'idata')
        medians = self.idata.posterior.median()
        median_params = jnp.array([medians[name].data for name in self.param_names])
        mod = self.renderer.render_source(median_params, self.profile_type)
        return mod

class FitMulti(BaseFitter):
    """
    Class used to fit multiple sources within a single image
    """
    def __init__(self,
        data: ArrayLike,
        weight_map: ArrayLike,
        psf_map: ArrayLike,
        mask: Optional[ArrayLike] = None,
        sky_model: Optional[str] = None,
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
        super().__init__(data,weight_map,psf_map,mask = mask,sky_model = sky_model, renderer = renderer, renderer_kwargs = renderer_kwargs)

        if type(self.renderer) not in [FourierRenderer,HybridRenderer]:
            raise AssertionError('Currently only FourierRenderer and HybridRenderer Supported for FitMulti')
    
    def autogenerate_priors(self,
        catalog: Union[pandas.DataFrame,dict, np.recarray]
        )-> None:
        """Ingest a catalog-like data structure containing prior positions and parameters for multiple sources in a single image. The format of the catalog can be a `pandas.DataFrame`, `numpy` RecordArray, dictionary, or any other format so-long as the following fields exist and can be directly indexed: 'x', 'y', 'flux', 'r' and 'type'

        Parameters
        ----------
        catalog : Union[pandas.DataFrame,dict, np.recarray]
            Object containing information about the sources to be fit
        """
        prior_list = multi_prior(self.data, catalog)
        self.prior_list = prior_list
        self.N_sources = len(prior_list)
        self.source_types = catalog['type']

        self.source_params = []
        for cur_type in self.source_types:
            self.source_params.append(base_profile_params[cur_type])

        #set sky priors
        self.generate_sky_priors()

    def build_model(self,) -> Callable:
        """Generate Numpyro model for the specified image, profile and priors

        Returns
        -------
        model: Callable
            Function specifying the current model in Numpyro, can be passed to inference algorithms
        """
        #Set up reparamaterization
        reparam_dict= {}
        for j,source in enumerate(self.prior_list):
            for key in source.keys():
                if hasattr(source[key], 'transforms'):
                    reparam_dict[key] = TransformReparam()
            if self.source_types[j] != 'pointsource':
                reparam_dict[f'theta_{j:d}'] = CircularReparam()

        @numpyro.handlers.reparam(config = reparam_dict)
        def model():
            #Loop through sources to generate variables
            source_variables = []
            for j,(prior, params) in enumerate( zip(self.prior_list, self.source_params) ):
                sampled_params = jnp.array([numpyro.sample(f'{name}_{j:d}',prior[f'{name}_{j:d}']) for name in params])
                source_variables.append( sampled_params )

            out = self.renderer.render_multi(self.source_types,source_variables)
            sky_params = sample_sky(self.prior_dict, self.sky_model)
            sky = self.renderer.render_sky(sky_params, self.sky_model)

            obs = out + sky
            
            with numpyro.handlers.mask(mask = self.mask):
                numpyro.sample("obs", dist.Normal(obs, self.rms_map), obs=self.data)

        return model
    def render_best_fit(self,):
        raise NotImplementedError