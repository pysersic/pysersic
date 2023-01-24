import jax
import jax.numpy as jnp
from jax import jit
from numpyro import distributions as dist, infer
import numpyro
import arviz as az

from numpyro.infer import SVI, Trace_ELBO, RenyiELBO
from numpyro.infer.initialization import init_to_median
from jax import random


from pysersic.rendering import *
from pysersic.utils import autoprior, sample_func_dict, sample_sky

class FitSingle():
    def __init__(self,data,weight_map,psf_map,mask = None,sky_model = None, profile_type = 'sersic', renderer = FourierRenderer, renderer_kwargs = {}):
        # Assert weightmap shap is data shape
        if data.shape != weight_map.shape:
            raise AssertionError('Weight map ndims must match input data')
        
        if sky_model not in [None,'flat','tilted-plane']:
            raise AssertionError('Sky model must match one of: None,flat, tilted-plane')
        else:
            self.sky_model = sky_model

        self.renderer = renderer(data.shape, jnp.array(psf_map), **renderer_kwargs)
        
        if profile_type in ['sersic','doublesersic','exp','dev','pointsource']:
            self.profile_type = profile_type
        else:
            raise AssertionError('Profile must be one of: sersic,doublesersic,pointsource')

        self.data = jnp.array(data) 
        self.weight_map = jnp.array(weight_map)
        self.rms_map = 1/jnp.sqrt(weight_map)
        if mask is None:
            self.mask = jnp.ones_like(self.data).astype(jnp.bool_)
        else:
            self.mask = jnp.logical_not(jnp.array(mask)).astype(jnp.bool_)

        self.prior_dict = {}

    def set_prior(self,parameter,distribution):
        self.prior_dict[parameter] = distribution
    
    def autogenerate_priors(self):
        prior_dict = autoprior(self.data, self.profile_type)
        for i in prior_dict.keys():
            self.set_prior(i,prior_dict[i])
        
        #set sky priors
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
    


    def build_model(self,):

        sample_func = sample_func_dict[self.profile_type]

        def model():
            params = sample_func(self.prior_dict)
            out = self.renderer.render_source(params, self.profile_type)

            sky_params = sample_sky(self.prior_dict, self.sky_model)
            sky = self.renderer.render_sky(sky_params, self.sky_model)

            obs = out + sky
            
            with numpyro.handlers.mask(mask = self.mask):
                numpyro.sample("obs", dist.Normal(obs, self.rms_map), obs=self.data)

        return model
    
    def injest_data(self, sampler = None, svi_res_dict = {},purge_extra = True):
        
        if sampler is None and (svi_res_dict is None):
            return AssertionError("Must svi results dictionary or sampled sampler")

        elif not sampler is None:
            self.az_data = az.from_numpyro(sampler)
        else:
            assert 'guide' in svi_res_dict.keys()
            assert 'model' in svi_res_dict.keys()
            assert 'svi_result' in svi_res_dict.keys()

            rkey = random.PRNGKey(5)
            post_raw = svi_res_dict['guide'].sample_posterior(rkey, svi_res_dict['svi_result'].params, sample_shape = ((1000,)))
            #Convert to arviz
            post_dict = {}
            for key in post_raw:
                post_dict[key] = post_raw[key][jnp.newaxis,]
            self.az_data = az.from_dict(post_dict)

        if purge_extra:
            var_names = list(self.az_data.posterior.to_dataframe().columns)
            to_drop = []
            for var in var_names:
                if ('base' in var) or ('auto' in var):
                    to_drop.append(var)

            self.az_data = self.az_data.posterior.drop_vars(to_drop)

        return az.summary(self.az_data)
    def sample(self,
                sampler_kwargs = dict(init_strategy=init_to_median, 
                target_accept_prob = 0.9),
                mcmc_kwargs = dict(num_warmup=1000,
                num_samples=1000,
                num_chains=2,
                progress_bar=True),
                rkey = jax.random.PRNGKey(3)     
        ):
        model =  self.build_model()
        
        self.sampler =infer.MCMC(infer.NUTS(model, **sampler_kwargs),**mcmc_kwargs)
        self.sampler.run(rkey)

        summary = self.injest_data(sampler = self.sampler)

        return summary

    def optimize(self, rkey = random.PRNGKey(1) ):
        optimizer = numpyro.optim.Adam(jax.example_libraries.optimizers.inverse_time_decay(1e-1, 500, 5, staircase=True) )
        
        model = self.build_model()
        guide = numpyro.infer.autoguide.AutoMultivariateNormal(model)
        
        svi = SVI(model, guide, optimizer, loss= RenyiELBO(num_particles=5), )
        svi_result = svi.run(rkey, 5000)
        
        self.svi_res_dict = dict(guide = guide, model = model, svi_result = svi_result)
        summary = self.injest_data(svi_res_dict= self.svi_res_dict)
        return summary