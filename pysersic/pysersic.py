import jax
import jax.numpy as jnp
from jax import jit
from numpyro import distributions as dist, infer
import numpyro
import arviz as az

from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.initialization import init_to_median
from jax import random


from pysersic.rendering import *
from pysersic.utils import autoprior


class FitSingle():
    def __init__(self,data,weight_map,psf_map,mask = None,sky_model = None, profile = 'single', renderer = FourierRenderer, renderer_kwargs = {}):
        # Assert weightmap shap is data shape
        if data.shape != weight_map.shape:
            raise AssertionError('Weight map ndims must match input data')
        
        if sky_model not in [None,'flat','tilted-plane']:
            raise AssertionError('Sky model must match one of: None,flat, tilted-plane')
        else:
            self.sky_model = sky_model

        self.renderer = renderer(jnp.array(data.shape), jnp.array(psf_map), **renderer_kwargs)

        if profile == 'single':
            self.render_func = renderer.render_sersic
        #elif profile == 'double':
        #    self.render_func = renderer.render_doublesersic
        #elif profile == 'ps':
        #    self.render_func = renderer.render_pointsource
        else:
            raise AssertionError('currently only single sersic supported')

        self.data = jnp.array(data) 
        self.weight_map = jnp.array(weight_map)
        self.rms_map = 1/jnp.sqrt(weight_map)
        if mask is None:
            self.mask = jnp.ones_like(self.data).astype(jnp.bool_)
        else:
            self.mask = jnp.logical_not(jnp.array(mask)).astype(jnp.bool_)

        self.prior_dict = {}

    def set_prior(self,parameter,distribution):
        #setattr(self,parameter+'_prior',distribution)
        self.prior_dict[parameter] = distribution
    
    def autogenerate_priors(self):
        prior_dict = autoprior(self.data)
        for i in prior_dict.keys():
            self.set_prior(i,prior_dict[i])
    

    def build_model(self,):
        def model():
            prior_dict = self.prior_dict
            #Need to have someway to change the parameters given different profiles
            flux = numpyro.sample('flux', prior_dict['flux'])
            n = numpyro.sample('n',prior_dict['n'])
            r_eff = numpyro.sample('r_eff',prior_dict['r_eff'])
            ellip = numpyro.sample('ellip',prior_dict['ellip'])
            theta = numpyro.sample('theta',prior_dict['theta'])
            x_0 = numpyro.sample('x_0',prior_dict['x_0'])
            y_0 = numpyro.sample('y_0',prior_dict['y_0'])

            #collect params and render scene
            params = jnp.array([x_0,y_0,flux,r_eff,n, ellip, theta])
            out = self.renderer.render_sersic(*params)
            
            if self.sky_model =='flat':
                sky_back = numpyro.sample('sky0', dist.Normal(0, 1e-3))
                out = out + sky_back
            if self.sky_model =='tilted_plane':
                sky_back = numpyro.sample('sky0', dist.Normal(0, 1e-3))
                sky_x_sl = numpyro.sample('sky1', dist.Normal(0, 1e-3))
                sky_y_sl = numpyro.sample('sky2', dist.Normal(0, 1e-3))
                out  = out + sky_back + (self.renderer.X -  self.im_shape[1][0]/2.)*sky_x_sl + (self.renderer.Y - self.im_shape[1]/2.)*sky_y_sl
           
            with numpyro.handlers.mask(mask = self.mask):
                numpyro.sample("obs", dist.Normal(out, self.rms_map), obs=self.data)

        return model
    
    def injest_data(self, sampler = None, svi_res_dict = {}):
        
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

    def sample(self,
                sampler_kwargs = dict(init_strategy=init_to_median, 
                target_accept_prob = 0.9),
                mcmc_kwargs = dict(num_warmup=1000,
                num_samples=1000,
                num_chains=2,
                progress_bar=True),       
        ):

        model =  self.build_model()
        
        self.sampler =infer.MCMC(infer.NUTS(model, **sampler_kwargs),**mcmc_kwargs)
        self.sampler.run(jax.random.PRNGKey(3))

        self.injest_data(sampler = self.sampler)

        return az.summary(self.az_data)


    def optimize(self):
        optimizer = numpyro.optim.Adam(jax.example_libraries.optimizers.inverse_time_decay(1e-1, 1000, 0.1, staircase=True) )
        
        model = self.build_model()
        guide = numpyro.infer.autoguide.AutoLaplaceApproximation(model)
        
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO(), )
        svi_result = svi.run(random.PRNGKey(1), 3000)
        
        self.svi_res_dict = dict(guide = guide, model = model, svi_result = svi_result)
        self.injest_data(svi_res_dict= self.svi_res_dict)
        return az.summary(self.az_data)