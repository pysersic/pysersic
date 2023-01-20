import jax
import jax.numpy as jnp
from jax import jit
from numpyro import distributions as dist, infer
import numpyro
import arviz as az

from numpyro.infer import SVI, Trace_ELBO
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

        self.renderer = renderer(data.shape, psf_map, **renderer_kwargs)

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
            log_flux = numpyro.sample("log_flux",prior_dict['log_flux'])
            flux = numpyro.deterministic("flux", jnp.power(10,log_flux))
            
            log_n = numpyro.sample("log_n",prior_dict['log_n'])
            n = numpyro.deterministic("n", jnp.power(10,log_n))
            
            r_eff = numpyro.sample("r_eff",prior_dict['r_eff'])

            ellip = numpyro.sample("ellip",prior_dict['ellip'])
            theta = numpyro.sample("theta", prior_dict['theta'])
            x_0 = numpyro.sample("x_0",prior_dict['x_0'])
            y_0 = numpyro.sample("y_0",prior_dict['y_0'])

            #collect params and render scene
            params = jnp.array([x_0,y_0,flux,r_eff,n, ellip, theta])
            out = self.render_func(*params)
            
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
            return AssertionError("Must supply trained guide or sampled sampler")

        elif not sampler is None:
            self.az_data = az.from_numpyro(sampler)
            #Do other things 

        else:

            #Write function to sample posterior from svi guide 
            return NotImplementedError
    
    def sample(self,num_warmup=1000,
                num_samples=1000,
                num_chains=2,
                progress_bar=True):

        model = self.build_model()
        
        self.sampler =infer.MCMC(
                                infer.NUTS(model),
                                num_warmup=num_warmup,
                                num_samples=num_samples,
                                num_chains=num_chains,
                                progress_bar=progress_bar,
                            )
        self.sampler.run(jax.random.PRNGKey(3))
        self.injest_data(sampler = self.sampler)

        return az.summary(self.az_data)


    def optimize(self):
        optimizer = numpyro.optim.Adam(jax.example_libraries.optimizers.inverse_time_decay(1e-1, 500, 0.5, staircase=True) )
        
        model = self.build_model()
        self.guide = numpyro.infer.autoguide.AutoMultivariateNormal(self.model)
        
        svi = SVI(model, self.guide, optimizer, loss=Trace_ELBO(), )
        svi_result = svi.run(random.PRNGKey(1), 5000)
        
        #still need to write this function
        self.injest_data()# need to write this part
        return az.summary(self.az_data)