import jax
import jax.numpy as jnp
import tensorflow_probability as tfp
from jax import jit
from numpyro import distributions as dist, infer
import numpyro
import arviz as az
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from numpyro.infer import SVI, Trace_ELBO
from jax import random




@jax.jit
def conv_fft(image,psf_fft):
            img_fft = jnp.fft.fft2(image)
            conv_fft = img_fft*psf_fft
            conv_im = jnp.fft.ifft2(conv_fft)
            return jnp.abs(conv_im)


class FitSersic():
    def __init__(self,data,weight_map,psf_map):
        # Assert weightmap shap is data shape
        if data.shape != weight_map.shape:
            raise AssertionError('Weight map ndims must match input data')
        self.im_shape = data.shape
        self.psf_shape = psf_map.shape
        f1d1 = jnp.fft.fftfreq(data.shape[0])
        f1d2 = jnp.fft.fftfreq(data.shape[1])
        fx,fy = jnp.meshgrid(f1d1,f1d2)
        self.fft_shift_arr_x = jnp.exp(jax.lax.complex(0.,-1.)*2.*3.1415*-1*self.psf_shape[0]/2.*fx)
        self.fft_shift_arr_y = jnp.exp(jax.lax.complex(0.,-1.)*2.*3.1415*-1*self.psf_shape[1]/2.*fy)
        self.psf_fft = jnp.fft.fft2(psf_map, s = self.im_shape)*self.fft_shift_arr_x*self.fft_shift_arr_y
        self.xgrid, self.ygrid = jnp.meshgrid(jnp.arange(data.shape[0]),jnp.arange(data.shape[1]))
        self.data = data 
        self.weight_map = weight_map
        self.psf_map = psf_map
        self.prior_dict = {}



    @staticmethod
    @jit
    def Sersic2D(xgrid,ygrid,x_0,y_0, flux, r_eff, n,ellip, theta,psf_fft,**kwargs):
        bn = 1.9992*n - 0.3271
        a, b = r_eff, (1 - ellip) * r_eff
        cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)
        x_maj = (xgrid - x_0) * cos_theta + (ygrid - y_0) * sin_theta
        x_min = -(xgrid - x_0) * sin_theta + (ygrid - y_0) * cos_theta
        amplitude = flux*bn**(2*n) / ( jnp.exp(bn + jax.scipy.special.gammaln(2*n) ) *r_eff**2 *jnp.pi*2*n )
        z = jnp.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)
        out = amplitude * jnp.exp(-bn * (z ** (1 / n) - 1))
        out = conv_fft(out,psf_fft)
        return out



    def set_prior(self,parameter,distribution):
        #setattr(self,parameter+'_prior',distribution)
        self.prior_dict[parameter] = distribution
    
    
    def model(self,image_error=None,image=None,sky=None,):
        prior_dict = self.prior_dict
        log_flux = numpyro.sample("log_flux",prior_dict['log_flux'])
        flux = numpyro.deterministic("flux", jnp.power(10,log_flux))

        r_eff = numpyro.sample("r_eff",prior_dict['r_eff'])
        log_n = numpyro.sample("log_n",prior_dict['log_n'])
        n = numpyro.deterministic("n", jnp.power(10,log_n))
        ellip = numpyro.sample("ellip",prior_dict['ellip'])
        theta = numpyro.sample("theta", prior_dict['theta'])
        x_0 = numpyro.sample("x_0",prior_dict['x_0'])
        y_0 = numpyro.sample("y_0",prior_dict['y_0'])

        out = FitSersic.Sersic2D(self.xgrid,self.ygrid,x_0,y_0,flux,r_eff,n,ellip,theta,self.psf_fft)
        
        if sky =='flat':
            sky_back = numpyro.sample('sky0', dist.Normal(0, 1e-3))
            out = out + sky_back
        if sky=='tilted_plane':
            sky_back = numpyro.sample('sky0', dist.Normal(0, 1e-3))
            sky_x_sl = numpyro.sample('sky1', dist.Normal(0, 1e-3))
            sky_y_sl = numpyro.sample('sky2', dist.Normal(0, 1e-3))
            out  = out + sky_back + (self.xgrid - image.shape[0]/2.)*sky_x_sl + (self.ygrid - image.shape[0]/2.)*sky_y_sl
        
        #log_nu = numpyro.sample('log_nu',dist.Uniform(0,2))
        #nu_eff = numpyro.deterministic('nu_eff',jnp.power(10,log_nu))
        #numpyro.sample("obs", dist.StudentT(nu_eff,out, jnp.sqrt(nu_eff/(nu_eff-2))*image_error), obs=image)
        
        numpyro.sample("obs", dist.Normal(out, image_error), obs=image)

    def sample(self,num_warmup=1000,
                num_samples=1000,
                num_chains=2,
                progress_bar=True):
        self.sampler =infer.MCMC(
                                infer.NUTS(self.model),
                                num_warmup=num_warmup,
                                num_samples=num_samples,
                                num_chains=num_chains,
                                progress_bar=progress_bar,
                            )
        self.sampler.run(jax.random.PRNGKey(3),image_error=self.weight_map,image=self.data)
        self.inf_data = az.from_numpyro(self.sampler)
        return az.summary(self.inf_data)

    def plot_bestfit(self):
        d = self.data
        bf = FitSersic.Sersic2D(self.xgrid,self.ygrid,**az.summary(self.inf_data).to_dict(),psf_fft=self.psf_fft)
        fig, ax = plt.subplots(1,3,figsize=(10,3),constrained_layout=True)
        ax[0].imshow(d,origin='lower',vmin=jnp.mean(d)-3*jnp.std(d),vmax=jnp.mean(d)+3*jnp.std(d))
        ax[1].imshow(bf,origin='lower',vmin=jnp.mean(bf)-3*jnp.std(bf),vmax=jnp.mean(bf)+3*jnp.std(bf))
        im2 = ax[2].imshow(d-bf,origin='lower',vmin=jnp.mean(d-bf)-3*jnp.std(d-bf),vmax=jnp.mean(d-bf)+3*jnp.std(d-bf))
        ax_divider = make_axes_locatable(ax[2])
        cax1 = ax_divider.append_axes("right", size="7%", pad="2%")
        cb1 = fig.colorbar(im2, cax=cax1)
        return fig, ax



    def optimize(self):
        optimizer = numpyro.optim.Adam(jax.example_libraries.optimizers.inverse_time_decay(1e-1, 500, 0.5, staircase=True) )
        guide = numpyro.infer.autoguide.AutoMultivariateNormal(self.model)
        svi = SVI(self.model, guide, optimizer, loss=Trace_ELBO(), )
        svi_result = svi.run(random.PRNGKey(1), 5000,image=self.data,image_error=self.weight_map)
        infodict = {'psf_fft':self.psf_fft,
                    'xgrid':self.xgrid,
                    'ygrid':self.ygrid,
                    'data':self.data}
        res = MAP(result=svi_result,svi=svi,guide=guide,infodict=infodict)
        return res 


class MAP():
    def __init__(self,result,svi,guide,infodict):
        self.result = result
        self.params = result.params 
        self.svi = svi 
        self.guide = guide 
        self.infodict = infodict
        
    def quantiles(self,quantiles):
        return self.guide.quantiles(self.params,quantiles)
    def mean_model(self):
        mean_params = self.quantiles([0.5])
        param_dict = {}
        for i in mean_params.keys():
            param_dict[i] = mean_params[i][0]
        bf = FitSersic.Sersic2D(self.infodict['xgrid'],self.infodict['ygrid'],**param_dict,psf_fft=self.infodict['psf_fft'])
        return bf, param_dict

    def plot_bestfit(self):
        bf, param_dict = self.mean_model() 
        d = self.infodict['data']
        fig, ax = plt.subplots(1,3,figsize=(10,3),constrained_layout=True)
        ax[0].imshow(d,origin='lower',vmin=jnp.mean(d)-3*jnp.std(d),vmax=jnp.mean(d)+3*jnp.std(d))
        ax[1].imshow(bf,origin='lower',vmin=jnp.mean(bf)-3*jnp.std(bf),vmax=jnp.mean(bf)+3*jnp.std(bf))
        im2 = ax[2].imshow(d-bf,origin='lower',vmin=jnp.mean(d-bf)-3*jnp.std(d-bf),vmax=jnp.mean(d-bf)+3*jnp.std(d-bf))
        ax_divider = make_axes_locatable(ax[2])
        cax1 = ax_divider.append_axes("right", size="7%", pad="2%")
        cb1 = fig.colorbar(im2, cax=cax1)
        return fig, ax

        


        