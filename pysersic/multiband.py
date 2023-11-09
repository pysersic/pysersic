
from pysersic.pysersic import BaseFitter, FitMulti,FitSingle
import abc
import numpy as np
import jax.numpy as jnp
import copy
from numpyro import distributions as dist, sample, deterministic, handlers, infer
import jax

from scipy.interpolate import make_interp_spline
from typing import List,Union,Optional
import arviz as az


class BaseMultiBandFitter(BaseFitter):
    def __init__(self, fitter_list: Union[List[FitSingle], List[FitMulti]],
                  wavelengths: jax.Array,
                  linked_params: List[str],
                  const_params: Optional[List[str]] = [],
                  band_names: Optional[List[str]] = None,
                  linked_params_range: Optional[dict] = {}, 
                  wv_to_save: Optional[jax.Array] = None,
                  rescale_unlinked_priors:Optional[bool] = False,) -> None:
        """Base class to multi-band fitting, to be used in other classes where the "linking" function is defined

        Parameters
        ----------
        fitter_list : List[FitSingle]
            List of FitSingle objects for all the different bands
        wavelengths : jax.Array
            The wavelengths of each band, They will be normalized so don't need to rescale
        band_names : Optional[List[str]], optional
            Names of bands, if None will default to 'band_0', 'band_1' etc.
        linked_params : Optional[List[str]], optional
            Which parameters to link across bands, by default None
        linked_params_range : Optional[dict], optional
            Ranges to enforce on any link parameters, by default {}
        wv_to_save : Optional[jax.Array], optional
            Additional wavelengths to track linking parameters for, by default None
        rescale_unlinked_priors: Optional[bool]:
            Whether or not to rescale priors of unlinked parameters based on fits to individual bands
        """
        self.fitter_list = copy.deepcopy(fitter_list)

        self.n_bands = len(fitter_list)
        assert len(wavelengths) == self.n_bands, "Number of fitters given does not much length of wavelength array"
        self.wavelengths = jnp.array(wavelengths)
        
        for fitter in self.fitter_list:
            assert isinstance(fitter, type(fitter_list[0])), "All fitters must be of same type"
        
        if isinstance(self.fitter_list[0], FitSingle):
            self.fitter_type = 'single'
        else:
            self.fitter_type = 'multi'
 
        if band_names is None:
            self.band_names = [f'Band_{i}' for i in range(len(self.wavelengths))]
        else:
            self.band_names = band_names
    
        # Make sure all priors have the same number of parameters
        self.param_names = self.fitter_list[0].prior.param_names
        for band_fitter in self.fitter_list:
            assert band_fitter.prior.param_names  == self.param_names, "All fitters must have the same parameters"

        for band,fitter in zip(self.band_names,self.fitter_list):
            fitter.prior.update_suffix(f'_{band}')

        # Normalize lamdda's to help convergence
        self.wv_av = np.mean(self.wavelengths)
        self.wv_range = np.max(self.wavelengths) - np.min(self.wavelengths)
        self.wv_normed = (self.wavelengths - self.wv_av)/ self.wv_range
        self.wv_to_save = wv_to_save
        if self.wv_to_save is not None:
            self.wv_to_save_normed = (self.wv_to_save - self.wv_av)/self.wv_range
        else:
            self.wv_to_save_normed = None


        # Set up which parameters are linked and which are not
        self.linked_params = linked_params
        for param in self.linked_params:
            assert param in self.param_names, f"Linked parameter ({param}) not in parameter list"
        
        self.const_params = const_params
        for param in self.const_params:
            assert param in self.param_names, f"const parameter ({param}) not in parameter list"

        self.unlinked_params = [p for p in self.param_names if (p not in self.linked_params and p not in self.const_params)]

        #Update reparam dict, linked parameters are handled in each subclass
        self.reparam_dict = {}
        for band_name, band_fitter in zip(self.band_names, self.fitter_list):
            for param_name in self.unlinked_params:
                val = band_fitter.prior.reparam_dict[f'{param_name}_{band_name}']
                self.reparam_dict.update( {f'{param_name}_{band_name}':val} )

        #Find mean and scale of prior samples of all bands to re-scale
        self.linked_params_mean = {}
        self.linked_params_scale = {}
        key = jax.random.key(self.n_bands)
        for param in self.linked_params:
                params_prior_samples = []
                for fitter, band in zip(self.fitter_list, self.band_names):
                    key, _ = jax.random.split(key)
                    dist_cur = fitter.prior.dist_dict[f'{param}_{band}']
                    params_prior_samples.append( dist_cur.sample(key, sample_shape = (200,)) )
                params_prior_samples = jnp.stack(params_prior_samples)

                self.linked_params_mean[param] = params_prior_samples.mean()
                self.linked_params_scale[param] = params_prior_samples.std()


        #Set defaults for all parameters that have known limits
        self.linked_params_range = {}
        for param in self.linked_params:
            if 'n' in param:
                self.linked_params_range[param] = [0.65,8]
            if 'ellip' in param:
                self.linked_params_range[param] = [0.,0.9]
            if 'theta' in param:
                self.linked_params_range[param] = [0,2*np.pi]

        self.linked_params_range.update(linked_params_range) #User-specific updates


        #For constant parameters use the prior in the of the first band
        self.const_prior_dict = {}
        for const_param in self.const_params:
            param_in_band_name = f'{const_param}_{self.band_names[0]}'
            prior_dist = copy.copy(self.fitter_list[0].prior.dist_dict[param_in_band_name] )
            self.const_prior_dict[const_param] = copy.deepcopy(prior_dist)
            
            if param_in_band_name in self.fitter_list[0].prior.reparam_dict:
                self.reparam_dict[const_param] = self.fitter_list[0].prior.reparam_dict[param_in_band_name]
        

        # Use estimated posterior of each individual band to set priors on unlinked parameters
        # Since these are "nuisance" in some sense that should help convergence etc. when sampling
        if rescale_unlinked_priors:
            for fitter in self.fitter_list:
                if hasattr(fitter, 'svi_results'):

                        svi_summ_no_suffix = az.summary(fitter.svi_results.idata, round_to=5).T.to_dict()
                        svi_summ = {}
                        for key in svi_summ_no_suffix:
                            svi_summ[f'{key}{fitter.prior.suffix}'] = svi_summ_no_suffix[key]
                else:    
                    svi_res = fitter.estimate_posterior()
                    svi_summ = az.summary(svi_res.idata, round_to=5)

    
                    
                for key, summ_param in svi_summ.items():
                    param = key.replace(fitter.prior.suffix, '')
                    if param in self.unlinked_params:
                        prior_dist = fitter.prior.dist_dict[key]

                        #Check if prior dist is transformed, most likely yes
                        is_t = isinstance(prior_dist, dist.TransformedDistribution)
                        if is_t:
                            base_dist = prior_dist.base_dist
                        else:
                            base_dist = prior_dist
                        
                        #Check if there are bounds on prior
                        if isinstance(base_dist.support,dist.constraints._Real):
                            fitter.prior.set_gaussian_prior(param, summ_param['mean'], summ_param['sd']*2.)
                        else:
                            #If so then use a truncated distribution
                            low = base_dist.support.lower_bound
                            high = base_dist.support.upper_bound
                            
                            #If there are transforms then apply them
                            if is_t:
                                for t in prior_dist.transforms:
                                    low = t(low)
                                    high = t(high)
                                
                            fitter.prior.set_truncated_gaussian_prior(param, summ_param['mean'], summ_param['sd']*2., low = low, high = high)
                
                if fitter.prior.sky_type != 'none':
                    sky_params = [param for param in svi_summ.keys() if 'sky' in param]
                    for param in sky_params:
                        param_no_suffix = param.replace(fitter.prior.suffix, '')
                        fitter.prior.sky_prior.update_prior(param_no_suffix, svi_summ[param]['mean'], svi_summ[param]['sd']*2.)
        ##Dummy variables until we figure out what to do
        self.data = 0.
        self.rms = 0.
        self.psf = 0.
        self.mask = 0.
        self.loss_func = 0.
        self.renderer = 0.
        self.prior = 0.
        print ('return model')
    
    @abc.abstractmethod
    def sample_param_at_bands(self, name):
        return NotImplementedError

    def build_model(self,return_model : bool = False):
        @handlers.reparam(config = self.reparam_dict)
        def model():
            #Draw heriarchichal parameters for each linked parameter
            params_dict = {}
            for name in self.band_names:
                params_dict[name] = {}
            
            for param_name in self.linked_params:
                x = self.sample_param_at_bands(param_name)
                for i,band_name in enumerate(self.band_names):
                    params_dict[band_name][param_name] = deterministic(f'{param_name}_{band_name}', x[i])

            for param, prior_dist in self.const_prior_dict.items():
                value = sample(param, prior_dist)
                for band in self.band_names:
                    params_dict[band][param] = value

            all_obs = []
            for band_name,band_fitter in zip(self.band_names, self.fitter_list):
                #Draw values for unlinked parameters based on individual priors (with suffix_names)
                for param_name in self.unlinked_params:
                    params_dict[band_name][param_name] = sample(f'{param_name}_{band_name}', band_fitter.prior.dist_dict[f'{param_name}_{band_name}'])
                
                #Render each band like normal
                if self.fitter_type == 'single':
                    out = band_fitter.renderer.render_source(params_dict[band_name],
                                                             band_fitter.prior.profile_type,
                                                             suffix = "")
                else:
                    out = band_fitter.renderer.render_for_model(params_dict[band_name],
                                                                band_fitter.prior.catalog['type'],
                                                                suffix = "")


                sky = band_fitter.prior.sample_sky(band_fitter.renderer.X, band_fitter.renderer.Y)
                obs = out + sky
                all_obs.append(obs)
                band_fitter.loss_func(obs, band_fitter.data, band_fitter.rms, band_fitter.mask, suffix = f'_{band_name}')
            if return_model:
                deterministic('model', jnp.array(all_obs))
        return model
    

class FitMultiBandPoly(BaseMultiBandFitter):
    def __init__(self, 
                 fitter_list: List[FitSingle],
                 wavelengths: jax.Array,
                 linked_params: List[str],
                 const_params: Optional[List[str]] = [],
                 poly_order: Optional[int] = 3,
                 band_names: List[str] | None = None,
                 linked_params_range: dict | None = {},
                 wv_to_save: jax.Array | None = None,
                 rescale_unlinked_priors: Optional[bool] = False,) -> None:

        super().__init__(fitter_list, wavelengths, linked_params,const_params, band_names, linked_params_range, wv_to_save, rescale_unlinked_priors=rescale_unlinked_priors)
        self.poly_order = poly_order

    def restrict_func(self,x, hi,low):
        return  jax.lax.logistic(x) *(hi-low) + low

    def sample_param_at_bands(self, name):
        prior_sigma = 1.

        poly_coeff = sample(f'{name}_poly_coeff',dist.Normal(scale=prior_sigma), sample_shape=(self.poly_order,))
        value_at_bands_normed = jnp.polyval(poly_coeff, self.wv_normed)
        value_to_save_normed = jnp.polyval(poly_coeff, self.wv_to_save_normed)

        if name in self.linked_params_range:
            low,hi = self.linked_params_range[name]
            
            # Use Tanh to make sure parameters are constrained
            value_at_bands = self.restrict_func(value_at_bands_normed,hi,low)
            
            deterministic(f'{name}_at_wv', 
                            self.restrict_func(value_to_save_normed,hi,low)
            )
        else:
            value_at_bands = value_at_bands_normed*self.linked_params_scale[name] + self.linked_params_mean[name]

            deterministic(f'{name}_at_wv', 
                            value_to_save_normed*self.linked_params_scale[name] + self.linked_params_mean[name]
            )

        return value_at_bands


class FitMultiBandBSpline(BaseMultiBandFitter):
    def __init__(self, 
                 fitter_list: List[FitSingle],
                 wavelengths: jax.Array,
                 linked_params: List[str],
                 const_params: Optional[List[str]] = [],
                 band_names: List[str] | None = None,
                 linked_params_range: dict | None = {},
                 wv_to_save: jax.Array | None = None,
                 rescale_unlinked_priors: Optional[bool] = False,
                 N_knots = 4,
                 spline_k = 2,
                 pad_knots:bool = True,
                ) -> None:

        super().__init__(fitter_list, wavelengths, linked_params,const_params, band_names, linked_params_range, wv_to_save, rescale_unlinked_priors=rescale_unlinked_priors)
        self.N_knots = N_knots
        self.spline_k = spline_k
        min_lambda, max_lambda = jnp.min(self.wavelengths), jnp.max(self.wavelengths)
        
        #Pad knots by 10% on either side, leads to more uniform "importance"
        if pad_knots:
            lambda_pad = (max_lambda - min_lambda)/10.
        else:
            lambda_pad = 0
        
        #Set up spline matrix
        bspl_class = make_interp_spline(x = np.linspace(min_lambda-lambda_pad, max_lambda + lambda_pad, num = self.N_knots, endpoint=True),
                                                y = np.ones(self.N_knots), k = spline_k)
        self.dmat_bands = jnp.array( bspl_class.design_matrix(self.wavelengths, bspl_class.t,k = spline_k).toarray() )
        wv_to_save_dmat = jnp.clip(self.wv_to_save, a_min = min_lambda, a_max=max_lambda)
        self.dmat_save = jnp.array( bspl_class.design_matrix(wv_to_save_dmat, bspl_class.t,k = spline_k).toarray() )
        print ('pad knot loc, ps 1.5')
    
    def sample_param_at_bands(self, name):
        prior_sigma = 1.5
        m,s = self.linked_params_mean[name], self.linked_params_scale[name]

        if name in self.linked_params_range:
            low,hi = self.linked_params_range[name]
            
            dist_to_sample = dist.TransformedDistribution(
                dist.Uniform(low=jnp.zeros(self.N_knots), high = jnp.ones(self.N_knots)),
                dist.transforms.AffineTransform(scale=hi-low, loc = low)
            )

        else:
            dist_to_sample = dist.TransformedDistribution(
                dist.Normal(scale = prior_sigma*jnp.ones(self.N_knots) ),
                dist.transforms.AffineTransform(scale=s, loc = m)
            )

        with handlers.reparam(config={f'bspl_w_{name}': infer.reparam.TransformReparam()}):
            weights_cur = sample(f'bspl_w_{name}',dist_to_sample)
        
        value_at_bands = jnp.dot(self.dmat_bands,weights_cur)
        
        deterministic(f'{name}_at_wv', 
                        jnp.dot(self.dmat_save,weights_cur)
        )

        return value_at_bands

