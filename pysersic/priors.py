
from numpyro import distributions as dist, infer, sample
import jax.numpy as jnp 
import jax
import pandas
import numpy as np
from typing import Union, Optional, Iterable
from abc import ABC
from .utils.utils import render_tilted_plane_sky
from photutils.morphology import data_properties
import astropy.units as u
import matplotlib.pyplot as plt 

base_sky_types = ['none','flat','tilted-plane']
base_sky_params = dict(
    zip(base_sky_types,
    [ [],['sky_back',], ['sky_back','sky_x_sl','sky_y_sl'] ]
    )
)

base_profile_types = ['sersic','doublesersic','pointsource','exp','dev']
base_profile_params =dict( 
    zip(base_profile_types,
    [ ['xc','yc','flux','r_eff','n','ellip','theta'],
    ['xc','yc','flux','f_1', 'r_eff_1','n_1','ellip_1', 'r_eff_2','n_2','ellip_2','theta'],
    ['xc','yc','flux'],
    ['xc','yc','flux','r_eff','ellip','theta'],
    ['xc','yc','flux','r_eff','ellip','theta'],]
    )
)


class BasePrior(ABC):
    """
    Base class for priors with sky sampling included
    """
    def __init__(self, sky_type = 'none') -> None:
        """Initialize a base prior class

        Parameters
        ----------
        sky_type : str, optional
            Type of sky modle to use, one of: none,flat or tilted-plane, by default 'none'
        """
        self.reparam_dict = {}
        self.sky_type = sky_type
        if self.sky_type not in base_sky_types:
            raise AssertionError("Sky type must be one of: ", base_sky_types)
        elif self.sky_type == 'none':
            self.sample_sky = self.sample_sky_none
    
        elif self.sky_type == 'flat':
            self._set_dist('sky_back',dist.TransformedDistribution(
                                dist.Normal(),
                                dist.transforms.AffineTransform(0.0,1e-3),)
                            )
            self.reparam_dict['sky_back'] = infer.reparam.TransformReparam()
            self.sample_sky = self.sample_sky_flat
            
        elif self.sky_type == 'tilted-plane':
            self._set_dist('sky_back',dist.TransformedDistribution(
                                dist.Normal(),
                                dist.transforms.AffineTransform(0.0,1e-3),)
                            )
            self._set_dist('sky_x_sl',  dist.TransformedDistribution(
                                dist.Normal(),
                                dist.transforms.AffineTransform(0.0,1e-4),)
            )

            self._set_dist('sky_y_sl',dist.TransformedDistribution(
                                dist.Normal(),
                                dist.transforms.AffineTransform(0.0,1e-4),)
            )
            
            self.reparam_dict['sky_back'] = infer.reparam.TransformReparam()
            self.reparam_dict['sky_x_sl'] = infer.reparam.TransformReparam()
            self.reparam_dict['sky_y_sl'] = infer.reparam.TransformReparam()
            self.sample_sky = self.sample_sky_tilted_plane

    def sample_sky_none(self,X: jax.numpy.array,Y: jax.numpy.array)-> float:
        """Simple wrapper to generate no sky

        Parameters
        ----------
        X : jax.numpy.array
            2d array, X pixel values
        Y : jax.numpy.array
            2d array, Y pixel values

        Returns
        -------
        float = 0
            returns 0
        """
        return 0.
    
    def sample_sky_flat(self,X: jax.numpy.array,Y: jax.numpy.array)-> float:
        """
        Sample and generate a flat sky background

        Parameters
        ----------
        X : jax.numpy.array
            2d array, X pixel values
        Y : jax.numpy.array
            2d array, Y pixel values

        Returns
        -------
        float
            sampled background value
        """
        sky_back = sample('sky_back', self.sky_back_prior_dist)
        return sky_back
    
    def sample_sky_tilted_plane(self,X: jax.numpy.array,Y: jax.numpy.array)-> jax.numpy.array:
        """
        Sample and generate a tilted-plane sky background

        Parameters
        ----------
        X : jax.numpy.array
            2d array, X pixel values
        Y : jax.numpy.array
            2d array, Y pixel values

        Returns
        -------
        jax.numpy.array
            renderned sky background
        """
        sky_back = sample('sky_back', self.sky_back_prior_dist)
        sky_x_sl = sample('sky_x_sl', self.sky_x_sl_prior_dist)
        sky_y_sl = sample('sky_y_sl', self.sky_y_sl_prior_dist)
        return render_tilted_plane_sky(X,Y, sky_back,sky_x_sl,sky_y_sl)

    def _set_dist(self, var_name: str, dist: dist.Distribution)-> None:
        """Set prior for a given variable

        Parameters
        ----------
        var_name : str
            variable name
        dist : dist.Distribution
            Numpyro distribution object specifying prior
        """
        self.__setattr__(var_name+'_prior_dist', dist)

    def _get_dist(self, var_name: str)-> dist.Distribution:
        """
        Get prior for a given variable

        Parameters
        ----------
        var_name : str
            variable name

        Returns
        -------
        dist.Distribution
            Numpyro distribution clas describing prior
        """
        return self.__getattribute__(var_name +'_prior_dist')
    

class PySersicSourcePrior(BasePrior):
    """
    Class used for priors for single source fitting in PySersic
    """
    def __init__(self, profile_type: str, sky_type: Optional[str] = 'none',suffix: Optional[str] =  "") -> None:
        """Initialize PySersicSourcePrior class

        Parameters
        ----------
        profile_type : str
            Type of profile
        sky_type : Optional[str], optional
            Type of sky model to use, one of: none, flat, tilted-plane, by default 'none'
        suffix : Optional[str], optional
            Additional suffix to add to each variable name, used in PySersicMultiPrior, by default ""
        """
        super().__init__(sky_type= sky_type)
        assert profile_type in base_profile_types
        self.profile_type = profile_type
        self.param_names = base_profile_params[self.profile_type]
        self.repr_dict = {}
        self.built = False
        self.suffix = suffix

    def __repr__(self) -> str:
        out = f"Prior for a {self.profile_type} source:"
        num_dash = len(out)
        out += "\n" + "-"*num_dash + "\n"
        for (var, descrip) in self.repr_dict.items():
            out += var + " ---  " + descrip + "\n"    
        return out
    
    def _build_dist_list(self)-> None:
        """
        Function to combine all distributions into list, sets self.dist_list
        """
        self.dist_list = []
        assert self.check_vars() # check and make sure all is good

        for param in self.param_names:
            self.dist_list.append(self._get_dist(param+self.suffix))
        
        self.built = True
    
    def __call__(self) -> jax.numpy.array:
        """
        Sample variables from prior

        Returns
        -------
        jax.numpy.array
            sampled variables
        """
        if not self.built: 
            self._build_dist_list()

        arr = []
        for (param,prior) in zip(self.param_names,self.dist_list):
            if issubclass(type(prior), dist.Distribution):
                arr.append(sample(param+self.suffix, prior) )
            else:
                arr.append(prior)
        return jnp.array(arr)      

    def set_gaussian_prior(self,var_name: str, loc: float, scale: float) -> "PySersicSourcePrior":
        """
        Set a Gaussian prior for a variable

        Parameters
        ----------
        var_name : str
            variable name
        loc : float
            mean 
        scale : float
            standard deviation

        Returns
        -------
        PySersicSourcePrior
            returns self to allow chaining
        """

        prior_dist = dist.TransformedDistribution(
            dist.Normal(),
            dist.transforms.AffineTransform(loc,scale),)
        
        self._set_dist(var_name + self.suffix, prior_dist)
        self.reparam_dict[var_name  + self.suffix] = infer.reparam.TransformReparam()
        self.repr_dict[var_name] = f"Normal w/ mu = {loc:.2f}, sigma = {scale:.2f}"
        return self
    
    def set_uniform_prior(self, var_name:str, low: float,high: float)-> "PySersicSourcePrior":
        """
        Set a uniform prior for a variable

        Parameters
        ----------
        var_name : str
            variable name
        low : float
            lower bound
        high : float
            upper bound

        Returns
        -------
        PySersicSourcePrior
            returns self to allow chaining

        """
        shift = low
        scale = high-low
        prior_dist = dist.TransformedDistribution(
            dist.Uniform(),
            dist.transforms.AffineTransform(shift,scale),)
        self._set_dist(var_name + self.suffix, prior_dist)
        self.reparam_dict[var_name + self.suffix] = infer.reparam.TransformReparam()
        self.repr_dict[var_name] = f"Uniform between: {low:.2f} -> {high:.2f}"
        return self
    
    def set_truncated_gaussian_prior(self,
            var_name:str, 
            loc: float,
            scale:float, 
            low: Optional[float] = None,
            high: Optional[float] = None) -> "PySersicSourcePrior":
        """
        Set a truncated Gaussian prior for a given variable

        Parameters
        ----------
        var_name : str
            variable name
        loc : float
            mean
        scale : float
            standard deviation
        low : Optional[float], optional
            lower bound, by default None
        high : Optional[float], optional
            upper bound, by default None

        Returns
        -------
        PySersicSourcePrior
            Returns self to allow chaining
        """
        if low is not None:
            low_scaled = (low - loc)/scale
        else:
            low = -jnp.inf
            low_scaled = None
        
        if high is not None:
            high_scaled = (high - loc)/scale
        else:
            high_scaled = None
            high = jnp.inf
        prior_dist = dist.TransformedDistribution(
            dist.TruncatedNormal(low= low_scaled, high = high_scaled),
            dist.transforms.AffineTransform(loc,scale),)
    
        self._set_dist(var_name + self.suffix, prior_dist)
        self.reparam_dict[var_name + self.suffix] = infer.reparam.TransformReparam()
        self.repr_dict[var_name] = f"Truncated Normal w/ mu = {loc:.2f}, sigma = {scale:.2f}, between: {low:.2f} -> {high:.2f}"
        return self

    def set_custom_prior(self, 
            var_name: str, 
            prior_dist: dist.Distribution, 
            reparam: Optional[infer.reparam.Reparam] = None)-> "PySersicSourcePrior":
        """Set a custom distribution as the prior for a given variable

        Parameters
        ----------
        var_name : str
            variable name
        prior_dist : dist.Distribution
            Numpyro Distribution object describing prior
        reparam : Optional[infer.reparam.Reparam], optional
            Optional reparamaterization to use for variable, by default None

        Returns
        -------
        PySersicSourcePrior
            Returns self to allow chaining
        """
        self._set_dist(var_name + self.suffix, prior_dist)
        if reparam is not None:
            self.reparam_dict[var_name + self.suffix] = reparam
        self.repr_dict[var_name] = "Custom prior of type: "+ str(prior_dist.__class__)
        return self
    
    def check_vars(self, verbose = False) -> bool:
        """
        Function to check if all variable for the specified profile type are set with no extras

        Parameters
        ----------
        verbose : bool, optional
            Wheter to print out missing and extra variables, by default False

        Returns
        -------
        bool
            True if all variable for given type are present with no extra, False otherwise
        """
        missing = []
        for var in self.param_names:
            if not hasattr(self, f"{var + self.suffix}_prior_dist"):
                missing.append(var)
        
        extra = []
        for (name, descrip) in self.repr_dict.items():
            if (name not in self.param_names) and ('sky' not in name):
                extra.append(name)
        if verbose:
            print ("Missing params for profile: ", missing)
            print ("Extra params, will not be used: ", extra)
        
        if len(missing)== 0 and len(extra)==0:
            out = True
        else:
            out = False
        return out
    
        
class PySersicMultiPrior(BasePrior):
    """
    Class used for priors for multi source fitting in PySersic
    """
    def __init__(self, 
            catalog: Union[pandas.DataFrame,dict, np.recarray],
            sky_type: Optional[str] = 'none',     
            )-> None:
        """
        Ingest a catalog-like data structure containing prior positions and parameters for multiple sources in a single image. The format of the catalog can be a `pandas.DataFrame`, `numpy` RecordArray, dictionary, or any other format so-long as the following fields exist and can be directly indexed: 'x', 'y', 'flux', 'r' and 'type'

        Parameters
        ----------
        image : jax.numpy.array
            science image
        catalog : Union[pandas.DataFrame,dict, np.recarray]
            Object containing information about the sources to be fit
        Returns
        -------
        prior_list : Iterable
            List containing a prior dictionary for each source
        """

        super().__init__(sky_type = sky_type)
        self.catalog = catalog
        self.all_priors = []
        self.N_sources = len(catalog['x'])
        image = jnp.ones((100,100)) # dummy image
        for ind in range(len(catalog['x'])):

            init = dict(flux_guess = catalog['flux'][ind], r_eff_guess = catalog['r'][ind], position_guess = (catalog['x'][ind],catalog['y'][ind]) )

            if catalog['type'][ind] == 'sersic':
                prior = generate_sersic_prior(image,suffix = f'_{ind:d}', **init)
            
            elif catalog['type'][ind] == 'doublesersic':
                prior = generate_doublesersic_prior(image,suffix = f'_{ind:d}', **init)

            elif catalog['type'][ind] == 'pointsource':
                init.pop('r_eff_guess')
                prior = generate_pointsource_prior(image,suffix = f'_{ind:d}', **init)

            elif catalog['type'][ind] == 'exp':
                prior = generate_exp_prior(image,suffix = f'_{ind:d}', **init)
            
            elif catalog['type'][ind] == 'dev':
                prior = generate_dev_prior(image,suffix = f'_{ind:d}', **init)
        
            self.all_priors.append(prior)
            self.reparam_dict.update(prior.reparam_dict)
    
    def __repr__(self,)-> str:
        out = f"PySersicMultiPrior containing {len(self.all_priors):d} sources \n"
        for i, source_prior in enumerate(self.all_priors):
            out += f"\nSource {i:d} : " + source_prior.__repr__()
        return out
    
    def __call__(self) -> list:
        """Sample prior for all sources

        Returns
        -------
        list
            a list of jax.numpy.arrays contraining sampled variables for each source
        """
        all_params = []
        for prior_cur in self.all_priors:
            all_params.append(prior_cur())
        return all_params

def autoprior(image_properties,
            profile_type: str,
            mask: Optional[jax.numpy.array] = None,
            sky_type: Optional[str] = 'none')-> PySersicSourcePrior:
    """Function to generate default priors based on a given image and profile type

    Parameters
    ----------
    image : jax.numpy.array
        Masked image
    profile_type : str
        Type of profile
    sky_type : str, default 'none'
        Type of sky model to use, must be one of: 'none', 'flat', 'tilted-plane'
    Returns
    -------
    dict
        Dictionary containing numpyro Distribution objects for each parameter
    """
    image_properties = ImageProperties(image,mask)
    if profile_type == 'sersic':
        prior_dict = generate_sersic_prior(image_properties, sky_type = sky_type)
    
    elif profile_type == 'doublesersic':
        prior_dict = generate_doublesersic_prior(image_properties, sky_type = sky_type)

    elif profile_type == 'pointsource':
        prior_dict = generate_pointsource_prior(image_properties, sky_type = sky_type)

    elif profile_type in 'exp':
        prior_dict = generate_exp_prior(image_properties, sky_type = sky_type)

    elif profile_type in 'dev':
        prior_dict = generate_dev_prior(image_properties, sky_type = sky_type)
    
    return prior_dict


def generate_sersic_prior(image_properties,
                        sky_type: Optional[str] = 'none',
                        suffix: Optional[str] = '')-> PySersicSourcePrior:
    """ Derive automatic priors for a sersic profile based on an input image.

    Parameters
    ----------
    image_properties: ImageProperties
        class containing the guesses for the various properties
    sky_type : str, default 'none'
        Type of sky model to use, must be one of: 'none', 'flat', 'tilted-plane'
    suffix: str, default ''
        suffix to add to the parameter (e.g., for multi source fitting)
    Returns
    -------
    dict
        Dictionary containing numpyro Distribution objects for each parameter

    """
    prior = PySersicSourcePrior('sersic', sky_type= sky_type, suffix=suffix)
    prior.set_gaussian_prior('flux',image_properties.flux_guess,image_properties.flux_guess_err)
    prior.set_truncated_gaussian_prior('r_eff', image_properties.r_eff_guess,image_properties.r_scale, low = 0.5)
    prior.set_uniform_prior('ellip', 0, 0.9)
    prior.set_custom_prior('theta', dist.VonMises(loc = image_properties.theta_guess,concentration=2), reparam= infer.reparam.CircularReparam() )
    prior.set_uniform_prior('n', 0.5, 8)
    prior.set_gaussian_prior('xc', image_properties.xc_guess, 1)
    prior.set_gaussian_prior('yc', image_properties.yc_guess, 1)

    return prior 

def generate_exp_prior(image_properties,
        sky_type: Optional[str] = 'none',
        suffix: Optional[str] = '')-> PySersicSourcePrior:
    """ Derive automatic priors for a exp or dev profile based on an input image.
    
    Parameters
    ----------
    image_properties: ImageProperties
        class containing the guesses for the various properties
    sky_type : str, default 'none'
        Type of sky model to use, must be one of: 'none', 'flat', 'tilted-plane'
    
    Returns
    -------
    dict
        Dictionary containing numpyro Distribution objects for each parameter

    """
    
    prior = PySersicSourcePrior('exp', sky_type= sky_type, suffix=suffix)
    prior.set_gaussian_prior('flux',image_properties.flux_guess,image_properties.flux_guess_err)
    prior.set_truncated_gaussian_prior('r_eff', image_properties.r_eff_guess,image_properties.r_scale, low = 0.5)
    prior.set_uniform_prior('ellip', 0,0.9)
    prior.set_custom_prior('theta', dist.VonMises(loc = image_properties.theta_guess,concentration=2), reparam= infer.reparam.CircularReparam() )
    prior.set_gaussian_prior('xc', image_properties.xc_guess, 1)
    prior.set_gaussian_prior('yc', image_properties.yc_guess, 1)

    return prior

def generate_dev_prior(image_properties,
        sky_type: Optional[str] = 'none',
        suffix: Optional[str] = '')-> PySersicSourcePrior:
    """ Derive automatic priors for a exp or dev profile based on an input image.
    
    Parameters
    ----------
    image_properties: ImageProperties
        class containing the guesses for the various properties
    sky_type : str, default 'none'
        Type of sky model to use, must be one of: 'none', 'flat', 'tilted-plane'
    
    Returns
    -------
    dict
        Dictionary containing numpyro Distribution objects for each parameter

    """
    
    prior = PySersicSourcePrior('dev', sky_type= sky_type, suffix=suffix)
    prior.set_gaussian_prior('flux',image_properties.flux_guess,image_properties.flux_guess_err)
    prior.set_truncated_gaussian_prior('r_eff', image_properties.r_eff_guess,image_properties.r_scale, low = 0.5)
    prior.set_uniform_prior('ellip', 0,0.9)
    prior.set_custom_prior('theta', dist.VonMises(loc = image_properties.theta_guess,concentration=2), reparam= infer.reparam.CircularReparam() )
    prior.set_gaussian_prior('xc', image_properties.xc_guess, 1)
    prior.set_gaussian_prior('yc', image_properties.yc_guess, 1)

    return prior

def generate_doublesersic_prior(image_properties,
        sky_type: Optional[str] = 'none',
        suffix: Optional[str] = '')-> PySersicSourcePrior:
    """ Derive automatic priors for a double sersic profile based on an input image.
    
    Parameters
    ----------
    image_properties: ImageProperties
        class containing the guesses for the various properties
    sky_type : str, default 'none'
        Type of sky model to use, must be one of: 'none', 'flat', 'tilted-plane'
        
    Returns
    -------
    dict
        Dictionary containing numpyro Distribution objects for each parameter

    """

    prior = PySersicSourcePrior('doublesersic', sky_type= sky_type, suffix=suffix)
    prior.set_gaussian_prior('flux',image_properties.flux_guess,image_properties.flux_guess_err)
    prior.set_uniform_prior('f_1', 0.,1.)

    prior.set_custom_prior('theta', dist.VonMises(loc = image_properties.theta_guess,concentration=2), reparam= infer.reparam.CircularReparam() )

    r_loc1 = image_properties.r_eff_guess/1.5
    r_scale1 = jnp.sqrt(image_properties.r_eff_guess/1.5)
    prior.set_truncated_gaussian_prior('r_eff_1', r_loc1,r_scale1, low = 0.5)

    r_loc2 = image_properties.r_eff_guess*1.5
    r_scale2 = jnp.sqrt(image_properties.r_eff_guess*1.5)
    prior.set_truncated_gaussian_prior('r_eff_2', r_loc2,r_scale2, low = 0.5)


    prior.set_uniform_prior('ellip_1', 0,0.9)
    prior.set_uniform_prior('ellip_2', 0,0.9)

    prior.set_truncated_gaussian_prior('n_1',4,1, low = 0.5,high = 8)
    prior.set_truncated_gaussian_prior('n_2',1,1, low = 0.5,high = 8)

    prior.set_gaussian_prior('xc', image_properties.xc_guess, 1)
    prior.set_gaussian_prior('yc', image_properties.yc_guess, 1)


    return prior

def generate_pointsource_prior(image_properties,
        sky_type: Optional[str] = 'none',
        suffix: Optional[str] = '')-> PySersicSourcePrior:
    """ Derive automatic priors for a pointsource based on an input image.
    
    Parameters
    ----------
    image_properties: ImageProperties
        class containing the guesses for the various properties
    sky_type : str, default 'none'
        Type of sky model to use, must be one of: 'none', 'flat', 'tilted-plane'
        
    Returns
    -------
    dict
        Dictionary containing numpyro Distribution objects for each parameter

    """

    prior = PySersicSourcePrior('pointsource' , sky_type= sky_type, suffix=suffix)
    
    prior.set_gaussian_prior('flux',image_properties.flux_guess,image_properties.flux_guess_err)
    

    prior.set_gaussian_prior('xc', image_properties.xc_guess, 1)
    prior.set_gaussian_prior('yc', image_properties.yc_guess, 1)

    return prior



class ImageProperties():
    def __init__(self,image,mask=None):
        self.image = image 
        self.mask = mask 
        if self.mask is not None:
            self.cat = data_properties(self.image,mask=self.mask.astype(bool))
        else:
            self.cat = data_properties(self.image)
    
    def visualize(self,figsize=(6,6),cmap='gray',scale=1):
        if not hasattr(self,'flux_guess'):
            self.set_flux_guess() 
        if not hasattr(self,'r_eff_guess'):
            self.set_r_eff_guess()
        if not hasattr(self,'theta_guess'):
            self.set_theta_guess() 
        if not hasattr(self,'xc_guess'):
            self.set_position_guess()
        
        fig, ax = plt.subplots(figsize=figsize,)
        if self.mask is not None:
            image= np.ma.masked_array(self.image,mask=self.mask)
        else:
            image = self.image
        vmin = np.mean(image) - scale*np.std(image)
        vmax = np.mean(image) + scale*np.std(image)
        ax.imshow(image,cmap=cmap,origin='lower',vmin=vmin,vmax=vmax)
        ax.plot(self.xc_guess,self.yc_guess,'x',color='r')
        
        dx = 10*self.r_eff_guess*np.cos(self.theta_guess+np.pi/2.)
        dy = 10*self.r_eff_guess*np.sin(self.theta_guess+np.pi/2.)
        ax.arrow(self.xc_guess,self.yc_guess,dx=dx,dy=dy,width=0.1,head_width=1)
        arr = np.linspace(0,2*np.pi,100)
        x = np.cos(arr) * self.r_eff_guess + self.xc_guess 
        y = np.sin(arr) * self.r_eff_guess + self.yc_guess 
        ax.plot(x,y,'r',lw=2)
        plt.show() 


        

    def measure_properties(self,**kwargs):
        self.set_flux_guess(**kwargs)
        self.set_r_eff_guess(**kwargs)
        self.set_theta_guess(**kwargs)
        self.set_position_guess(**kwargs)

    def set_flux_guess(self,flux_guess=None,flux_guess_err = None,**kwargs):
        if flux_guess is None:
            flux_guess = self.cat.segment_flux
        if flux_guess_err is not None:
            flux_guess_err = flux_guess_err
        else:
            if flux_guess > 0:
                flux_guess_err = 2*jnp.sqrt( flux_guess )
            else:
                flux_guess_err = jnp.sqrt(jnp.abs(flux_guess))
                flux_guess = 0.
        
        self.flux_guess = flux_guess 
        self.flux_guess_err = flux_guess_err
    
    def set_r_eff_guess(self,r_eff_guess=None,**kwargs):
        if r_eff_guess is None:
            r_eff_guess = self.cat.kron_radius.value
    
        self.r_eff_guess = r_eff_guess
        self.r_scale = jnp.sqrt(r_eff_guess) 

    def set_theta_guess(self,theta_guess=None,**kwargs):
        if theta_guess is None:
            theta_guess = self.cat.orientation.to(u.rad).value

        if np.isnan(theta_guess):
            theta_guess = 0
        self.theta_guess = theta_guess 
    
    def set_position_guess(self,position_guess=None,**kwargs):
        if position_guess is None:
            self.xc_guess = self.cat.centroid_win[0]
            self.yc_guess = self.cat.centroid_win[1]
        else:
            self.xc_guess = position_guess[0]
            self.yc_guess = position_guess[1]

    def autoprior(self,
                profile_type: str,
                sky_type: Optional[str] = 'none')-> PySersicSourcePrior:
        """Function to generate default priors based on a given image and profile type

        Parameters
        ----------
        image : jax.numpy.array
            Masked image
        profile_type : str
            Type of profile
        sky_type : str, default 'none'
            Type of sky model to use, must be one of: 'none', 'flat', 'tilted-plane'
        Returns
        -------
        dict
            Dictionary containing numpyro Distribution objects for each parameter
        """
        if profile_type == 'sersic':
            prior_dict = generate_sersic_prior(self, sky_type = sky_type)
        
        elif profile_type == 'doublesersic':
            prior_dict = generate_doublesersic_prior(self, sky_type = sky_type)

        elif profile_type == 'pointsource':
            prior_dict = generate_pointsource_prior(self, sky_type = sky_type)

        elif profile_type in 'exp':
            prior_dict = generate_exp_prior(self, sky_type = sky_type)

        elif profile_type in 'dev':
            prior_dict = generate_dev_prior(self, sky_type = sky_type)
        
        return prior_dict
