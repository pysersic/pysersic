
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Union, Tuple

import astropy.units as u
from astropy.stats import biweight_scale as bws
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas
from numpyro import distributions as dist, infer, sample
from photutils.morphology import data_properties
from copy import copy

base_sky_types = ['none','flat','tilted-plane']
base_sky_params = dict(
    zip(base_sky_types,
    [ [],['sky_back',], ['sky_back','sky_x_sl','sky_y_sl'] ]
    )
)

@jax.jit
def render_tilted_plane_sky(X,Y,back,x_sl,y_sl ):
    xmid = float(X.shape[0]/2.)
    ymid = float(Y.shape[0]/2.)
    return back + (X-xmid)*x_sl + (Y-ymid)*y_sl


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

class BaseSkyPrior(ABC):
    def __init__(self,sky_guess: float, sky_guess_err: float, suffix: str = ''):
        """Base class for sky priors

        Parameters
        ----------
        sky_guess : float
            Initial guess for level of background, by default None
        sky_guess_err : float
            Uncertainty on initial guess, by default None
        suffix : str, optional
            suffix to be added on end of variables, by default ''
        """
        self.sky_guess = sky_guess
        self.sky_guess_err = sky_guess_err
        self.reparam_dict = {}
        self.dist_dict = {}
        self.suffix = suffix
        self.type = 'Base'
        self.repr_dict = {}
    
    @abstractmethod
    def sample(self,X: jnp.array,Y: jnp.array):
        return NotImplementedError

    def __repr__(self,):
        string = f"sky type - {self.type}\n"
        for k,v in self.repr_dict.items():
            string += f"{k} --- {v}\n"
        return string
    
class NoSkyPrior(BaseSkyPrior):
    def __init__(self, sky_guess: float, sky_guess_err: float, suffix: str = ''):
        """Class for no sky model

        Parameters
        ----------
        sky_guess : float
            Initial guess for level of background, by default None
        sky_guess_err : float
            Uncertainty on initial guess, by default None
        suffix : str, optional
            suffix to be added on end of variables, by default ''
        """
        super().__init__(sky_guess, sky_guess_err, suffix)
        self.type = 'None'
    def sample(self,X,Y):
        return 0.
    
class FlatSkyPrior(BaseSkyPrior):
    def __init__(self, sky_guess: float, sky_guess_err: float, suffix: str = ''):
        """Class for sky model of constant background

        Parameters
        ----------
        sky_guess : float
            Initial guess for level of background, by default None
        sky_guess_err : float
            Uncertainty on initial guess, by default None
        suffix : str, optional
            suffix to be added on end of variables, by default ''
        """
        super().__init__(sky_guess, sky_guess_err, suffix)
        self.reparam_dict['sky_back'+self.suffix] = infer.reparam.TransformReparam()
        self.dist_dict['sky_back'+self.suffix] = dist.TransformedDistribution(
                                dist.Normal(),
                                dist.transforms.AffineTransform(self.sky_guess,self.sky_guess_err),)
        self.type = 'Flat'
        self.repr_dict["sky_back"] = f"Normal with mu = {self.sky_guess:.3e} and sd = {self.sky_guess_err:.3e}"

    def sample(self, X,Y):
        return sample('sky_back'+self.suffix, self.dist_dict['sky_back'+self.suffix])

class TiltedPlaneSkyPrior(BaseSkyPrior):

    def __init__(self, sky_guess: float, sky_guess_err: float, suffix: str = ''):
        """Class for tilted-plane sky model

        Parameters
        ----------
        sky_guess : float
            Initial guess for level of background, by default None
        sky_guess_err : float
            Uncertainty on initial guess, by default None
        suffix : str, optional
            suffix to be added on end of variables, by default ''
        """
        super().__init__(sky_guess, sky_guess_err, suffix)
        
        
        self.dist_dict['sky_back'+self.suffix] = dist.TransformedDistribution(
                                dist.Normal(),
                                dist.transforms.AffineTransform(self.sky_guess,self.sky_guess_err),)
        self.dist_dict['sky_x_sl'+self.suffix] = dist.TransformedDistribution(
                                dist.Normal(),
                                dist.transforms.AffineTransform(0.,0.1*self.sky_guess_err),)
        self.dist_dict['sky_y_sl'+self.suffix] = dist.TransformedDistribution(
                                dist.Normal(),
                                dist.transforms.AffineTransform(0.,0.1*self.sky_guess_err),)
    

        self.reparam_dict['sky_back'+self.suffix] = infer.reparam.TransformReparam()
        self.reparam_dict['sky_x_sl'+self.suffix] = infer.reparam.TransformReparam()
        self.reparam_dict['sky_y_sl'+self.suffix] = infer.reparam.TransformReparam()
        self.type = 'Tilted-plane'

        self.repr_dict["sky_back"] = f"Normal with mu = {self.sky_guess:.3e} and sd = {self.sky_guess_err:.3e}"
        self.repr_dict["sky_x_sl"] = f"Normal with mu = 0. and sd = {self.sky_guess_err*0.1:.3e}"
        self.repr_dict["sky_y_sl"] = f"Normal with mu = 0. and sd = {self.sky_guess_err*0.1:.3e}"


    def sample(self, X,Y):

        back = sample('sky_back'+self.suffix, self.dist_dict['sky_back'+self.suffix])
        x_sl = sample('sky_x_sl'+self.suffix, self.dist_dict['sky_x_sl'+self.suffix])
        y_sl = sample('sky_y_sl'+self.suffix, self.dist_dict['sky_y_sl'+self.suffix])
        
        x_mid = float(X.shape[0]/2.)
        y_mid = float(Y.shape[0]/2.)

        return back + (X-x_mid)*x_sl + (Y-y_mid)*y_sl

class BasePrior(ABC):
    """
    Base class for priors with sky sampling included
    """
    def __init__(self, sky_type = 'none',sky_guess=None,sky_guess_err=None, suffix = '') -> None:
        """Initialize a base prior class

        Parameters
        ----------
        sky_type : str, optional
            Type of sky mode to use, one of: none,flat or tilted-plane, by default 'none'
        sky_guess : float
            Initial guess for level of background, by default None
        sky_guess_err : float
            Uncertainty on initial guess, by default None
        suffix : str, optional
            suffix to be added on end of variables, by default ''
        """
        self.reparam_dict = {}
        self.dist_dict = {}
        self.repr_dict = {}
        self.sky_type = sky_type
        self.suffix = suffix

        if sky_guess is None:
            self.sky_guess = 0.0
            self.sky_guess_err = 0.
        else:
            assert sky_guess is not None and sky_guess_err is not None, 'If using fitting a sky then must supply a guess and uncertainty on the background value'
            self.sky_guess = sky_guess
            self.sky_guess_err = sky_guess_err

        if self.sky_type not in base_sky_types:
            raise AssertionError("Sky type must be one of: ", base_sky_types)
        
        elif self.sky_type == 'none':
            self.sky_prior = NoSkyPrior(0.,0.)
    
        elif self.sky_type == 'flat':
            self.sky_prior = FlatSkyPrior(sky_guess=self.sky_guess, sky_guess_err=self.sky_guess_err, suffix = self.suffix)

        elif self.sky_type == 'tilted-plane':
            self.sky_prior = TiltedPlaneSkyPrior(sky_guess=self.sky_guess, sky_guess_err=self.sky_guess_err, suffix = self.suffix)
    

    def update_suffix(self, new_suffix: str):
        """Change suffix for variable names

        Parameters
        ----------
        new_suffix : str
        """
        old_suffix = copy(self.suffix)

        if len(old_suffix) == 0: # If no suffix then add
            name_change = lambda s: s+new_suffix
        else: # Otherwise replace
            name_change = lambda s: s.replace(old_suffix, new_suffix)

        new_dist_dict = {}
        for k,v in self.dist_dict.items():
            new_dist_dict[name_change(k)] = self.dist_dict[k]

        new_reparam_dict = {} 
        for k,v in self.dist_dict.items():
            new_reparam_dict[name_change(k)] = self.reparam_dict[k]

        self.reparam_dict = new_reparam_dict
        self.dist_dict = new_dist_dict
        self.suffix = new_suffix

        new_dist_dict = {}
        for k,v in self.sky_prior.dist_dict.items():
            new_dist_dict[name_change(k)] = self.sky_prior.dist_dict[k]

        new_reparam_dict = {} 
        for k,v in self.sky_prior.dist_dict.items():
            new_reparam_dict[name_change(k)] = self.sky_prior.reparam_dict[k]

        self.sky_prior.reparam_dict = new_reparam_dict
        self.sky_prior.dist_dict = new_dist_dict
        self.sky_prior.suffix = new_suffix
    
    def sample_sky(self,X: jax.numpy.array,Y: jax.numpy.array)-> float:
        """Sample sky parameters and return sky model

        Parameters
        ----------
        X : jax.numpy.array
            2D mesh grid of pixel x pixel indices
        Y : jax.numpy.array
            2D mesh grid of pixel y pixel indices


        Returns
        -------
        float
            sampled and rendered sky model
        """
        return self.sky_prior.sample(X,Y)

    def _set_dist(self, var_name: str, dist: dist.Distribution)-> None:
        """Set prior for a given variable

        Parameters
        ----------
        var_name : str
            variable name
        dist : dist.Distribution
            Numpyro distribution object specifying prior
        """
        self.dist_dict[var_name] = dist

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
        return self.dist_dict[var_name]



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
    
    def __call__(self) -> jax.numpy.array:
        """
        Sample variables from prior

        Returns
        -------
        jax.numpy.array
            sampled variables
        """

        res_dict = {}
        for param,prior in self.dist_dict.items():
            res_dict[param] = sample(param, prior) 
        return res_dict


class PySersicSourcePrior(BasePrior):
    """
    Class used for priors for single source fitting in PySersic
    """
    def __init__(self, 
                profile_type: str, 
                sky_type: Optional[str] = 'none',
                sky_guess: Optional[float]=None,
                sky_guess_err: Optional[float] = None,
                suffix: Optional[str] =  "") -> None:
        """Initialize PySersicSourcePrior class

        Parameters
        ----------
        profile_type : str
            Type of profile
        sky_type : Optional[str], optional
            Type of sky model to use, one of: none, flat, tilted-plane, by default 'none'
        sky_guess : float
            Initial guess for level of background, by default None
        sky_guess_err : float
            Uncertainty on initial guess, by default None
        suffix : Optional[str], optional
            Additional suffix to add to each variable name, by default ""
        """
        super().__init__(sky_type= sky_type,sky_guess=sky_guess,sky_guess_err=sky_guess_err, suffix = suffix)
        assert profile_type in base_profile_types
        self.profile_type = profile_type
        self.param_names = base_profile_params[self.profile_type]

    def __repr__(self) -> str:
        out = f"Prior for a {self.profile_type} source:"
        num_dash = len(out)
        out += "\n" + "-"*num_dash + "\n"
        for (var, descrip) in self.repr_dict.items():
            out += var + " ---  " + descrip + "\n"    
        out += self.sky_prior.__repr__()
        return out
    

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
            if not var in self.dist_dict.keys():
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
            sky_guess: Optional[float] = None,
            sky_guess_err: Optional[float] = None,  
            suffix: Optional[str] = ''  
            )-> None:
        """
        Ingest a catalog-like data structure containing prior positions and parameters for multiple sources in a single image. 
        The format of the catalog can be a `pandas.DataFrame`, `numpy` RecordArray, dictionary, or any other format so-long as 
        the following fields exist and can be directly indexed: 'x', 'y', 'flux', 'r' and 'type'

        Parameters
        ----------
        image : jax.numpy.array
            science image
        catalog : Union[pandas.DataFrame,dict, np.recarray]
            Object containing information about the sources to be 
        sky_type : Optional[str], optional
            Type of sky model to use, one of: none, flat, tilted-plane, by default 'none'
        sky_guess : float
            Initial guess for level of background, by default None
        sky_guess_err : float
            Uncertainty on initial guess, by default None
        suffix : Optional[str], optional
            Additional suffix to add to each variable name, by default ""
        """
        properties = SourceProperties(-99)

        if sky_type != 'none':
            assert sky_guess is not None and sky_guess_err is not None, "If using a sky model must provide initial guess and uncertainty"
            properties.set_sky_guess(sky_guess = sky_guess, sky_guess_err = sky_guess_err)
        else:
            properties.set_sky_guess(sky_guess = 0., sky_guess_err = 0.)
        
        super().__init__(sky_type = sky_type,sky_guess=sky_guess,sky_guess_err=sky_guess_err)

        self.catalog = catalog
        self.N_sources = len(catalog['x'])
        self.suffix = suffix

        #Loop through catalog to generate priors
        for ind in range(len(catalog['x'])):
            properties.set_flux_guess(catalog['flux'][ind])
            properties.set_r_eff_guess(r_eff_guess = catalog['r'][ind])
            properties.set_position_guess((catalog['x'][ind],catalog['y'][ind]) )
            try:
                properties.set_theta_guess(catalog['theta'][ind])
            except:
                properties.set_theta_guess(0)
            
            dummy_prior = properties.generate_prior(catalog['type'][ind], 
                                              sky_type= 'none',
                                              suffix  = f'_{ind:d}{suffix}')

            for k,v in dummy_prior.dist_dict.items():
                self._set_dist(k,v)
            
            for k,v in dummy_prior.reparam_dict.items():
                self.reparam_dict[k] = v

            for k,v in dummy_prior.repr_dict.items():
                self.repr_dict[f'{k}_{ind}'] = v
            del dummy_prior
    
    def __repr__(self,)-> str:
        out = f"PySersicMultiPrior containing {self.N_sources:d} sources \n"
        for i, profile_type in enumerate( self.catalog['type'] ):
            out_cur = f"Source #{i} of type - {profile_type}:"
            num_dash = len(out_cur)
            out_cur += "\n" + "-"*num_dash + "\n"

            for var in base_profile_params[profile_type]:
                descrip = self.repr_dict[f'{var}_{i}']
                out_cur += f"{var}_{i} ---  {descrip}\n"

            out += out_cur  
        out += self.sky_prior.__repr__()
        return out



class SourceProperties():
    """
    A Class used to estimate initial guesses for source properties. 
    If no guesses are provided, then the class will estimate them 
    using the `photutls` package and the `data_properties()` function.
    """
    def __init__(self,
                 image: Union[np.array,jnp.array], 
                 mask: Union[np.array,jnp.array] = None):
        """Initialize the a SourceProperties object

        Parameters
        ----------
        image : np.array
            science image
        mask : np.array, optional
            pixel by pixel mask, by default None
        """
        # Force back to numpy for photutils compatibility
        if not isinstance(image,int):
            self.image = np.array(image)
            if mask is not None:
                self.mask = np.array(mask) 
            else:
                self.mask = None
            
            if self.mask is not None:
                self.cat = data_properties(self.image,mask=self.mask.astype(bool))
                self.image = np.ma.masked_array(self.image, self.mask )
            else:
                self.cat = data_properties(self.image)
            _ = self.measure_properties() 

    def measure_properties(self,**kwargs) -> SourceProperties:
        """Measure default properties of the source

        Returns
        -------
        SourceProperties
            returns self
        """
        self.set_flux_guess(**kwargs)
        self.set_r_eff_guess(**kwargs)
        self.set_theta_guess(**kwargs)
        self.set_position_guess(**kwargs)
        self.set_sky_guess(**kwargs)
        return self
    
    def set_sky_guess(self,
                      sky_guess: Optional[float] = None,
                      sky_guess_err:Optional[float] = None,
                      n_pix_sample:int = 5, 
                      **kwargs)-> SourceProperties:
        """Measure or set guess for initial sky background level. 
        If no estimate is provided, the median of the n_pix_sample number of pixels around each edge is used

        Parameters
        ----------
        sky_guess : Optional[float], optional
            Initial guess for level of background, by default None
        sky_guess_err : Optional[float], optional
            Uncertainity on inital guess, by default None
        n_pix_sample : int, optional
            Number of pixels around each edge to use to estimate sky level if neccesary, by default 5

        Returns
        -------
        SourceProperties
            returns self
        """

        if sky_guess is None and hasattr(self, 'image'):
            med, std, npix = estimate_sky(self.image, n_pix_sample= n_pix_sample)
            self.sky_guess = med
        elif sky_guess is not None:
            self.sky_guess = sky_guess
        else:
            raise RuntimeError("Need to either supply image or sky_guess_err to source_properties class")
        if sky_guess_err is None and hasattr(self, 'image'):
            med, std, npix = estimate_sky(self.image, n_pix_sample= n_pix_sample)
            self.sky_guess_err = 2*std/np.sqrt(npix)
        elif sky_guess_err is not None:
            self.sky_guess_err = sky_guess_err
        else:
            raise RuntimeError("Need to either supply image or sky_guess_Err to source_properties class")
        return self

    def set_flux_guess(self,
                       flux_guess: Optional[float] = None,
                       flux_guess_err: Optional[float] = None,
                       **kwargs)-> SourceProperties:
        """Measure or set guess for initial flux. 
        If no estimate is provided, the flux of the source in estimated as the total flux 
        within the sgmentatated region for the source

        Parameters
        ----------
        flux_guess : Optional[float], optional
            Initial guess for flux, by default None
        flux_guess_err : Optional[float], optional
            Uncertainty on initial guess, by default None

        Returns
        -------
        SourceProperties
            returns self
        """
        if flux_guess is None:
            flux_guess = self.cat.segment_flux
        if flux_guess_err is not None:
            flux_guess_err = flux_guess_err
        else:
            if flux_guess > 0:
                flux_guess_err = 2*np.sqrt( flux_guess )
            else:
                flux_guess_err = 2*np.sqrt(np.abs(flux_guess))
                flux_guess = 0.
        
        self.flux_guess = flux_guess 
        self.flux_guess_err = flux_guess_err
        return self 
    
    def set_r_eff_guess(self,
                        r_eff_guess:Optional[float] = None, 
                        r_eff_guess_err:Optional[float] = None, 
                        **kwargs)-> SourceProperties:
        """Measure or set guess for effective radius. 
        If no estimate is provided, the r_eff of the source in estimated using photutils

        Parameters
        ----------
        r_eff_guess : Optional[float], optional
            Initial guess for effective radius, by default None
        r_eff_guess_err : Optional[float], optional
            Uncertainty on initial guess, by default None

        Returns
        -------
        SourceProperties
            returns self
        """
        if r_eff_guess is None:
            r_eff_guess = self.cat.fluxfrac_radius(0.5).to(u.pixel).value
        
        if r_eff_guess_err is not None:
            self.r_eff_guess_err = r_eff_guess_err
        else:
            self.r_eff_guess_err = 2*np.sqrt(r_eff_guess) 

        self.r_eff_guess = r_eff_guess
        return self

    def set_theta_guess(self,
                        theta_guess: Optional[float] = None,
                        **kwargs)-> SourceProperties:
        """Measure or set guess for initial position angle. 
        If no estimate is provided, the position angle of the source in estimated 
        using the data_properties() function from photutils

        Parameters
        ----------
        theta_guess : Optional[float], optional
            Estimate of the position angle in radians, by default None

        Returns
        -------
        SourceProperties
            returns self
        """

        if theta_guess is None:
            theta_guess = self.cat.orientation.to(u.rad).value

        if np.isnan(theta_guess):
            theta_guess = 0
        self.theta_guess = theta_guess 
        return self
    
    def set_position_guess(self,
                           position_guess: Optional[Iterable[float,float]]=None,
                           **kwargs)-> SourceProperties:
        """Measure or set guess for initial position. 
        If no estimate is provided, the position of the source in estimated 
        using the data_properties() function from photutils

        Parameters
        ----------
        position_guess : Optional[Iterable[float,float]], optional
            A 2 element list, tuple or array which contain the x,y pixel values of the inital guess for the centroid, by default None

        Returns
        -------
        SourceProperties
            returns self
        """
        if position_guess is None:
            self.xc_guess = self.cat.centroid_win[0]
            self.yc_guess = self.cat.centroid_win[1]
        else:
            self.xc_guess = position_guess[0]
            self.yc_guess = position_guess[1]
        return self
    
    def generate_prior(self,
                profile_type: str,
                sky_type: Optional[str] = 'none',
                suffix: Optional[str] = '')-> PySersicSourcePrior:
        """Function to generate default priors based on a given image and profile type

        Parameters
        ----------
        profile_type : str
            Type of profile
        sky_type : str, default 'none'
            Type of sky model to use, must be one of: 'none', 'flat', 'tilted-plane'
        suffix : str, default ''
            Add suffix onto all source related variables, generally only used to number sources in MultiPrior
        Returns
        -------
        PySersicSourcePrior
            Pysersic prior object to be used to initialize FitSingle
        """
        
        prior = PySersicSourcePrior(profile_type=profile_type, 
                                    sky_type = sky_type,
                                    suffix=suffix,
                                    sky_guess=self.sky_guess,
                                    sky_guess_err=self.sky_guess_err)

        # 3 properties common to all sources
        prior.set_gaussian_prior('flux',self.flux_guess,self.flux_guess_err)
        prior.set_gaussian_prior('xc', self.xc_guess, 1.)
        prior.set_gaussian_prior('yc', self.yc_guess, 1.)

        if profile_type in ['exp','dev','sersic']:
            prior.set_truncated_gaussian_prior('r_eff', self.r_eff_guess,self.r_eff_guess_err, low = 0.5)
            prior.set_uniform_prior('ellip', 0, 0.9)
            prior.set_custom_prior('theta', 
                                   dist.VonMises(loc = self.theta_guess,concentration=2), 
                                   reparam= infer.reparam.CircularReparam() )
            if profile_type == 'sersic':
                prior.set_uniform_prior('n', 0.65, 8)

        elif profile_type == 'doublesersic':

            prior.set_uniform_prior('f_1', 0.,1.)

            prior.set_custom_prior('theta', 
                                   dist.VonMises(loc = self.theta_guess,concentration=2), 
                                   reparam= infer.reparam.CircularReparam() )

            r_loc1 = self.r_eff_guess/1.5
            r_eff_guess_err1 = jnp.sqrt(self.r_eff_guess/1.5)
            prior.set_truncated_gaussian_prior('r_eff_1', r_loc1,r_eff_guess_err1, low = 0.5)

            r_loc2 = self.r_eff_guess*1.5
            r_eff_guess_err2 = jnp.sqrt(self.r_eff_guess*1.5)
            prior.set_truncated_gaussian_prior('r_eff_2', r_loc2,r_eff_guess_err2, low = 0.5)


            prior.set_uniform_prior('ellip_1', 0,0.9)
            prior.set_uniform_prior('ellip_2', 0,0.9)

            prior.set_truncated_gaussian_prior('n_1',4,1, low = 0.65,high = 8)
            prior.set_truncated_gaussian_prior('n_2',1,1, low = 0.65,high = 8)

        
        return prior
    
    def visualize(self,
                  figsize: Tuple[float,float]= (6.,6.),
                  cmap:str = 'gray',
                  scale: float = 1.)-> None:
        """Display a figure summarizing the current guess for the source properties

        Parameters
        ----------
        figsize : Tuple[float,float], optional
            figure, by default (6.,6.)
        cmap : str, optional
            color map, by default 'gray'
        scale : float, optional
            number of +/- std's at which to clip image, by default 1
        """
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

def estimate_sky(image: np.array, mask: Optional[np.array] = None, n_pix_sample:int = 5 )-> Tuple[float,float,int]:
    """Estimate the sky background using the edge of the cutout

    Parameters
    ----------
    im : np.array
        image, either an array or masked array
    mask : Optional[np.array], optional
        mask to apply, if im is not a masked array already, by default None
    n_pix_sample : int, optional
        number of pixels around the edge to use, by default 5

    Returns
    -------
    Tuple[float,float,int]
        a tuple containing the median, standard deviation and number of pixels used
    """
    if not np.ma.is_masked(image) and not mask is None:
        image = np.ma.masked_array(image, mask )
    edge_pixels = np.concatenate((image[:n_pix_sample,:],image[-n_pix_sample:,:],image[n_pix_sample:-n_pix_sample,:n_pix_sample],image[n_pix_sample:-n_pix_sample,-n_pix_sample:]),axis=None)
    median_val = np.ma.median(edge_pixels)
    err_on_median = bws(edge_pixels)
    return median_val, err_on_median, np.prod(edge_pixels.shape)

def autoprior(image: np.array, profile_type: 'str', mask:np.array =None, sky_type: str = 'none') -> PySersicSourcePrior:
    """Simple wrapper function to generate a prior using the built-in defaults. This can be used as a starting place but may not work for all sources

    Parameters
    ----------
    image : np.array
        science image
    profile_type : str
        Type of profile to be fit
    mask : np.array, optional
        pixel by pixel mask, by default None
    sky_type : str
        Type of sky to fit, default 'none'

    Returns
    -------
    PySersicSourcePrior
        Prior object that can be used in initializing FitSingle
    """

    props = SourceProperties(image = image, mask = mask)
    
    prior = props.generate_prior(profile_type = profile_type, sky_type = sky_type)
    return prior

