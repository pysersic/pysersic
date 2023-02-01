
from numpyro import distributions as dist, sample
import jax.numpy as jnp 
import jax
import pandas
import numpy as np
from typing import Union, Optional, Callable, Iterable


def autoprior(image: jax.numpy.array,
        profile_type: str)-> dict:
    """Function to generate default priors based on a given image and profile type

    Parameters
    ----------
    image : jax.numpy.array
        Masked image
    profile_type : str
        Type of profile

    Returns
    -------
    dict
        Dictionary containing numpyro Distribution objects for each parameter
    """
    if profile_type == 'sersic':
        prior_dict = generate_sersic_prior(image)
    
    elif profile_type == 'doublesersic':
        prior_dict = generate_doublesersic_prior(image)

    elif profile_type == 'pointsource':
        prior_dict = generate_pointsource_prior(image)
   
    elif profile_type in ['exp','dev']:
        prior_dict = generate_exp_dev_prior(image)
    
    return prior_dict

def generate_sersic_prior(image: jax.numpy.array, 
        flux_guess: Optional[float] = None,
        r_eff_guess: Optional[float] = None, 
        position_guess: Optional[Iterable] = None)-> dict:
    """ Derive automatic priors for a sersic profile based on an input image.

    Parameters
    ----------
    image : jax.numpy.array
        Masked image
    flux_guess : Optional[float], optional
        Estimate of total flux, by default None
    r_eff_guess : Optional[float], optional
        Estimate of effective radius, by default None
    position_guess : Optional[Iterable], optional
        Estimate of central position, by default None

    Returns
    -------
    dict
        Dictionary containing numpyro Distribution objects for each parameter

    """

    if flux_guess is None:
        flux_guess = jnp.sum(image)
    flux_prior = dist.TransformedDistribution(
                                dist.Normal(),
                                dist.transforms.AffineTransform(flux_guess,jnp.sqrt(flux_guess)),)
    

    image_dim_min = jnp.min(jnp.array([image.shape[0],image.shape[1]])) 
    if r_eff_guess is None:
        r_eff_guess = image_dim_min/10
    
    r_loc = r_eff_guess
    r_scale = jnp.sqrt(r_eff_guess)
    low_scaled = (1. - r_loc)/r_scale
    reff_prior = dist.TransformedDistribution(
                                dist.TruncatedNormal(low = low_scaled),
                                dist.transforms.AffineTransform(r_loc,r_scale) )
    

    ellip_prior = dist.Uniform(0,0.8) 
    theta_prior = dist.VonMises(loc = 0,concentration=0)
    n_prior = dist.TruncatedNormal(loc = 2, scale= 1., low = 0.5, high=8)

    if position_guess is None:
        xc_guess = image.shape[0]/2
        yc_guess = image.shape[1]/2
    else:
        xc_guess = position_guess[0]
        yc_guess = position_guess[1]
    xc_prior = dist.TransformedDistribution(
                                dist.Normal(),
                                dist.transforms.AffineTransform(xc_guess,1) )
    yc_prior =  dist.TransformedDistribution(
                                dist.Normal(),
                                dist.transforms.AffineTransform(yc_guess,1) )

    prior_dict = {
        'xc': xc_prior,
        'yc': yc_prior,
        'flux': flux_prior,
        'r_eff': reff_prior,
        'n': n_prior,
        'ellip': ellip_prior,
        'theta': theta_prior,
    }
    return prior_dict 

def generate_exp_dev_prior(image: jax.numpy.array, 
        flux_guess: Optional[float] = None,
        r_eff_guess: Optional[float] = None, 
        position_guess: Optional[Iterable] = None)-> dict:
    """ Derive automatic priors for a exp or dev profile based on an input image.
    
    Parameters
    ----------
    image : jax.numpy.array
        Masked image
    flux_guess : Optional[float], optional
        Estimate of total flux, by default None
    r_eff_guess : Optional[float], optional
        Estimate of effective radius, by default None
    position_guess : Optional[Iterable], optional
        Estimate of central position, by default None

    Returns
    -------
    dict
        Dictionary containing numpyro Distribution objects for each parameter

    """
    
    prior_dict = generate_sersic_prior(image, flux_guess = flux_guess, r_eff_guess = r_eff_guess, position_guess=position_guess)
    prior_dict.pop('n')
    
    return prior_dict

def generate_doublesersic_prior(image: jax.numpy.array, 
        flux_guess: Optional[float] = None,
        r_eff_guess: Optional[float] = None, 
        position_guess: Optional[Iterable] = None)-> dict:
    """ Derive automatic priors for a double sersic profile based on an input image.
    
    Parameters
    ----------
    image : jax.numpy.array
        Masked image
    flux_guess : Optional[float], optional
        Estimate of total flux, by default None
    r_eff_guess : Optional[float], optional
        Estimate of effective radius, by default None
    position_guess : Optional[Iterable], optional
        Estimate of central position, by default None

    Returns
    -------
    dict
        Dictionary containing numpyro Distribution objects for each parameter

    """

    if flux_guess is None:
        flux_guess = jnp.sum(image)
    flux_prior = dist.TransformedDistribution(
                                dist.Normal(),
                                dist.transforms.AffineTransform(flux_guess,jnp.sqrt(flux_guess)),)
    frac_1 = dist.Uniform()

    image_dim_min = jnp.min(jnp.array([image.shape[0],image.shape[1]])) 
    if r_eff_guess is None:
        r_eff_guess = image_dim_min/10

    r_loc = r_eff_guess/1.5
    r_scale = jnp.sqrt(r_eff_guess)
    low_scaled = (1. - r_loc)/r_scale
    reff_1_prior = dist.TransformedDistribution(
                                dist.TruncatedNormal(low = low_scaled),
                                dist.transforms.AffineTransform(r_loc,r_scale) )
    
    r_loc = r_eff_guess*1.5
    r_scale = jnp.sqrt(r_eff_guess)
    low_scaled = (1. - r_loc)/r_scale
    reff_2_prior = dist.TransformedDistribution(
                                dist.TruncatedNormal(low = low_scaled),
                                dist.transforms.AffineTransform(r_loc,r_scale) )
    

    ellip_1_prior = dist.Uniform(0,0.8)
    ellip_2_prior = dist.Uniform(0,0.8) 

    n_1_prior = dist.TruncatedNormal(loc = 4, scale= 1, low = 0.5, high=8)
    n_2_prior = dist.TruncatedNormal(loc = 1, scale= 1, low = 0.5, high=8)

    theta_prior = dist.VonMises(loc = 0,concentration=0)
    
    if position_guess is None:
        xc_guess = image.shape[0]/2
        yc_guess = image.shape[1]/2
    else:
        xc_guess = position_guess[0]
        yc_guess = position_guess[1]

    xc_prior = dist.TransformedDistribution(
                                dist.Normal(),
                                dist.transforms.AffineTransform(xc_guess,1) )
    yc_prior =  dist.TransformedDistribution(
                                dist.Normal(),
                                dist.transforms.AffineTransform(yc_guess,1) )

    prior_dict = {
        'xc': xc_prior,
        'yc': yc_prior,
        'flux': flux_prior,
        'f_1':frac_1,
        'r_eff_1': reff_1_prior,
        'n_1': n_1_prior,
        'ellip_1': ellip_1_prior,
        'r_eff_2': reff_2_prior,
        'n_2': n_2_prior,
        'ellip_2': ellip_2_prior,
        'theta': theta_prior,
    }
    return prior_dict 

def generate_pointsource_prior(image: jax.numpy.array, 
        flux_guess: Optional[float] = None,
        position_guess: Optional[Iterable] = None)-> dict:
    """ Derive automatic priors for a pointsource based on an input image.
    
    Parameters
    ----------
    image : jax.numpy.array
        Masked image
    flux_guess : Optional[float], optional
        Estimate of total flux, by default None
    position_guess : Optional[Iterable], optional
        Estimate of central position, by default None

    Returns
    -------
    dict
        Dictionary containing numpyro Distribution objects for each parameter

    """

    if flux_guess is None:
        flux_guess = jnp.sum(image)
    flux_prior = dist.TransformedDistribution(
                                dist.Normal(),
                                dist.transforms.AffineTransform(flux_guess,jnp.sqrt(flux_guess)),)

    if position_guess is None:
        xc_guess = image.shape[0]/2
        yc_guess = image.shape[1]/2
    else:
        xc_guess = position_guess[0]
        yc_guess = position_guess[1]

    xc_prior = dist.TransformedDistribution(
                                dist.Normal(),
                                dist.transforms.AffineTransform(xc_guess,0.5) )
    yc_prior =  dist.TransformedDistribution(
                                dist.Normal(),
                                dist.transforms.AffineTransform(yc_guess,0.5) )

    prior_dict = {
        'xc': xc_prior,
        'yc': yc_prior,
        'flux': flux_prior,
    }
    return prior_dict 

def multi_prior(image: jax.numpy.array,
        catalog: Union[pandas.DataFrame,dict, np.recarray]
        )-> Iterable:
    """Ingest a catalog-like data structure containing prior positions and parameters for multiple sources in a single image. The format of the catalog can be a `pandas.DataFrame`, `numpy` RecordArray, dictionary, or any other format so-long as the following fields exist and can be directly indexed: 'x', 'y', 'flux', 'r' and 'type'

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
    all_priors = []


    for ind in range(len(catalog['x'])):

        init = dict(flux_guess = catalog['flux'][ind], r_eff_guess = catalog['r'][ind], position_guess = (catalog['x'][ind],catalog['y'][ind]) )

        if catalog['type'][ind] == 'sersic':
            prior_dict = generate_sersic_prior(image, **init)
        
        elif catalog['type'][ind] == 'doublesersic':
            prior_dict = generate_doublesersic_prior(image, **init)

        elif catalog['type'][ind] == 'pointsource':
            init.pop('r_eff_guess')
            prior_dict = generate_pointsource_prior(image, **init)

        elif catalog['type'][ind] in ['exp','dev']:
            prior_dict = generate_exp_dev_prior(image, **init)
    
        source_dict = {}
        for key in prior_dict.keys():
            source_dict[key+f'_{ind:d}'] = prior_dict[key]
        
        all_priors.append(source_dict)
    return all_priors

