
from numpyro import distributions as dist, sample
from numpyro.handlers import reparam
from numpyro.infer.reparam import TransformReparam
import jax.numpy as jnp 

def autoprior(image,profile_type):
    if profile_type == 'sersic':
        prior_dict = generate_sersic_prior(image)
    
    elif profile_type == 'doublesersic':
        prior_dict = generate_doublesersic_prior(image)

    elif profile_type == 'pointsource':
        prior_dict = generate_pointsource_prior(image)
   
    elif profile_type in ['exp','dev']:
        prior_dict = generate_exp_dev_prior(image)
    
    return prior_dict

def generate_sersic_prior(image, flux_guess = None, r_eff_guess = None, position_guess = None):
    """
    Derive automatic priors for a sersic profile based on an input image.
    """

    if flux_guess is None:
        flux_guess = jnp.sum(image)
    flux_prior = dist.TransformedDistribution(
                                dist.Normal(),
                                dist.transforms.AffineTransform(flux_guess,jnp.sqrt(flux_guess)),)
    

    image_dim_min = jnp.min(jnp.array([image.shape[0],image.shape[1]])) 
    if r_eff_guess is None:
        r_eff_guess = image_dim_min/10
    reff_prior = dist.TruncatedNormal(loc =  r_eff_guess,scale =  r_eff_guess/3, low = 1,high = image_dim_min/4.0)

    ellip_prior = dist.Uniform(0,0.8) 
    theta_prior = dist.Uniform(-jnp.pi/2.,jnp.pi/2.) 
    n_prior = dist.TruncatedNormal(loc = 2, scale= 0.5, low = 0.5, high=8)

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

def generate_exp_dev_prior(image, flux_guess = None, r_eff_guess = None, position_guess = None):
    
    prior_dict = generate_sersic_prior(image, flux_guess = flux_guess, r_eff_guess = r_eff_guess, position_guess=position_guess)
    prior_dict.pop('n')
    
    return prior_dict

def generate_doublesersic_prior(image, flux_guess = None, r_eff_guess = None, position_guess = None):
    """
    Derive automatic priors for a double-sersic profile based on an input image.
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
    reff_1_prior = dist.TruncatedNormal(loc =  r_eff_guess/1.5,scale =  r_eff_guess/3, low = 1,high = image_dim_min/4.0) 
    reff_2_prior = dist.TruncatedNormal(loc =  r_eff_guess*1.5,scale =  r_eff_guess/3, low = 1,high = image_dim_min/4.0) 

    ellip_1_prior = dist.Uniform(0,0.8)
    ellip_2_prior = dist.Uniform(0,0.8) 

    n_1_prior = dist.TruncatedNormal(loc = 4, scale= 0.5, low = 0.5, high=8)
    n_2_prior = dist.TruncatedNormal(loc = 1, scale= 0.5, low = 0.5, high=8)

    theta_prior = dist.Uniform(-jnp.pi/2.,jnp.pi/2.) 
    
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

def generate_pointsource_prior(image, flux_guess = None,  position_guess = None):
    """
    Derive automatic priors for a pointsource based on an input image.
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

def multi_prior(image,catalog):
    
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


def sample_sky(prior_dict, sky_type):
    if sky_type is None:
        params = 0
    elif sky_type == 'flat':

        sky0 = sample('sky0', prior_dict['sky0'])
        params = sky0
    else:
        sky0 = sample('sky0', prior_dict['sky0'])
        sky1 = sample('sky1', prior_dict['sky1'])
        sky2 = sample('sky2', prior_dict['sky2'])
        params = jnp.array([sky0,sky1,sky2])
    return params

def sample_sersic(prior_dict,add_on = ''):
    flux = sample('flux'+add_on, prior_dict['flux'+add_on])
    n = sample('n'+add_on,prior_dict['n'+add_on])
    r_eff = sample('r_eff'+add_on,prior_dict['r_eff'+add_on])
    ellip = sample('ellip'+add_on,prior_dict['ellip'+add_on])
    theta = sample('theta'+add_on,prior_dict['theta'+add_on])
    xc = sample('xc'+add_on,prior_dict['xc'+add_on])
    yc = sample('yc'+add_on,prior_dict['yc'+add_on])

    #collect params and render scene
    params = jnp.array([xc,yc,flux,r_eff,n, ellip, theta])
    return params

def sample_dev_exp(prior_dict,add_on = ''):
    flux = sample('flux'+add_on, prior_dict['flux'+add_on])
    r_eff = sample('r_eff'+add_on,prior_dict['r_eff'+add_on])
    ellip = sample('ellip'+add_on,prior_dict['ellip'+add_on])
    theta = sample('theta'+add_on,prior_dict['theta'+add_on])
    xc = sample('xc'+add_on,prior_dict['xc'+add_on])
    yc = sample('yc'+add_on,prior_dict['yc'+add_on])

    #collect params and render scene
    params = jnp.array([xc,yc,flux,r_eff, ellip, theta])
    return params

def sample_doublesersic(prior_dict,add_on = ''):
    flux = sample('flux'+add_on, prior_dict['flux'+add_on])
    f_1 = sample('f_1'+add_on, prior_dict['f_1'+add_on])
    n_1 = sample('n_1'+add_on,prior_dict['n_1'+add_on])
    r_eff_1 = sample('r_eff_1'+add_on,prior_dict['r_eff_1'+add_on])
    ellip_1 = sample('ellip_1'+add_on,prior_dict['ellip_1'+add_on])
    n_2 = sample('n_2'+add_on,prior_dict['n_2'+add_on])
    r_eff_2 = sample('r_eff_2'+add_on,prior_dict['r_eff_2'+add_on])
    ellip_2 = sample('ellip_2'+add_on,prior_dict['ellip_2'+add_on])
    theta = sample('theta'+add_on,prior_dict['theta'])
    xc = sample('xc'+add_on,prior_dict['xc'+add_on])
    yc = sample('yc'+add_on,prior_dict['yc'+add_on])

    params = jnp.array([xc, yc, flux, f_1, r_eff_1, n_1, ellip_1, r_eff_2, n_2, ellip_2, theta])
    return params

def sample_pointsource(prior_dict,add_on = ''):
    flux = sample('flux'+add_on, prior_dict['flux'+add_on])
    xc = sample('xc'+add_on,prior_dict['xc'+add_on])
    yc = sample('yc'+add_on,prior_dict['yc'+add_on])

    params = jnp.array([xc,yc,flux,])
    return params

sample_func_dict = {'sersic':sample_sersic,'doublesersic':sample_doublesersic, 'pointsource':sample_pointsource, 'exp':sample_dev_exp, 'dev':sample_dev_exp}
