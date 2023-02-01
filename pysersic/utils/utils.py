import jax.numpy as jnp
from numpyro import distributions as dist, sample


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


def gaussian_loss(mod: jnp.array,
                data: jnp.array,
                rms:jnp.array)-> float:
    """Basic Gaussian loss function using given uncertainties

    Parameters
    ----------
    mod : jnp.array
        Model image
    data : jnp.array
        data to be fit
    rms : jnp.array
        per pixel 1-sigma uncertainties

    Returns
    -------
    float
        Sampled loss function
    """

    return sample("Loss", dist.Normal(mod, rms), obs=data)    
    
