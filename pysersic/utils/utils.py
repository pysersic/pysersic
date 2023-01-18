
from numpyro import distributions as dist
import jax.numpy as jnp 



def autoprior(image,verbose=False):
    """
    Derive automatic priors based on an input image.
    """
    image_sum = jnp.sum(image)
    log_flux_prior = dist.Uniform(jnp.floor(jnp.log10(image_sum)),jnp.ceil(jnp.log10(image_sum)))
    im_shape = image.shape
    image_dim_min = jnp.min(jnp.array([image.shape[0],image.shape[1]])) 
    reff_prior = dist.Uniform(0.1,image_dim_min) 

    ellip_prior = dist.Uniform(0,1)
    theta_prior = dist.Uniform(0,jnp.pi)
    log_n_prior = dist.Uniform(0,1)
    x_cen = im_shape[1] / 2.0 
    y_cen = im_shape[0] / 2.0
    x_third = im_shape[1] / 3.0
    y_third = im_shape[0] / 3.0
    x0_prior = dist.Uniform(x_cen-x_third,x_cen+x_third)
    y0_prior = dist.Uniform(y_cen-y_third,y_cen+y_third)

    prior_dict = {
        'log_flux': log_flux_prior,
        'r_eff': reff_prior,
        'log_n': log_n_prior,
        'ellip': ellip_prior,
        'theta': theta_prior,
        'x_0': x0_prior,
        'y_0': y0_prior
    }
    if verbose:
        print(
            f'log_flux: {jnp.floor(jnp.log10(image_sum))} - {jnp.ceil(jnp.log10(image_sum))}'
        )
        print(f'reff: 0.1 - {image_dim_min}')
        print(f'x0: {x_cen-x_third} - {x_cen+x_third}')
        print(f'y0: {y_cen-y_third} - {y_cen+y_third}')

    return prior_dict 

