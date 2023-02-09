import jax.numpy as jnp
from numpyro import distributions as dist, sample, handlers, factor,deterministic

def gaussian_loss(mod: jnp.array,
                data: jnp.array,
                rms:jnp.array,
                mask: jnp.array)-> float:
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

    with handlers.mask(mask = mask):
        loss = sample("Loss", dist.Normal(mod, rms), obs=data)
    return loss

def cash_loss(mod: jnp.array,
                data: jnp.array,
                rms:jnp.array,
                mask: jnp.array)-> float:
    """
    Cash statistic based on Poisson statistics derrived in Cash (1979) (DOI 10.1086/156922) and advocated for in Erwin (2015) (https://arxiv.org/abs/1408.1097). Since the is based on Poisson statistics, scaling of the image will produce different confidence intervals. Additionally since a logorithm is taken of the model image, negative values will cause issues.

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
    
    with handlers.mask(mask = mask):
        loss = factor('cash_loss', -1*(mod - data*jnp.log(mod)))
    return loss

def gaussian_loss_w_frac(mod: jnp.array,
                data: jnp.array,
                rms:jnp.array,
                mask: jnp.array)-> float:
    """Gaussian loss with and additional fractional increase to all uncertainties such that,

    $$ \sigma_{new,i} = (1 + f) * \sigma_{old,i} $$

    f is a free parameter varied between -0.5 and 2

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
    
    scatter_frac = sample('frac_scatter_increase', dist.TruncatedNormal(low = -0.5, high = 2) )
    with handlers.mask(mask = mask):
        loss =  sample("Loss", dist.Normal(mod, (1+ scatter_frac)*rms), obs=data)    
    return loss

def gaussian_loss_w_const(mod: jnp.array,
                data: jnp.array,
                rms:jnp.array,
                mask: jnp.array)-> float:
    """Gaussian loss with and additional systematic increase such_that

    $$ \sigma_{new,i}^2 = \sigma_{old,i}^2 + \sigma_{sys}^2 $$

    \sigma_{sys} is a free parameter with a Normal prior, with loc = 0 and scale = mean(rms) truncated to ensure > 0

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
    
    sys_scatter_base = sample('sys_scatter_base', dist.TruncatedNormal(low = 0, scale = 1 ) )
    sys_scatter = deterministic('sys_scatter', sys_scatter_base*jnp.mean(rms))
    with handlers.mask(mask = mask):
        loss =  sample("Loss", dist.Normal(mod, jnp.sqrt(rms**2 + sys_scatter**2)), obs=data)    
    return loss

def student_t_loss(mod: jnp.array,
                data: jnp.array,
                rms:jnp.array,
                mask: jnp.array)-> float:
    """
    Student T loss, with a fixed df = 5. This has fatter tails than Gaussian loss (or chi squared) so is less punishing to outliers

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
    
    with handlers.mask(mask = mask):
        loss =  sample("Loss", dist.StudentT(5,mod,(5/4)*rms), obs=data)    
    return loss

def student_t_loss_free_nu(mod: jnp.array,
                data: jnp.array,
                rms:jnp.array,
                mask: jnp.array)-> float:
    """
    Student T loss, with free df varied between 2 and 50. At low df, Student T has fatter tails than Gaussian loss (or chi squared) so is less punishing to outliers. At high df, the Student T approachs a Gaussian distribution

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
    nu_eff_base = sample('nu_eff_base', dist.Uniform())
    nu = deterministic('nu_eff', nu_eff_base*48 + 2)
    with handlers.mask(mask = mask):
        loss =  sample("Loss", dist.StudentT(nu,mod,(nu/(nu-1))*rms), obs=data)    
    return loss