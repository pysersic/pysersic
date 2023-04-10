import jax.numpy as jnp
from numpyro import distributions as dist, sample, handlers, factor,deterministic
from typing import Optional

def gaussian_loss(mod: jnp.array,
                data: jnp.array,
                rms: jnp.array,
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
                rms: jnp.array,
                mask: jnp.array)-> float:
    """
    Cash statistic based on Poisson statistics derrived in Cash (1979) (DOI 10.1086/156922) and advocated for in Erwin (2015) (https://arxiv.org/abs/1408.1097) for use in Sersic fitting. Since the is based on Poisson statistics, scaling of the image will produce different confidence intervals. Additionally, since a logorithm is taken of the model image, negative values associated with different sky models will cause issues.

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
                rms: jnp.array,
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

def gaussian_loss_w_sys(mod: jnp.array,
                data: jnp.array,
                rms: jnp.array,
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
                rms: jnp.array,
                mask: jnp.array,
                nu: Optional[int] = 5)-> float:
    """
    Student T loss, with a df = 5 by default. This has fatter tails than Gaussian loss (or chi squared) so is more resiliant to outliers

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
    nu = 5.
    with handlers.mask(mask = mask):
        loss =  sample("Loss", dist.StudentT(nu,mod,jnp.sqrt((nu-2.)/2.)*rms), obs=data)    
    return loss

def student_t_loss_free_sys(mod: jnp.array,
                data: jnp.array,
                rms: jnp.array,
                mask: jnp.array,
                nu: Optional[int] = 5)-> float:
    """
    Student T loss, which has fatter tails than Gaussian loss (or chi squared) so is so is more resiliant to outliers. In addition, add additional systematic increase such that

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
    rms_new = jnp.sqrt((nu-2.)/2.)*jnp.sqrt(rms**2 + sys_scatter**2)

    with handlers.mask(mask = mask):
        loss =  sample("Loss", dist.StudentT(nu, mod, rms_new), obs=data)    
    return loss

def pseudo_huber_loss(mod: jnp.array,
                data: jnp.array,
                rms: jnp.array,
                mask: jnp.array,
                delta: Optional[int] = 3
                )-> float:
    """
    Pseudo huber loss function of the form:

    $$ L = \delta^2 * ( \sqrt{1 + (a/\delta)^2} - 1) $$

    where a is the residuals scaled by the rms and delta can be chosen. This loss function is more robust to outliers than the Gaussian loss function as it is meant to transition for L2 to L1 loss at residuals greater than delta. The delta parameter is 3 by default but can be varied.

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
        res = (data-mod)/rms
        loss = factor('pseudo_huber_loss', -1.*(jnp.sqrt(1 + ( res/delta )**2) - 1) )
    return loss

def gaussian_mixture(mod: jnp.array,
                data: jnp.array,
                rms: jnp.array,
                mask: jnp.array,
                c: Optional[float] = 5.
                )-> float:
    """
    Gaussian mixture loss function, with one representing a "contaminating" outlier distribution with standard deviation equal to c*rms where c is 5 by default. The "contaminating fraction" or fraction of outliers is a free parameter with a Uniform prior between 0 and 0.25.

    Parameters
    ----------
    Parameters
    ----------
    mod : jnp.array
        Model image
    data : jnp.array
        data to be fit
    rms : jnp.array
        per pixel 1-sigma uncertainties
    c : float, optional
        factor to increase rms for outlier distribution, by default 5

    Returns
    -------
    float
        _description_
    """
    contam_frac_base = sample('contam_frac_base', dist.TruncatedNormal(low = 0, scale = 1., high = 5.) )
    contam_frac = deterministic('contam_frac', contam_frac_base*0.05 )

    mixture_dists = dist.Categorical(probs = jnp.array([1-contam_frac, contam_frac]))
    component_dists = dist.Normal(jnp.stack([mod,mod],axis = -1), jnp.stack([rms,c*rms],axis = -1) )

    with handlers.mask(mask = mask):
        loss = sample("Loss", dist.MixtureSameFamily(mixture_dists, component_dists), obs=data)
    return loss

def gaussian_mixture_w_sys(mod: jnp.array,
                data: jnp.array,
                rms: jnp.array,
                mask: jnp.array,
                c: Optional[float] = 5.
                )-> float:
    """
    Gaussian mixture loss function, with one representing a "contaminating" outlier distribution with standard deviation equal to c*rms where c is 5 by default. The "contaminating fraction" or fraction of outliers is a free parameter with a Uniform prior between 0 and 0.25.

    Parameters
    ----------
    Parameters
    ----------
    mod : jnp.array
        Model image
    data : jnp.array
        data to be fit
    rms : jnp.array
        per pixel 1-sigma uncertainties
    c : float, optional
        factor to increase rms for outlier distribution, by default 5

    Returns
    -------
    float
        _description_
    """
    contam_frac_base = sample('contam_frac_base', dist.TruncatedNormal(low = 0, scale = 1., high = 5.) )
    contam_frac = deterministic('contam_frac', contam_frac_base*0.05 )

    sys_scatter_base = sample('sys_scatter_base', dist.TruncatedNormal(low = 0, scale = 1 ) )
    sys_scatter = deterministic('sys_scatter', sys_scatter_base*jnp.mean(rms))

    rms_new = jnp.sqrt(rms**2 + sys_scatter**2)
    mixture_dists = dist.Categorical(probs = jnp.array([1-contam_frac, contam_frac]))
    component_dists = dist.Normal(jnp.stack([mod,mod],axis = -1), jnp.stack([rms_new,c*rms_new],axis = -1) )

    with handlers.mask(mask = mask):
        loss = sample("Loss", dist.MixtureSameFamily(mixture_dists, component_dists), obs=data)
    return loss