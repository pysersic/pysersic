
from numpyro import distributions as dist, infer
import jax.numpy as jnp 



def autoprior(image,verbose=False):
    """
    Derive automatic priors based on an input image.
    """
    image_sum = jnp.sum(image)
    log_flux_prior = dist.Uniform(jnp.log10(image_sum)-0.5,jnp.log10(image_sum)+0.5)
    im_shape = image.shape
    image_dim_min = jnp.min(jnp.array([image.shape[0],image.shape[1]])) 
    reff_prior = dist.Uniform(1,image_dim_min/4.0) 

    ellip_prior = dist.Uniform(0,1)
    theta_prior = dist.Uniform(0,jnp.pi)
    log_n_prior = dist.Uniform(0,1)
    max_pix = jnp.argmax(image)
    unraveled = jnp.unravel_index(max_pix,image.shape)
    x0_max = unraveled[1]
    y0_max = unraveled[0]

    x0_prior = dist.Uniform(x0_max-10,x0_max+10)
    y0_prior = dist.Uniform(y0_max-10,y0_max+10)

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
        print(f'x0: {x0_max-10} - {x0_max+10}')
        print(f'y0: {y0_max-10} - {y0_max+10}')

    return prior_dict 



class BatchPriors():
    def __init__(self,catalog):
        """
        Ingest a catalog-like data structure containing prior positions and parameters for multiple galaxies in a single image.
        The format of the catalog can be a `pandas.DataFrame`, `numpy` RecordArray, dictionary, or any other format so-long as 
        the following fields exist and can be directly indexed: 'X', 'Y', 'LOGFLUX', and 'R'. Optional additional fields include
        'THETA' and 'ELLIP' -- if these are on hand they may be provided. If only some galaxies have these measured, the flag -99
        can be used to indicate as such. All columns/fields must be the same length
        """
        self.catalog = catalog 
        self.pos_sigma = 5.0
        self.rpix_sigma = 5.0 
        self.ellip_sigma = 0.2
        self.theta_sigma = jnp.pi/2.0
        self.logflux_sigma = 0.5

        self.parse_catalog()         

        

    def parse_catalog(self):
        self.prior_dict = {} 
        try:
            theta_exists = self.catalog['THETA']
            theta_exists = True
        except KeyError:
            theta_exists=False
        try:
            ellip_exists = self.catalog['ELLIP']
            ellip_exists = True
        except KeyError:
            ellip_exists = False
        if theta_exists and ellip_exists:
            for i in range(len(self.catalog)):
                self.prior_dict[f'x0_{i}'] = dist.Normal(self.catalog['X'][i],self.pos_sigma)
                self.prior_dict[f'y0_{i}'] = dist.Normal(self.catalog['Y'][i],self.pos_sigma)
                self.prior_dict[f'log_flux_{i}'] = dist.Normal(self.catalog['LOGFLUX'][i],self.logflux_sigma)
                self.prior_dict[f'r_eff_{i}'] = dist.TruncatedNormal(self.catalog['R'][i], self.rpix_sigma,low=1) 
                if self.catalog['THETA'][i] != -99.0:
                    self.prior_dict[f'theta_{i}'] = dist.Normal(self.catalog['THETA'][i],self.theta_sigma)
                else:
                    self.prior_dict[f'theta_{i}'] = dist.Uniform(0,jnp.pi)
                if self.catalog['ELLIP'][i] != -99.0:
                    self.prior_dict[f'ellip_{i}'] = dist.TruncatedNormal(self.catalog['ELLIP'][i],self.ellip_sigma,low=0,high=1)
                else:
                    self.prior_dict[f'ellip_{i}'] = dist.Uniform(0,1)
        elif theta_exists:
            for i in range(len(self.catalog)):
                self.prior_dict[f'x0_{i}'] = dist.Normal(self.catalog['X'][i],self.pos_sigma)
                self.prior_dict[f'y0_{i}'] = dist.Normal(self.catalog['Y'][i],self.pos_sigma)
                self.prior_dict[f'log_flux_{i}'] = dist.Normal(self.catalog['LOGFLUX'][i],self.logflux_sigma)
                self.prior_dict[f'r_eff_{i}'] = dist.TruncatedNormal(self.catalog['R'][i], self.rpix_sigma,low=1) 
                self.prior_dict[f'ellip_{i}'] = dist.Uniform(0,1)
                if self.catalog['THETA'][i] != -99.0:
                    self.prior_dict[f'theta_{i}'] = dist.Normal(self.catalog['THETA'][i],self.theta_sigma)
                else:
                    self.prior_dict[f'theta_{i}'] = dist.Uniform(0,jnp.pi)
        elif ellip_exists: 
            for i in range(len(self.catalog)):
                self.prior_dict[f'x0_{i}'] = dist.Normal(self.catalog['X'][i],self.pos_sigma)
                self.prior_dict[f'y0_{i}'] = dist.Normal(self.catalog['Y'][i],self.pos_sigma)
                self.prior_dict[f'log_flux_{i}'] = dist.Normal(self.catalog['LOGFLUX'][i],self.logflux_sigma)
                self.prior_dict[f'r_eff_{i}'] = dist.TruncatedNormal(self.catalog['R'][i], self.rpix_sigma,low=1) 
                self.prior_dict[f'theta_{i}'] = dist.Uniform(0,jnp.pi)
                if self.catalog['ELLIP'][i] != -99.0:
                    self.prior_dict[f'ellip_{i}'] = dist.TruncatedNormal(self.catalog['ELLIP'][i],self.ellip_sigma,low=0,high=1)
                else:
                    self.prior_dict[f'ellip_{i}'] = dist.Uniform(0,1)
        else:
            for i in range(len(self.catalog)):
                self.prior_dict[f'x0_{i}'] = dist.Normal(self.catalog['X'][i],self.pos_sigma)
                self.prior_dict[f'y0_{i}'] = dist.Normal(self.catalog['Y'][i],self.pos_sigma)
                self.prior_dict[f'log_flux_{i}'] = dist.Normal(self.catalog['LOGFLUX'][i],self.logflux_sigma)
                self.prior_dict[f'r_eff_{i}'] = dist.TruncatedNormal(self.catalog['R'][i], self.rpix_sigma,low=1) 
                self.prior_dict[f'ellip_{i}'] = dist.Uniform(0,1)
                self.prior_dict[f'theta_{i}'] = dist.Uniform(0,jnp.pi)
        return self.prior_dict





