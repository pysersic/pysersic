
from numpyro import distributions as dist, infer,sample,deterministic
import jax.numpy as jnp 


def autoprior(image,verbose=False):
    """
    Derive automatic priors based on an input image.
    """
    flux_guess = jnp.sum(image)
    flux_prior = dist.TruncatedNormal(scale = flux_guess/3, loc = flux_guess,low = 0 )
    image_dim_min = jnp.min(jnp.array([image.shape[0],image.shape[1]])) 
    reff_prior = dist.TruncatedNormal(loc = image_dim_min/10.,scale = 2, low = 1,high = image_dim_min/4.0) 

    ellip_prior = dist.Uniform(0,0.8) 
    theta_prior = dist.Uniform(-jnp.pi/2.,jnp.pi/2.) 
    n_prior = dist.TruncatedNormal(loc = 2, scale= 0.5, low = 0.5, high=8) 
    x0_max = image.shape[0]/2
    y0_max = image.shape[1]/2

    x0_prior = dist.Normal(loc = x0_max) 
    y0_prior =  dist.Normal(loc = y0_max)  

    prior_dict = {
        'flux': flux_prior,
        'r_eff': reff_prior,
        'n': n_prior,
        'ellip': ellip_prior,
        'theta': theta_prior,
        'x_0': x0_prior,
        'y_0': y0_prior
    }
    return prior_dict 



class BatchPriors():
    def __init__(self,catalog,pos_sigma=5.0,rpix_sigma=5.0,ellip_sigma=0.2,theta_sigma=jnp.pi/2,logflux_sigma=0.5):
        """
        Ingest a catalog-like data structure containing prior positions and parameters for multiple galaxies in a single image.
        The format of the catalog can be a `pandas.DataFrame`, `numpy` RecordArray, dictionary, or any other format so-long as 
        the following fields exist and can be directly indexed: 'X', 'Y', 'LOGFLUX', and 'R'. Optional additional fields include
        'THETA' and 'ELLIP' -- if these are on hand they may be provided. Theta must be in radians. 
        If only some galaxies have these measured, the flag -99.0 can be used to indicate as such. 
        All columns/fields must be the same length.

        Parameters
        ----------
        catalog: any
            structured data containing accessible fields as described above. Must have a len(). 
        pos_sigma: float, default: 5.0
            sigma for a normally-distributed prior on central position x0, y0, in pixels.
        rpix_sigma: float, default: 5.0
            sigma for a normally-distributed prior on r_eff,  in pixels. 
        ellip_sigma: float, default: 0.2
            sigma for a (truncated) normally-distributed prior on ellipticity. 
        theta_sigma: float, default: jnp.pi/2 
            sigma for a normally-distributed prior on theta. 
        logflux_sigma: float, default: 0.5
            sigma for a normally-distributed prior on the log of the total flux.

        
        """
        self.catalog = catalog 
        self.pos_sigma = pos_sigma
        self.rpix_sigma = rpix_sigma 
        self.ellip_sigma = ellip_sigma
        self.theta_sigma = theta_sigma
        self.logflux_sigma = logflux_sigma

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





