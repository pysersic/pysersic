
from numpyro import distributions as dist, sample
import jax.numpy as jnp 

def autoprior(image,profile_type):
    if profile_type == 'sersic':
        prior_dict = generate_sersic_prior(image)
    
    elif profile_type == 'doublesersic':
        prior_dict = generate_doublesersic_prior(image)

    elif profile_type == 'pointsource':
        prior_dict = generate_pointsource_prior(image)
    
    return prior_dict

def generate_sersic_prior(image, flux_guess = None, r_eff_guess = None, position_guess = None):
    """
    Derive automatic priors for a sersic profile based on an input image.
    """

    if flux_guess is None:
        flux_guess = jnp.sum(image)
    flux_prior = dist.TruncatedNormal(scale = flux_guess/3, loc = flux_guess,low = 0 )
    

    image_dim_min = jnp.min(jnp.array([image.shape[0],image.shape[1]])) 
    if r_eff_guess is None:
        r_eff_guess = image_dim_min/10
    reff_prior = dist.TruncatedNormal(loc =  r_eff_guess,scale =  r_eff_guess/3, low = 1,high = image_dim_min/4.0) 

    ellip_prior = dist.Uniform(0,0.8) 
    theta_prior = dist.Uniform(-jnp.pi/2.,jnp.pi/2.) 
    n_prior = dist.TruncatedNormal(loc = 2, scale= 0.5, low = 0.5, high=8)

    if position_guess is None:
        x0_guess = image.shape[0]/2
        y0_guess = image.shape[1]/2
    else:
        x0_guess = position_guess[0]
        y0_guess = position_guess[1]
    x0_prior = dist.Normal(loc = x0_guess) 
    y0_prior =  dist.Normal(loc = y0_guess)  

    prior_dict = {
        'x_0': x0_prior,
        'y_0': y0_prior,
        'flux': flux_prior,
        'r_eff': reff_prior,
        'n': n_prior,
        'ellip': ellip_prior,
        'theta': theta_prior,
    }
    return prior_dict 

def generate_doublesersic_prior(image, flux_guess = None, r_eff_guess = None, position_guess = None):
    """
    Derive automatic priors for a double-sersic profile based on an input image.
    """

    if flux_guess is None:
        flux_guess = jnp.sum(image)
    flux_prior = dist.TruncatedNormal(scale = flux_guess/3, loc = flux_guess,low = 0 )
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
        x0_guess = image.shape[0]/2
        y0_guess = image.shape[1]/2
    else:
        x0_guess = position_guess[0]
        y0_guess = position_guess[1]
    x0_prior = dist.Normal(loc = x0_guess) 
    y0_prior =  dist.Normal(loc = y0_guess)  

    prior_dict = {
        'x_0': x0_prior,
        'y_0': y0_prior,
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
    flux_prior = dist.TruncatedNormal(scale = flux_guess/3, loc = flux_guess,low = 0 )

    if position_guess is None:
        x0_guess = image.shape[0]/2
        y0_guess = image.shape[1]/2
    else:
        x0_guess = position_guess[0]
        y0_guess = position_guess[1]
    x0_prior = dist.Normal(loc = x0_guess) 
    y0_prior =  dist.Normal(loc = y0_guess)  

    prior_dict = {
        'x_0': x0_prior,
        'y_0': y0_prior,
        'flux': flux_prior,
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




def sample_sersic(prior_dict):
    flux = sample('flux', prior_dict['flux'])
    n = sample('n',prior_dict['n'])
    r_eff = sample('r_eff',prior_dict['r_eff'])
    ellip = sample('ellip',prior_dict['ellip'])
    theta = sample('theta',prior_dict['theta'])
    x_0 = sample('x_0',prior_dict['x_0'])
    y_0 = sample('y_0',prior_dict['y_0'])

    #collect params and render scene
    params = jnp.array([x_0,y_0,flux,r_eff,n, ellip, theta])
    return params

def sample_doublesersic(prior_dict):
    flux = sample('flux', prior_dict['flux'])
    f_1 = sample('f_1', prior_dict['f_1'])
    n_1 = sample('n_1',prior_dict['n_1'])
    r_eff_1 = sample('r_eff_1',prior_dict['r_eff_1'])
    ellip_1 = sample('ellip_1',prior_dict['ellip_1'])
    n_2 = sample('n_2',prior_dict['n_2'])
    r_eff_2 = sample('r_eff_2',prior_dict['r_eff_2'])
    ellip_2 = sample('ellip_2',prior_dict['ellip_2'])
    theta = sample('theta',prior_dict['theta'])
    x_0 = sample('x_0',prior_dict['x_0'])
    y_0 = sample('y_0',prior_dict['y_0'])

    params = jnp.array([x_0, y_0, flux, f_1, r_eff_1, n_1, ellip_1, r_eff_2, n_2, ellip_2, theta])
    return params

def sample_pointsource(prior_dict):
    flux = sample('flux', prior_dict['flux'])
    x_0 = sample('x_0',prior_dict['x_0'])
    y_0 = sample('y_0',prior_dict['y_0'])

    params = jnp.array([x_0,y_0,flux,])
    return params

sample_func_dict = {'sersic':sample_sersic,'doublesersic':sample_doublesersic, 'pointsource':sample_pointsource}
