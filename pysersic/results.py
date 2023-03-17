# I'm imagineing some class to plot results and compare model to data
# In imcascade the results class injests the fitter class which works well I think but definetly open to suggestions.
import matplotlib.pyplot as plt
from typing import Callable, Optional, Union
import numpyro 
import jax 
from jax import random 
import pandas 
import numpy as np 
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import corner
import jax.numpy as jnp
import arviz as az 
ArrayLike = Union[np.array, jax.numpy.array]
ListLike = Union[np.array,jax.numpy.array,list]
from pysersic.rendering import (
    BaseRenderer,
    FourierRenderer,
    HybridRenderer,
)
from pysersic.utils import gaussian_loss, train_numpyro_svi_early_stop 
import asdf 
import pandas as pd 
import xarray 

class PySersicResults():
    def __init__(self,
                data: ArrayLike,
                rms: ArrayLike,
                psf: ArrayLike,
                loss_func:Callable,
                renderer: BaseRenderer,
                mask: Optional[ArrayLike] = None,
                ):
        """Initialize Results Object

        Parameters
        ----------
        data : ArrayLike
            data from fitter
        rms : ArrayLike
            rms from fitter
        psf : ArrayLike
            psf from fitter
        loss_func : Callable
            the loss function chosen
        renderer : BaseRenderer
            the renderer created by the fitter
        mask : Optional[ArrayLike], optional
            mask, if it was used in the fitting, by default None
        """
        self.data = data 
        self.rms = rms 
        self.psf = psf 
        self.mask = mask 
        self.loss_func = loss_func 
        self.renderer = renderer




    def __repr__(self)->str:
        if not hasattr(self,'runtype'):
            self.runtype='unknown'
        out = f'PySersicResults object for pysersic fit of type: {self.runtype}\n'
        return out




    def add_prior(self,prior):
        """add the prior object to the result object once created

        Parameters
        ----------
        prior : pysersic.priors.BasePrior
            created prior object
        """
        self.prior = prior 
    def injest_data(self, 
                sampler: Optional[numpyro.infer.mcmc.MCMC] =  None, 
                svi_res_dict: Optional[dict] =  None,
                purge_extra: Optional[bool] = True,
                rkey: Optional[jax.random.PRNGKey] = random.PRNGKey(5)
        ) -> pandas.DataFrame:
        """Method to injest data from optimized SVI model or results of sampling. 
        When sampling data is input, sets the class attribute `sampling_results`, while for an SVI run, saves `svi_results.`
        A PysersicResults object can have at most data for one SVI and one sampling run. Both class atttributes are 
        instances of arviz.InferenceData objects.

        Parameters
        ----------
        sampler : Optional[numpyro.infer.mcmc.MCMC], optional
            numpyro sampler containing results
        svi_res_dict : Optional[dict], optional
            Dictionary containing 'guide', 'model' and 'svi_result' specifying a trained SVI model
        purge_extra : Optional[bool], optional
            Whether to purge variables containing 'auto', 'base' or 'unwrapped' often used in reparamaterization, by default True
        rkey : Optional[jax.random.PRNGKey], optional
            PRNG key to use, by default jax.random.PRNGKey(5)

        Returns
        -------
        pandas.DataFrame
            ArviZ Summary of results

        Raises
        ------
        AssertionError
            Must supply one of sampler or svi_dict
        """

        if sampler is None and (svi_res_dict is None):
            raise AssertionError("Must svi results dictionary or sampled sampler")

        elif sampler is not None:
            self.idata = az.from_numpyro(sampler)
            self.idata = self._parse_injested_data(self.idata,purge_extra=purge_extra)
            self.runtype = 'sampling'
        else:
            assert 'guide' in svi_res_dict.keys()
            assert 'model' in svi_res_dict.keys()
            assert 'svi_result' in svi_res_dict.keys()

            post_raw = svi_res_dict['guide'].sample_posterior(rkey, svi_res_dict['svi_result'].params, sample_shape = ((1000,)))
            #Convert to arviz
            post_dict = {}
            for key in post_raw:
                post_dict[key] = post_raw[key][jnp.newaxis,]
            self.idata = az.from_dict(post_dict)
            self.idata = self._parse_injested_data(self.idata,purge_extra=purge_extra)
            self.runtype='svi'

        return

    def _parse_injested_data(self,data:az.InferenceData,purge_extra:bool=True)->az.InferenceData:
        """Helper function to postprocess the poterior object (internal use).

        Parameters
        ----------
        data : arviz.InferenceData
            _description_
        purge_extra : bool, optional
            whether to purge extra params not part of the fitting, by default True

        Returns
        -------
        arviz.InferenceData
            the cleaned up object
        """
        var_names = list(data.posterior.to_dataframe().columns)
        
        for var in var_names:
            if 'theta' in var:
                new_theta = np.remainder(data['posterior'][var]+np.pi, np.pi)
                data['posterior'][var] = new_theta

        if purge_extra:
            to_drop = []
            for var in var_names:
                if ('base' in var) or ('auto' in var) or ('unwrapped' in var):
                    to_drop.append(var)

            data.posterior = data.posterior.drop_vars(to_drop)
        return data


    def summary(self)->pd.DataFrame:
        """Convenience function for returning the summary dataframe using the arviz summary.

        Returns
        -------
        pandas.DataFrame
            data frame containing the arviz summary of the fit.
        """
        return az.summary(self.idata)
    

    def render_best_fit_model(self,)->ArrayLike:
        """Create a model image using the median posterior values of the parameters.

        Returns
        -------
        ArrayLike
            model image
        """
        medians = self.idata.posterior.median() 
        median_params = jnp.array([medians[name].data for name in self.prior.param_names])
        mod = self.renderer.render_source(median_params, self.prior.profile_type)
        return mod
    

    def corner(self,quantiles=[.16,.50,.84,],**kwargs):
        """Return a corner plot of the parameter estimation

        Parameters
        ----------
        quantiles: ListLike, optional  
            which quantiles to mark on the corner plot, can pass None, by default [0.16,0.5,0.84]
        **kwargs:
            any additional arguments to pass to corner (for the single plot case).
        Returns
        -------
        matplotlib.figure
            fig object containing the corner plots
        """

        return corner.corner(self.idata,show_titles=True,quantiles=quantiles,**kwargs)

        
    def retrieve_param_quantiles(self,
                                quantiles:ListLike=[0.16,0.5,0.84],
                                return_dataframe:bool=False)->Union[pd.DataFrame,dict]:
        """retrieve quantiles on the parameter estimation

        Parameters
        ----------
        quantiles : ListLike, optional
            array of quantiles to pull, must be between 0 and 1, by default [0.16,0.5,0.84]
        return_dataframe : bool, optional
            whether to return dataframe instead of simple dict, by default False

        Returns
        -------
        Union[pd.DataFrame,dict]
            dict or dataframe with index/keys as parameters and columns/values as the chosen quantiles.
        """
        r = self.idata
        xx = r.posterior.quantile(quantiles).to_dict()
        out = {} 
        for i in xx['data_vars'].keys():
            out[i] = xx['data_vars'][i]['data']
        if return_dataframe==False:
            return out
        else:
            df = pd.DataFrame.from_dict(out).T
            df.columns = quantiles
            return df 


    def latex_table(self,quantiles:ListLike=[0.16,0.5,0.84]):
        """
        Generate a simple AASTex deluxetable with the fit parameters. Prints the result.

        Parameters
        ----------
        quantiles : ListLike, optional
            quantiles to use must be len 3 as we do upper-median and median-lower to get +/- values, by default [0.16,0.5,0.84]

        Raises
        ------
        AssertionError
            if the quantile list does not have three values 
        """
        out = "\\begin{deluxetable}{lr}[b]\n"
        out+= "\\tablehead{\n"
        out+= "\colhead{Parameter} & \colhead{\hspace{4.5cm}Value\hspace{.5cm}}}\n"
        out+="\caption{Best Fit Parameters for Pysersic Fit}\n"
        out+="\startdata \n"
        df = self.retrieve_param_quantiles(quantiles=quantiles,return_dataframe=True)
        if len(df.columns)!=3:
            raise AssertionError('Must Choose 3 quantile positions for +/- calculation')
        for i in df.index:
            q0 = quantiles[0]
            q1 = quantiles[1]
            q2 = quantiles[2]
            plus = df.loc[i,q2] - df.loc[i,q1]
            minus = df.loc[i,q1] - df.loc[i,q0]
            if '_' in i:
                use = i.split('_')[0] + f"_{{\\rm {i.split('_')[1]}}}"
            else:
                use=i
            if i =='theta':
                use = r'\theta'
            out+=f"{use} & {df.loc[i,0.50]:.3f}_{{-{minus:.3f}}}^{{+{plus:.3f}}} \\\\ \n"
        out+="\enddata \n"
        out+="\end{deluxetable}"
        print(out)

    def get_chains(self)->xarray.Dataset:
        """
        Wrapper for az.extract, producing the chains/draws for the run

        Returns
        -------
        xarray.Dataset
            chain object
        """

        return az.extract(self.idata)

    
    def compute_statistic(self,parameter:str,func:Callable,)->ArrayLike:
        """
        Compute an arbitrary array statistic on the chain for a given parameter.
        For example, the std of all ellipticity draws. or the mean of the sersic draws

        Parameters
        ----------
        parameter : str
            a legal parameter from the fit. Use e.g. `results.svi_summary()` to see them.
        func : Callable
            any function which reads in array-like data and computes something

        Returns
        -------
        ArrayLike
            the computed statistic(s)
        """
        chain = self.get_chains()
        return func(chain[parameter]).data
    
    
    def save_result(self,fname:str):
        """Save summary of the fit, copies of the data, the chains, and some other info about priors and renderers
        into an asdf file for later retrieval. 

        Parameters
        ----------
        fname : str
            filename to save to.
        """
        tree = {} 
        tree['input_data'] = {} 
        tree['input_data']['image'] = np.array(self.data)
        tree['input_data']['rms'] = np.array(self.rms)
        tree['input_data']['psf'] = np.array(self.psf)
        tree['input_data']['mask'] = np.array(self.mask)
        tree['loss_func'] = str(self.loss_func)
        tree['renderer'] = str(self.renderer)
        tree['contains_SVI_result'] =  self.runtype == 'svi'
        tree['contains_sampling_result'] =  self.runtype=='sampling'
        tree['prior_info'] = self.prior.__str__()
        tree['best__model'] = np.array(self.render_best_fit_model())
        tree['posterior'] = self.idata.to_dict()['posterior']
        for i in tree['posterior']:
            i = np.array(i)
        af = asdf.AsdfFile(tree=tree)
        if not fname.endswith('.asdf'):
            fname+='.asdf'
        af.write_to(fname)