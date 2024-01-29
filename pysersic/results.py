import copy
from typing import Callable, Optional, Tuple, Union

import arviz as az
import asdf
import re
import corner
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import pandas as pd
import xarray
from jax import random
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from .priors import PySersicMultiPrior, base_profile_params
from .rendering import BaseRenderer

ArrayLike = Union[np.array, jax.numpy.array]
ListLike = Union[np.array,jax.numpy.array,list]

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

    def add_method_used(self,method):
        self.svi_method_used = method 


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
                num_sample: Optional[int] = 1_000,
                rkey: Optional[jax.random.PRNGKey] = random.PRNGKey(5)
        ) -> pd.DataFrame:
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
        num_sample: Optional[int]
            Number of samples to draw from trained SVI posterior, no effect if sampling was used.
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
            self.input  = sampler
            self.idata = self._parse_injested_data(self.idata,purge_extra=purge_extra)
            self.runtype = 'sampling'
        else:
            assert 'guide' in svi_res_dict.keys()
            assert 'model' in svi_res_dict.keys()
            assert 'svi_result' in svi_res_dict.keys()
            self.input  = svi_res_dict
            post_raw = svi_res_dict['guide'].sample_posterior(rkey, svi_res_dict['svi_result'].params, sample_shape = ((num_sample,)))
            #Convert to arviz
            post_dict = {}
            for key in post_raw:
                post_dict[key] = post_raw[key][jnp.newaxis,]
            self.idata = az.from_dict(post_dict)
            self.idata = self._parse_injested_data(self.idata,purge_extra=purge_extra)
            self.runtype='svi'

        return

    def _parse_injested_data(self,data:az.InferenceData, purge_extra:bool = True, save_model: bool = True)->az.InferenceData:
        """Helper function to postprocess the poterior object (internal use).

        Parameters
        ----------
        data : arviz.InferenceData
            _description_
        purge_extra : bool, optional
            whether to purge extra params not part of the fitting, by default True
        save_model : bool
            Whether to set self.models with model images from posterior
        Returns
        -------
        arviz.InferenceData
            the cleaned up object
        """
        var_names = list( data.posterior.data_vars )
        
        for var in var_names:
            if 'theta' in var:
                new_theta = np.remainder(data['posterior'][var]+np.pi, np.pi)
                data['posterior'][var] = new_theta

        if purge_extra:
            to_drop = []
            for var in var_names:
                if ('base' in var) or ('auto' in var) or ('unwrapped' in var):
                    to_drop.append(var)
                elif 'model' in var:
                    to_drop.append(var)
                    if save_model:
                        self.models = data['posterior'][var]
            data.posterior = data.posterior.drop_vars(to_drop).drop_dims(['model_dim_0','model_dim_1'], errors = 'ignore')
        return data


    def summary(self)->pd.DataFrame:
        """Convenience function for returning the summary dataframe using the arviz summary.

        Returns
        -------
        pandas.DataFrame
            data frame containing the arviz summary of the fit.
        """
        return az.summary(self.idata)
    

    def get_median_model(self,):
        print('This function has been deprecated as it was buggy and not implemented well. If you require a model we suggest using Fit[x].find_map() of looking at the models from the posterior in PySersicResults.models')
        print('Please reach out if you have any questions.')
        return NotImplementedError

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
        xx = self.idata.posterior.quantile(quantiles, dim = ['chain','draw']).to_dict()
        out = {} 
        for i in xx['data_vars'].keys():
            out[i] = xx['data_vars'][i]['data']
        if return_dataframe==False:
            return out
        else:
            df = pd.DataFrame.from_dict(out).T
            df.columns = quantiles
            return df 

    def retrieve_med_std(self,return_dataframe:bool=False)->Union[pd.DataFrame,dict]:
        out = {}
        q_dict = self.retrieve_param_quantiles()
        for key in q_dict.keys():
            qs = q_dict[key]
            if isinstance(qs[0], list):
                qs = np.array(qs)
            med = qs[1]
            std = 0.5*np.abs(qs[2] - qs[0])
            out[key] = np.array([med,std])
        if return_dataframe:
            out = pd.DataFrame.from_dict(out).T.rename(columns= {0:'median',1:'std'})
        return out


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
        tree['method_used'] = self.runtype
        if self.runtype == 'svi':
            tree['svi_method_used'] = self.svi_method_used
        tree['prior_info'] = self.prior.__str__()
        tree['posterior'] = self.idata.to_dict()['posterior']
        for i in tree['posterior']:
            i = np.array(i)
        af = asdf.AsdfFile(tree=tree)
        if not fname.endswith('.asdf'):
            fname+='.asdf'
        af.write_to(fname)

    def sample_posterior(self, num_sample: int, purge_extra: Optional[bool] = True, rkey: Optional[jax.random.PRNGKey] = random.PRNGKey(7))-> az.InferenceData:
        """Generate extra samples from an trained SVI posterior

        Parameters
        ----------
        num_sample : int
            number of samples to draw
        purge_extra : Optional[bool], optional
            Whether to purge variables containing 'auto', 'base' or 'unwrapped' often used in reparamaterization, by default True
        rkey : Optional[jax.random.PRNGKey], optional
            PRNG key to use, by default jax.random.PRNGKey(7)

        Returns
        -------
        az.InferenceData
            arviz InferenceData object containing posterior
        """
        assert self.runtype == 'svi', "Can only add samples if SVI was used for inference"
        post_raw = self.input['guide'].sample_posterior(rkey, self.input['svi_result'].params, sample_shape = ((num_sample,)))
        #Convert to arviz
        post_dict = {}
        for key in post_raw:
            post_dict[key] = post_raw[key][jnp.newaxis,]
        idata = az.from_dict(post_dict)
        idata = self._parse_injested_data(idata,purge_extra=purge_extra, save_model= False)
        return idata

def parse_multi_results(results: PySersicResults, source_num: int) -> PySersicResults:
    """Function written to parse results from a FitMulti instance to isolate a single source. A new PySersicResults class is created with only the posteriors of the specified source. The original chains saved under `.idata_all`

    Parameters
    ----------
    results : PySersicResults
        Results class from a FitMulti instance
    source_num : int 
        Source number to isolate or if equal to -1 will reset to the joint posterior of all sources

    Returns
    -------
    PySersicResults
        Results class with all the meta-data the same but the specified source isolated. The original posterior is saved under `.idata_all`    
    """
    new_res = copy.copy(results)

    assert type(results.prior) == PySersicMultiPrior , "Results must be from a FitMulti instance"
    
    if source_num == -1:
        if not hasattr(new_res, 'idata_all'):
           print("No need to reset posterior, returning original")
        else:
            idata_all = copy.copy(new_res.idata_all)
            new_res.__setattr__('idata', idata_all)
            new_res.__delattr__('idata_all')
    else:
        #Select variables for source
        prof_type = new_res.prior.catalog['type'][source_num]
        param_names = base_profile_params[prof_type]
        source_names = [pname + f'_{source_num}' for pname in param_names]

        #Search for other meta variables like sky etc.
        meta_names = []
        for k in new_res.idata.posterior.keys():
            if re.search(f"_[0-{new_res.prior.N_sources:d}]", k) is None:
                meta_names.append(k)

        if hasattr(new_res, 'idata_all'):
            idata = new_res.idata_all
        else:
            idata = new_res.idata
            new_res.__setattr__('idata_all', idata)

        
        post_source = az.extract(idata, var_names = source_names+meta_names , combined = False)
        idata_source = az.InferenceData(posterior = post_source)
        idata_source.rename_vars(dict(zip(source_names,param_names)), inplace = True)

        new_res.__delattr__('idata')
        new_res.__setattr__('idata', idata_source)
    return new_res


def get_bounds(im: np.array ,scale: float) -> Tuple[float,float]:
    """Generate bounds based on image mean and standard deviation

    Parameters
    ----------
    im : np.array
        images
    scale : float
        Number of +/- sigmas for bounds

    Returns
    -------
    Tuple[float,float]
       Bounds to use
    """
    m = np.mean(im)
    s = np.std(im)
    vmin = m - scale*s 
    vmax = m+scale*s 
    return vmin, vmax

def plot_image(image: np.array,mask: np.array,sig: np.array,psf: np.array,cmap:str ='gray_r',scale:float = 2.0,size:float = 8.) -> Tuple[plt.Figure,plt.Axes]:
    """Plot a summary figure with the image, sigma map and psf side by side

    Parameters
    ----------
    image : np.array
        Image to plot
    mask : np.array
        Mask
    sig : np.array
        sigma or noise map
    psf : np.array
        Point spread function
    cmap : str, optional
        Color map to use, by default 'gray_r'
    scale : float, optional
        Number of +/- std's of image to make the bounds, by default 2.0
    size : float, optional
        Size of figure, will be size*3 x size, by default 8.

    Returns
    -------
    Tuple[plt.Figure,plt.Axes]
        Figure and axes objects
    """
    im_ratio = image.shape[0]/image.shape[1]
    fig, ax = plt.subplots(1,3,figsize=(size*3,size*im_ratio))
    masked_image = np.ma.masked_array(image,mask)
    masked_sigma = np.ma.masked_array(sig,mask)
    im_vmin, im_vmax = get_bounds(masked_image,scale)
    sig_vmin,sig_vmax = get_bounds(masked_sigma,scale)
    psf_vmin,psf_vmax = get_bounds(psf,scale)
    ax[0].imshow(masked_image,origin='lower',cmap=cmap,vmin=im_vmin,vmax=im_vmax)
    ax[1].imshow(masked_sigma,origin='lower',cmap=cmap,vmin=sig_vmin,vmax=sig_vmax)
    ax[2].imshow(psf,origin='lower',cmap=cmap,vmin=psf_vmin,vmax=psf_vmax)
    return fig, ax


def plot_residual(image: np.array,model: np.array,mask: np.array = None,scale:float =2.0,cmap:str ='gray_r',colorbar:bool =True,**resid_plot_kwargs)-> Tuple[plt.Figure,plt.Axes]:
    """Generate a summary plot comparing the data to the best fit model

    Parameters
    ----------
    image : np.array
        Original image
    model : np.array
        Best fit model image
    mask : np.array, optional
        Pixel by Pixel mask, by default None
    scale : float, optional
        Number of +/- std's to make the bounds of the image, by default 2.0
    cmap : str, optional
        color map to use, by default 'gray_r'
    colorbar : bool, optional
        Whether or not to show a color bar, by default True

    Returns
    -------
    Tuple[plt.Figure,plt.Axes]
        Figure and axes objects
    """
    fig, ax = plt.subplots(1,3,figsize=(13,3))
    if mask is not None:
        masked_image = np.ma.masked_array(image,mask)
        masked_model = np.ma.masked_array(model,mask)
    else:
        masked_image = image 
        masked_model = model 
    im_vmin, im_vmax = get_bounds(masked_image,scale)
    ax[0].imshow(masked_image,origin='lower',cmap=cmap,vmin=im_vmin,vmax=im_vmax)
    ax[1].imshow(masked_model,origin='lower',cmap=cmap,vmin=im_vmin,vmax=im_vmax)
    residual = masked_image - masked_model 
    ri = ax[2].imshow(residual,origin='lower',cmap='seismic',**resid_plot_kwargs)
    ax1_divider = make_axes_locatable(ax[2])
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
    cb1 = fig.colorbar(ri, cax=cax1)
    return fig, ax 