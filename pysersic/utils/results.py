# I'm imagineing some class to plot results and compare model to data
# In imcascade the results class injests the fitter class which works well I think but definetly open to suggestions.
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import corner
import jax.numpy as jnp

class PySersicResults():
    def __init__(self,result,svi,guide,infodict):
        self.result = result
        self.params = result.params 
        self.svi = svi 
        self.guide = guide 
        self.infodict = infodict
        
    def quantiles(self,quantiles):
        return self.guide.quantiles(self.params,quantiles)
    def mean_model(self):
        mean_params = self.quantiles([0.5])
        param_dict = {}
        for i in mean_params.keys():
            param_dict[i] = mean_params[i][0]
        #bf = FitSersic.Sersic2D(self.infodict['xgrid'],self.infodict['ygrid'],**param_dict,psf_fft=self.infodict['psf_fft'])
        #return bf, param_dict

    def plot_bestfit(self):
        bf, param_dict = self.mean_model() 
        d = self.infodict['data']
        fig, ax = plt.subplots(1,3,figsize=(10,3),constrained_layout=True)
        ax[0].imshow(d,cmap='gray',origin='lower',vmin=jnp.mean(d)-3*jnp.std(d),vmax=jnp.mean(d)+3*jnp.std(d))
        ax[1].imshow(bf,cmap='gray',origin='lower',vmin=jnp.mean(bf)-3*jnp.std(bf),vmax=jnp.mean(bf)+3*jnp.std(bf))
        im2 = ax[2].imshow(d-bf,cmap='seismic',origin='lower',vmin=jnp.mean(d-bf)-3*jnp.std(d-bf),vmax=jnp.mean(d-bf)+3*jnp.std(d-bf))
        ax_divider = make_axes_locatable(ax[2])
        cax1 = ax_divider.append_axes("right", size="7%", pad="2%")
        cb1 = fig.colorbar(im2, cax=cax1)
        return fig, ax

    def corner(self):
        return corner.corner(self.inf_data)
        


        