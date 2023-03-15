import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


def get_bounds(im,scale):
    m = np.mean(im)
    s = np.std(im)
    vmin = m - scale*s 
    vmax = m+scale*s 
    return vmin, vmax

def plot_image(image,mask,sig,psf,cmap='gray_r',scale=1.0,size=8):
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


def plot_residual(image,model,mask=None,scale=1.0,cmap='gray_r',colorbar=True,**resid_plot_kwargs):
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