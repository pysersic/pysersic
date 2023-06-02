from typing import Iterable, Union

import jax.numpy as jnp
from astropy.io import fits

from pysersic import FitMulti, FitSingle, priors

from .loss import gaussian_loss


def search_start(list_of_strings: Iterable, to_match: str)-> list:
    """Search a list of strings and match those which start with to_match

    Parameters
    ----------
    list_of_strings : Iterable
    to_match : str
        pattern to match

    Returns
    -------
    list
        List of results contianing the index and full string
    """
    result = []
    for j,string in enumerate(list_of_strings):
        if string.split()[0] == to_match:
            result.append([j,string])
    return result

def load_fits_from_galfit_line(line:str) -> jnp.array:
    """Load fits file from galfit config line

    Parameters
    ----------
    line : str

    Returns
    -------
    jnp.array
        Image in first HDU
    """
    fname = line.split()[1]
    return jnp.array( fits.getdata(fname) )

def match_line_and_load_galfit(list_of_strings: Iterable, to_match: str, assert_message: str = "")-> jnp.array:
    """Combine the above two functions to search and load fits files from a galfit config

    Parameters
    ----------
    list_of_strings : Iterable
    to_match : str
    assert_message : str, optional
        Message to raise if line file is not present

    Returns
    -------
    jnp.array
        _description_

    Raises
    ------
    UserWarning
        _description_
    """
    matches = search_start(list_of_strings, to_match)
    line = matches[0][1]
    if line.split()[1] == 'none':
        raise UserWarning(assert_message)
    return load_fits_from_galfit_line(line)

def generate_fitter_from_galfit_config(config_loc: str, loss: callable = gaussian_loss ) -> Union[FitSingle,FitMulti]:
    """Function to generate a Fitter instance from a galfit config file. This function is only meant to be a starting place to easily try PySersic on the many galfit config files we know you all have lying around. As such this does not follow all of the rules and constraints the galfit takes into account. For more fine grained control we recommend initializing a pysersic Fitter instance yourself, please see the `examples/` folder or the documentation for some help with this. 

    Parameters
    ----------
    config_loc : str
        Config file locations
    loss : callable, optional
        Loss function to use, see loss.py for more details, by default gaussian_loss

    Returns
    -------
    Union[FitSingle,FitMulti]
        Fitter based on config file. If there is only one source then returns a FitSingle object, otherwise returns a fit_multi
    """
    
    with open(config_loc,'r') as f:
        lines = []
        for line in f.readlines():
            if line[0] == '#' or line[:1] == '\n' or line[0] == '=':
                continue
            else:
                lines.append(line)
    
    im = match_line_and_load_galfit(lines, 'A)', assert_message='Missing science image, labelled D) in config file')
    rms = match_line_and_load_galfit(lines, 'C)', assert_message='Missing rms image, labelled C) in config file')
    psf = match_line_and_load_galfit(lines,'D)', assert_message='Missing psf image, labelled D) in config file')

    try:
        mask = match_line_and_load_galfit(lines,'F)')
    except:
        mask = jnp.zeros_like(im)
    zpt = float ( search_start(lines, 'J)')[0][1].split()[1] )
    source_list = search_start(lines, '0)')
    
    sky_type = 'none'
    source_dict = {}
    source_dict['type'] = []
    source_dict['flux'] = []
    source_dict['r'] = []
    source_dict['x'] = []
    source_dict['y'] = []

    for i,source in enumerate(source_list):
        ind = source[0]
        
        if i == len(source_list) -1:
            next_ind = None
        else:
            next_ind = source_list[i+1][0]
        source_string_list = lines[ind:next_ind]

        source_type = source[1].split()[1]
        if source_type == 'sersic' or 'dev' in source_type or 'exp' in source_type or source_type == 'psf':
            if 'dev' in source_type:
                source_type = 'dev'
            elif 'exp' in source_type:
                source_type = 'exp'
            elif source_type == 'psf':
                source_type = 'pointsource'

            mag = float( search_start(source_string_list, '3)')[0][1].split()[1] )
            flux = 10.**( (mag - zpt)/-2.5)
            
            x,y = search_start(source_string_list, '1)')[0][1].split()[1:3]
            x = float(x)
            y = float(y)
    
            source_dict['type'].append(source_type)
            source_dict['x'].append(x)
            source_dict['y'].append(y)
            source_dict['flux'].append(flux)

            if source_type != 'pointsource':
                r = float( search_start(source_string_list, '4)')[0][1].split()[1] )
            else:
                r = -99
            source_dict['r'].append(r)

        elif source_type == 'sky':
            print('sky')
            fit_back = search_start(source_string_list, '1)')[0][1].split()[2]
            back_init = float(search_start(source_string_list, '1)')[0][1].split()[1])
            fit_x_sl = search_start(source_string_list,'2)')[0][1].split()[2]
            fit_y_sl = search_start(source_string_list,'2)')[0][1].split()[2]
            if fit_y_sl == '1' or fit_x_sl == '1':
                sky_type = 'tilted-plane'
            elif fit_back == '1':
                sky_type = 'flat'
            else:
                sky_type = 'none'
        else:
            print (f"Skipping source {i}, profile type of {source_type} currently not supported")

    N_sources = len(source_dict['type'])
    types = source_dict['type']
    print(f"Found {N_sources} source(s) to fit of type(s) {types}")

    if len(source_dict['type']) == 1:
        prof_type = source_dict['type'][0]
        sp = priors.SourceProperties(im, mask = mask)
        sp.set_sky_guess(sky_guess = 0, sky_guess_err = 1)
        sp.set_flux_guess(flux_guess= source_dict['flux'][0])
        sp.set_position_guess( [source_dict['x'][0],source_dict['y'][0] ])

        if prof_type != 'psf':
            sp.set_r_eff_guess(r_eff_guess=source_dict['r'][0])
        prior = sp.generate_prior(prof_type, sky_type = sky_type)
        ps_fitter = FitSingle(data = im, rms = rms, psf = psf, prior = prior, loss_func = loss )

    elif len(source_dict['type']) > 1:
        multi_prior = priors.PySersicMultiPrior(catalog=source_dict, sky_type = sky_type,sky_guess=back_init, sky_guess_err = back_init + 1e-5)
        ps_fitter = FitMulti(data = im, rms = rms, psf = psf, prior = multi_prior, loss_func = loss)

    else:
        print ('Did not find any viable sources in galfit config')
        ps_fitter = 0
    
    return ps_fitter
