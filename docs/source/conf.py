# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'pysersic'
copyright = '2023, Imad Pasha & Tim Miller'
author = 'Imad Pasha & Tim Miller'

# The full version, including alpha/beta/rc tags
release = '0.1'


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['nbsphinx','autoapi.extension', 'sphinxcontrib.napoleon',
 'sphinx.ext.autodoc', 'sphinx.ext.inheritance_diagram','nbsphinx_link']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['.ipynb_checkpoints/*', '*.asdf', '*.log','*.npy']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

nbsphinx_prompt_width = 0 # no prompts in nbsphinx
nbsphinx_execute = 'never'
html_theme_options = {'body_max_width': 'auto'}
master_doc = 'index'

autoapi_dirs = ['../../pysersic']
autoapi_ignore = ["*checkpoint*"]
#autoapi_options = ['members','private-members','show-inheritance','show-module-summary','special-members','imported-members', ]
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']