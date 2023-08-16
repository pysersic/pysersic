import setuptools

setuptools.setup(
    name="pysersic",
    version="0.1.1",
    author="Imad Pasha & Tim Miller",
    author_email="imad.pasha@yale.edu",
    description="A Tool for fitting sersic profiles in python",
    long_description="""pysersic is a Python package for fitting Sersic (and other) profiles to astronomical images using Bayesian inference. It is built using the jax framework with inference performed using the numpyro probabilistic programming library. The code is hosted on GitHub and is available open source under the MIT license.""",
    long_description_content_type='text/markdown',
    download_url="https://github.com/pysersic/pysersic/archive/refs/tags/v0.1.1.tar.gz",
    packages=["pysersic",],
    install_requires=['numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'astropy',
        'corner',
        'photutils>=0.16',
        'jax',
        'numpyro',
        'arviz',
        'tqdm',
        'asdf',
        ]
 )
