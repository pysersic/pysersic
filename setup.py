import setuptools

setuptools.setup(
    name="pysersic",
    version="0.1",
    author="Imad Pasha & Tim Miller",
    author_email="imad.pasha@yale.edu",
    description="A Tool for fitting sersic profiles in python",
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
