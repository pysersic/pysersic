import setuptools

with open('README.md') as f:
    long_desc = f.read()

setuptools.setup(
    name="pysersic",
    version="0.1.1",
    author="Imad Pasha & Tim Miller",
    author_email="imad.pasha@yale.edu",
    description="A Tool for fitting sersic profiles in python in a Bayesian context, accelerated with JAX",
    long_description=long_desc,
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
        'sep']
 )
