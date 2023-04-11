# pysersic
Code for fitting sersic profiles to galaxy images using bayesian inference in python.

# Installing pysersic
`pysersic` is built on `jax`, if you are only planning on running it with you CPU, then it can be installed simply using `pip install "jax[cpu]"`. `jax` unfortunately does not work on windows. If you are planning on using a GPU or TPU, you must make sure to install the right CUDA version, see the `jax` install guide for more details: https://github.com/google/jax#installation.

Once `jax` is installed you can move on to installing `pysersic`. We will be uploading it to PyPi soon but in the meantime it can be installed from the github source files using,

``
$ cd < Directory where it will be installed >
$ git clone https://github.com/pysersic/pysersic
$ cd pysersic
$ pip install . -e
``

or

`` pip install git+https://github.com/pysersic/pysersic``