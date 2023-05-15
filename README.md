![pysersic logo featuring a spiral s-galaxy as the S](misc/pysersic.png)
# pysersic
Code for fitting sersic profiles to galaxy images using bayesian inference in python accelerated using `jax`. 

The code is still under activate development, so use with caution. The core functionality is mostly in place with more complete documentation and tutorials to come soon. For now there are a few example notebooks in the `examples/` folder. Still if you try the code out, we would welcome any feedback! The easiest way would be to reach out to us (Imad and Tim) directly or open a new issue on github.

# Installing pysersic
`pysersic` is built on `jax`, if you are only planning on running it with you CPU, then it can be installed simply using `pip install "jax[cpu]"`. `jax` unfortunately does not work on windows. If you are planning on using a GPU or TPU, you must make sure to install the right CUDA version. Please see the `jax` [install guide](https://github.com/google/jax#installation) for more details.

Once `jax` is installed you can move on to installing `pysersic`. We will be uploading it to PyPi soon but in the meantime it can be installed from the github source files locally,

```
$ cd < Directory where it will be installed >
$ git clone https://github.com/pysersic/pysersic
$ cd pysersic
$ pip install . -e
```

or by using pip to install directly from the github

` pip install git+https://github.com/pysersic/pysersic`
