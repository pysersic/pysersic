Installation
============

pysersic is built on `jax <https://github.com/google/jax>`_, which is fairly simple to install now but can be an issue in some use cases. If you are planning on running only with your CPU, then it can be installed simply using 

.. code-block:: bash

    $ pip install "jax[cpu]"

Unfortunately jax does not work on windows, so they reccomend utilizing the windows subsystem for linux (WSL) instead. If you are planning on using a GPU or TPU, you must make sure to install the right CUDA version. For more info please refer to the jax `install guide <https://github.com/google/jax#installation>`_. 

Once jax is installed you can move on to installing pysersic. We will be uploading it to PyPi soon but in the meantime it can be installed from the github source files locally,

.. code-block:: bash

    $ cd < Directory where it will be installed >
    $ git clone https://github.com/pysersic/pysersic
    $ cd pysersic
    $ pip install . -e
```

or by using pip to install directly from the github

.. code-block:: bash
    
    pip install git+https://github.com/pysersic/pysersic
