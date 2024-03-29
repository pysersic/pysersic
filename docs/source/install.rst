Installation
============

pysersic is built on `jax <https://github.com/google/jax>`_, which is fairly simple to install now but can be an issue in some use cases. If you are planning on running only with your CPU, then it can be installed simply using 

.. code-block:: bash

    $ pip install "jax[cpu]"

Unfortunately jax does not work on windows, so they reccommend utilizing the windows subsystem for linux (WSL) instead. If you are planning on using a GPU or TPU, you must make sure to install the right CUDA version. For more info please refer to the jax `install guide <https://github.com/google/jax#installation>`_. 

Once jax is installed you can move on to installing pysersic. If you would like to create an environment within which to install ``pysersic``, we have provided two files to this end: `environment.yml <https://github.com/pysersic/pysersic/blob/main/environment.yml>`_, which can be used to create a ``conda`` environment with the needed dependencies, and `environment-lock.yml <https://github.com/pysersic/pysersic/blob/main/environment-lock.yml>`_, which has a single frozen set of dependency versions that are known to work. That file may be helpful if you find you are having installation issues.

You can install the latest standard release of pysersic from pip: 

.. code-block:: bash

    $ pip install pysersic


Or, if you want the latest development build or the ability to edit the `pysersic` source code:

.. code-block:: bash

    $ cd < Directory where it will be installed >
    $ git clone https://github.com/pysersic/pysersic
    $ cd pysersic
    $ pip install . -e


or by using pip to install directly from the github

.. code-block:: bash
    
    pip install git+https://github.com/pysersic/pysersic

