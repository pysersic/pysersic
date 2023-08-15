Issues and Contributing
=======================

If you happen to find a bug, have a question or want to request a feature the best way to get in touch is to open an `issue on GitHub <https://github.com/pysersic/pysersic/issues>`_.

We are happy to have anyone contribute to improving pysersic. For smaller bug fixes etc. feel free to open a pull request. For larger changes or new features it is best to first open an issue to discuss the proposed changes and implementation details. 

For anyone who is new to contributing to projects on github, a good place to start is the `astropy guide <https://docs.astropy.org/en/stable/development/workflow/development_workflow.html>`_ for a nice primer on all the thing you will need to know!

If you have made any changes to the code it is a good idea to run the unit tests to make sure nothing is broken. This can be done with `pytest <https://github.com/pytest-dev/pytest>`_ by running the following command from the root directory of the repository,


.. code-block:: bash
    
    $ pytest .

These tests will also automatically be run using GitHub Actions on any pull request.