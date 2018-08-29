============
MicroInverse
============


.. image:: https://img.shields.io/pypi/v/MicroInverse.svg
        :target: https://pypi.python.org/pypi/MicroInverse

.. image:: https://img.shields.io/travis/AleksiNummelin/MicroInverse.svg
        :target: https://travis-ci.org/AleksiNummelin/MicroInverse

.. image:: https://readthedocs.org/projects/MicroInverse/badge/?version=latest
        :target: https://MicroInverse.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




MicroInverse is a Python package for inversion of a transport operator from tracer data.

It is based on the simple stochastic climate model approximation

.. image:: http://latex.codecogs.com/gif.latex?%5Cfrac%7Bd%7D%7Bdt%7D%20%5Cmathbf%7Bx%7D%28t%29%20%3D%20%5Cmathbf%7BB%7D%5Cmathbf%7Bx%7D%28t%29%20&plus;%20%5Cmathbf%7Bf%7D%28t%29

Where **x** is the vector of tracer anomaly timeseries, **B** is the transport operator, and **f** is 
the forcing of the system. Assuming that the forcing has a shorter decorrelation timescale than
the tracer we can solve for the transport operator:

.. image:: http://latex.codecogs.com/gif.latex?%5Cmathbf%7BB%7D%3D%5Cfrac%7B1%7D%7B%5Ctau%7D%5Clog%20%5Cleft%28%5Cleft%5B%20%5Cmathbf%7Bx%7D%28t&plus;%5Ctau%29%5Cmathbf%7Bx%7D%5ET%28t%29%5Cright%20%5D%20%5C%20%5Cleft%5B%5Cmathbf%7Bx%7D%28t%29%5Cmathbf%7Bx%7D%5ET%28t%29%20%5Cright%5D%5E%7B-1%7D%5Cright%29

Where tau is the chosen decorrelation timescale which should be larger than the forcing decorrelation timescale, 
but smaller than the decorrelation timescale of the tracer. 

In practice tau is hard to choose a priori which is why we suggest first inverting your data at multiple values 
of tau and combining the results afterwards using MicroInverse.MicroInverse_utils.combine_Taus().

MicroInverse will also relate **B** to velocity, diffusivity, and decay via advection-diffusion-relaxation equation (see `Nummelin et al. (2018)`_ for details)

* Free software: MIT license
* Documentation: https://MicroInverse.readthedocs.io.


Features
--------

* Estimates of velocity, diffusivity, and decay timescale from a timeseries of 2D tracer.

Credits
-------

This package is based on work by `Nummelin et al. (2018)`_ and Jeffress and Haine (2014a_, 2014b_)

.. _Nummelin et al. (2018): http://pages.jh.edu/~anummel1/
.. _2014a: https://doi.org/10.1002/qj.2313
.. _2014b: https://doi.org/10.1088/1367-2630/16/10/105001 

Package is created with Cookiecutter_ using the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
