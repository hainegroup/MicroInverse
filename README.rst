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

.. math:: 
   \frac{d}{dt} \mathbf{x}(t) = \mathbf{B}\mathbf{x}(t) + \mathbf{f}(t)

Where :math:`\mathbf{x}(t)` is the vector of tracer anomaly timeseries,  :math:`\mathbf{B}` is the transport operator, and 
:math:`\mathbf{f}(t)` is the forcing of the system. Assyming that the forcing has a shorter decorrelation timescale than
the tracer we can solve for the transport operator:

.. math:: 
   
   \mathbf{B}=\frac{1}{\tau}\log \left(\left[ \mathbf{x}(t+\tau)\mathbf{x}^T(t)\right ] \left[ \mathbf{x}(t)\mathbf{x}^T(t) \right]^{-1}\right).

Where :math:`\tau` is the chosen decorrelation timescale which should be larger than the forcing decorrelation timescale, 
but smaller than the decorrelation timescale of the tracer. 

In practice :math:`\tau` is hard to choose a priori which is why we suggest first inverting your data at multiple values 
of :math:`\tau` and combining the results afterwards using MicroInverse.combine_Taus().

MicroInverse will also relate :math:`\mathbf{B}` to velocity, diffusivity, and decay via advection-diffusion-relaxation equation (see `Nummelin et al. (2018)`__ for details)

* Free software: MIT license
* Documentation: https://MicroInverse.readthedocs.io.


Features
--------

* TODO

Credits
-------

This package is based on work by `Nummelin et al. (2018)`__ and Jeffress and Haine (2014a_, 2014b_)

.. _Nummelin: http://pages.jh.edu/~anummel1/
__ Nummelin_
.. _2014a: https://doi.org/10.1002/qj.2313
.. _2014b: https://doi.org/10.1088/1367-2630/16/10/105001 

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
