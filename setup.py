#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=6.0', 'scipy>=1.0.0', 'numpy>=1.14.2', 'joblib>=0.11', 
'xarray>=0.10.2', 'matplotlib>=2.2.2']

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Aleksi Nummelin",
    author_email='aleksi.h.nummelin@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Operating System :: OS Independent',
    ],
    description="MicroInverse is a Python package for inversion of a transport operator from tracer data.",
    entry_points={
        'console_scripts': [
            'MicroInverse=MicroInverse.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='MicroInverse',
    name='MicroInverse',
    packages=find_packages(include=['MicroInverse']),
    package_data={
      'MicroInverse': ['tests/adv_diff_fipy_solve_1000_vel0.05_r005_dx5000_dt1000.npz']
    },
    setup_requires=setup_requirements,
    test_suite='MicroInverse/tests',
    tests_require=test_requirements,
    url='https://github.com/AleksiNummelin/MicroInverse',
    version='0.3.0',
    zip_safe=False,
)
