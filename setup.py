#!/usr/bin/env python


from setuptools import setup, find_packages

NAME = 'extreme_events'

INSTALL_REQUIRES = [
    # Core dependencies
    "matplotlib ==3.1.2",
    "scipy ==1.4.1",
    "scikit-learn ==0.20.3",
    "tensorflow ==2.1.0",
    "pandas ==1.0.1",
    "numpy ==1.16.2",
]

# add development requires here
DEV_REQUIRES = [
    "pytest",
    "pytest-html",
    "pytest-xdist",
    "pytest-forked",
    "pytest-cov",
    "pylint",
    "pycodestyle",
    "setuptools",
]

setup(
    name=NAME,
    version='0.1.0',
    author="Ghayoor & Farazmad",
    url="https://github.com/hossen-code/extreme_events",
    author_email='hghayoor@gmail.com',
    packages=find_packages(),
    description="Extreme event prediction",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'dev': DEV_REQUIRES
    },
    include_package_data=True,
)
