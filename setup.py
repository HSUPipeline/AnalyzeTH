"""SpikeTools setup script."""

import os
from setuptools import setup, find_packages

# Get the current version number from inside the module
with open(os.path.join('code', 'version.py')) as version_file:
    exec(version_file.read())

# Load the long description from the README
with open('README.md') as readme_file:
    long_description = readme_file.read()

# Load the required dependencies from the requirements file
with open("requirements.txt") as requirements_file:
    install_requires = requirements_file.read().splitlines()

setup(
    name = 'analyzeth',
    version = __version__,
    description = 'Module for analyzing (single neuron spiking for now) data from the Treasure Hunt memory task.',
    long_description = long_description,
    python_requires = '>=3.6',
    maintainer = ['Thomas Donoghue', 'Cameron Holman', 'Zhixian (Claire) Han'],
    maintainer_email = ['tdonoghue.research@gmail.com', 'cameron.holman@columbia.edu', 'zh2497@columbia.edu'],
    url = 'https://github.com/JacobsSU/AnalyzeTH',
    packages = find_packages(),
    license = 'Apache License, 2.0',
    classifiers = [
        'Development Status :: 3 - Alpha'
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
    platforms = 'any',
    project_urls = {
        'Documentation' : 'WIP',
        'Bug Reports' : 'WIP',
        'Source' : 'WIP'
    },
    download_url = 'https://github.com/JacobsSU/AnalyzeTH',
    keywords = ['neuroscience', 'single units', 'spike analyses', 'electrophysiology', 
                'Treasure Hunt', 'memory', 'spatial navigation'],
    install_requires = install_requires,
    tests_require = ['pytest'],
)
