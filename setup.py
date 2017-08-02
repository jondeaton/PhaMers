#!/usr/bin/env python2.7

from distutils.core import setup

setup(
    name='PhaMers',
    version='1.0',
    description='Bioinformatic tool to bacteriophage identification',
    author='Jon Deaton',
    author_email='jdeaton@stanford.edu',
    url='github.com/jondeaton/PhaMers',
    packages=['phamers'],
    install_requires=["numpy", "pandas", "biopython", "scipy", "matplotlib", "sklearn", "logging", "dna_features_viewer"],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.txt').read(),
    install_requires=[
    "numpy",
    "pandas",
    "skit-learn",
    "logging",
    "dna_features_viewer",
    "biopython",
    ]

)