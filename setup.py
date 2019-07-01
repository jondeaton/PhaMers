from distutils.core import setup

setup(
    name='PhaMers',
    version='1.0',
    packages=['phamers',],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.txt').read(),
    requires=["numpy", "matplotlib", "scipy", "biopython", "pandas"]
)
