from setuptools import setup, find_packages

setup(
    name='multipoleprior',
    version='0.1',
    description='Multipole prior for lens modeling',
    author='Maverick S. H. Oh',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'numba',
        'lenstronomy'
    ],
    package_data={'multipoleprior': ['params_for_prior.json']},
)
