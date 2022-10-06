from setuptools import setup, find_packages

setup(
    name='diana',
    version='0.1',
    description='A package to train and integerise QNNs with the DIANA chip.',
    author='Abubakr Nada',
    author_email='abubakr_ehab@hotmail.com',
    packages=find_packages(),
    install_requires=[
        'torch',
        'networkx'
    ],
)
