from setuptools import setup, find_packages

setup(
    name="diana",
    version="0.1",
    description="A package to quantize QNNs for the DIANA chip.",
    packages=find_packages(),
    install_requires=["torch", "networkx"],
)
