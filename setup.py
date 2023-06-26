import setuptools


def find_packages():
    def recurse(package_name):
        return [package_name] + [package_name + '.' + p for p in setuptools.find_packages(package_name)]

    return recurse('dianaquantlib') + recurse('quantlib')


setuptools.setup(
    name="dianaquantlib",
    version="0.1",
    description="A package to quantize QNNs for the DIANA chip.",
    packages=find_packages(),
    install_requires=[
        'networkx==2.8.4',
        'torch==1.12',
        'torchvision',
        'onnx==1.12.0',
        'onnxruntime==1.15.0',
        'numpy',
        'tqdm',
        'pyyaml',
        'scikit-learn', # only for running examples
    ]
)
