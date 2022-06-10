from setuptools import setup

setup(
    name="extratorch",
    version="1.1",
    packages=["extratorch"],
    url="https://github.com/alexarntzen/extratorch.git",
    license="MIT",
    author="Alexander Johan Arntzen",
    author_email="hello@alexarntzen.com",
    description="Useful extra functions for the PyTorch library.",
    install_requires=[
        "torch>=1.8.1",
        "numpy>=1.18.2",
        "matplotlib>=3.2.0",
        "scikit-learn>=0.24.1",
        "tqdm>=4.63.0",
        "pandas>=1.2.4",
    ],
)
