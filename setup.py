from setuptools import setup
import re, os, sys

setup(
    name="lightning-tutorial",
    version="0.0.1",
    python_requires=">3.7.0",
    author="Michael E. Vinyard - Harvard University - Massachussetts General Hospital - Broad Institute of MIT and Harvard",
    author_email="mvinyard@broadinstitute.org",
    url="https://github.com/mvinyard/vinplots",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    description="pytorch-lightning tutorial",
    packages=[
        "lightning_tutorial",
    ],
    install_requires=[
        "torch>=1.12.1",
        "pytorch-lightning>=1.7.4",
        "scanpy>=1.9.1",
        "nb_black>=0.7",
        "torch-adata>=0.0.12",
        "vinplots>=0.0.72",
        "torch-composer>=0.0.3",
        "larry-dataset>=0.0.1",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    license="MIT",
)
