# Copyright (C) 2021 Mina Jamshidi
# <minajamshidi91@gmail.com>

from setuptools import setup, find_packages

with open('README.md', "r") as fh:
    long_description = fh.read()

setup(
    author='Mina Jamshidi',
    author_email='minajamshidi91@gmail.com',
    name='mypyutils',
    version='0.0.1',
    description='My custom-written Python classes and functions that make life easier!',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved",
        "Operating System :: OS Independent",
    ],
)
