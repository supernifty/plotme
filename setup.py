#!/usr/bin/env python

import setuptools
import pathlib
from glob import glob

name = "plotme"
version = "0.1"
release = "0.1.0"
here = pathlib.Path(__file__).parent.resolve()

setuptools.setup(
    name=name,
    version=version,
    packages=setuptools.find_packages(),
    scripts = [i for i in glob("plotme/*.py") if all(pattern not in i for pattern in ["__init__", "settings"])],
)

