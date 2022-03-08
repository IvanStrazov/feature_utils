# utf-8
# Python 3.10.0
# 2022-03-08


import setuptools
import feature_utils


setuptools.setup(
    name = "feature_utils",
    version = feature_utils.__version__,
    description = "make some feature's transformations",
    long_description = open("README.md").read(),
    author = "Ivan Strazov",
    author_email = "ivanstrazov@gmail.com",
    url = "https://github.com/IvanStrazov/feature_utils/",
    keywords = "ml features",

    packages = setuptools.find_packages(exclude=("config"))
)
