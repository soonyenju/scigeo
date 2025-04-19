#!/usr/bin/env python
#-*- coding:utf-8 -*-

#############################################
# File Name: setup.py
# Author: Songyan Zhu
# Mail: zhusy93@gmail.com
# Created Time:  2018-10-23 13:28:34
#############################################


from setuptools import setup, find_packages

setup(
	name = "scigeo",
	version = "0.0.15",
	keywords = ("easy geo warpper"),
	description = "read/dump/process geo data",
	long_description = "coming soon",
	license = "MIT Licence",

	url="https://github.com/soonyenju/scigeo",
	author = "Songyan Zhu",
	author_email = "zhusy93@gmail.com",

	packages = find_packages(),
	include_package_data = True,
	platforms = "any",
	install_requires=[
            # "geopandas==0.4.1",
            # "scipy==1.2.1",
            # "GDAL==2.3.3",
            # "pyproj==1.9.6",
            # "Shapely==1.6.4.post1",
            # "rasterio==1.0.21",
            # "numpy==1.16.0",
			"geopandas",
			"rasterio"

	]
)