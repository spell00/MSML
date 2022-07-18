#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 2021
@author: Simon Pelletier
"""


from setuptools import setup


setup(
    name='msml',
    version='0.1',
    packages=['msml', 'msml.dl', 'msml.dl.utils', 'msml.dl.models', 'msml.dl.models.pytorch',
              'msml.dl.train',  "msml.scikit_learn", "msml.scikit_learn.train"],
    url='',
    license='',
    author='simon pelletier',
    author_email='',
    description=''
)
