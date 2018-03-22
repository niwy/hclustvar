# -*- coding:utf-8 -*-
from setuptools import setup

setup(
    name='hclustvar',
    version='0.1.0',
    description=(
        'Package for hierarchical clustering of mixed variables: numeric and/or categorical.'
    ),
    author='niwy',
    author_email='weiyuan.ni@hotmail.com',
    license='MIT',
    packages=['hclustvar'],
    platforms=["all"],
    url='https://github.com/niwy/hclustvar',
    classifiers=[
        'Intended Audience :: Data Scientists',
        'License :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Data Mining :: Variable Analysis'
    ],
    keywords=["hierarchical clustering", "mixed variables"],
    install_requires=[
        'pandas>=0.22.0',
        'numpy>=1.14.0',
        'scipy>=1.0.0',
        'scikit-learn>=0.19.1'
    ]
)
