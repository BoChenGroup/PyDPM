#!/usr/bin/env python
# coding=utf-8

from setuptools import setup, find_packages


setup(
    name='pydpm',
    version='2.0.0',
    description='A python library focuses on constructing deep probabilistic models on GPU.',
    py_modules=['pydpm'],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Chaojie Wang, Wei Zhao, Jiawen Wu',
    author_email='xd_silly@163.com',
    maintainer='BoChenGroup',
    maintainer_email='13279389260@163.com',
    license='Apache License Version 2.0',
    packages=find_packages(),
    # package_data={'pydpm': c_package_data},
    # data_files=c_package_data,
    include_package_data=True,  # include all files
    platforms=["Windows", "Linux"],
    url='https://github.com/BoChenGroup/Pydpm',
    requires=['numpy', 'scipy', 'sklearn', 'pycuda'],
    classifiers=[
        'Environment :: GPU :: NVIDIA CUDA',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Topic :: Software Development :: Libraries'
    ],
)
