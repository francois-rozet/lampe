#!/usr/bin/env python

import setuptools

with open('README.md', 'r') as f:
    readme = f.read()

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='lampe',
    version='0.4.1',
    packages=setuptools.find_packages(),
    description='Likelihood-free AMortized Posterior Estimation with PyTorch',
    keywords='parameter inference bayes posterior amortized likelihood ratio mcmc torch',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='François Rozet',
    author_email='francois.rozet@outlook.com',
    license='MIT license',
    url='https://github.com/francois-rozet/lampe',
    project_urls={
        'Documentation': 'https://github.com/francois-rozet/lampe',
        'Source': 'https://github.com/francois-rozet/lampe',
        'Tracker': 'https://github.com/francois-rozet/lampe/issues',
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
    install_requires=required,
    extras_require={
        'docs': [
            'furo',
            'sphinx',
        ]
    },
    python_requires='>=3.8',
)
