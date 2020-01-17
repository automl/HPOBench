# -*- encoding: utf-8 -*-
import setuptools

with open('requirements.txt', encoding='utf-8') as fh:
    requirements = [line.strip() for line in fh.readlines()]

with open("hpolib/__version__.py", encoding='utf-8') as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

setuptools.setup(
    name='hpolib3',
    description='Automated machine learning.',
    version=version,
    packages=setuptools.find_packages(exclude=['test']),
    install_requires=requirements,
    extras_require={
        'xgboost': ['xgboost==0.90'],
        'singularity': ['Pyro4==4.77'],
    },
    author_email='eggenspk@informatik.uni-freiburg.de',
    license='Apache-2.0',
    platforms=['Linux'],
    classifiers=[
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ]
)
