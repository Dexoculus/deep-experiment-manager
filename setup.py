from setuptools import setup, find_packages

setup(
    name='ExperimentManager',
    version='0.1.0',
    description='A module for training, testing, and managing experiments in PyTorch.',
    author='Dexoculus',
    author_email='hyeonbin@hanyang.ac.kr',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'PyYAML',
        'matplotlib',

    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
