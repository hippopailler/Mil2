from setuptools import setup, find_packages

setup(
    name="MIL",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch',
        'pandas',
        'numpy',
        'rich'
    ]
)