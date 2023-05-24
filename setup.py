from setuptools import setup, find_packages

setup(
    name='swass',
    version='1.0',
    author='Mark Stent',
    author_email='mark@markstent.co.za',
    description='SWASS package for audio similarity calculation',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'librosa',
        'pypesq',
    ],
)
