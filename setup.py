from setuptools import setup

setup(
    name="fastuot",
    distname="",
    version='0.1.0',
    description="Fast computation of unbalanced OT problems",
    author='Thibault Sejourne',
    author_email='thibault.sejourne@ens.fr',
    url='https://github.com/thibsej/fast_uot',
    packages=['fastuot'],
    install_requires=[
              'numpy',
              'torch',
              'scipy',
              'numba',
              'matplotlib',
              'tqdm'
          ],
    license="MIT",
)
