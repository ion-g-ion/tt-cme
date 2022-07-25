from setuptools import setup
setup(name='TT-CME',
version='1.0',
description='Tensor-Train decomposition for the Chemical Master Equation',
url='https://github.com/ion-g-ion/torchTT',
author='Ion Gabriel Ion',
author_email='ion.ion.gabriel@gmail.com',
license='MIT',
packages=['TTCME'],
install_requires=['numpy>=1.18','torch>=1.7','opt_einsum','scipy','numba','torchtt'],
test_suite='tests',
zip_safe=False) 