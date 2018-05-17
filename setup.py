from setuptools import setup

setup(name='slir',
      version='0.5.1',
      description='Python package for Sparse Linear Regression',
      author='Misato Tanaka',
      author_email='kamitanilab.contact@gmail.com',
      url='https://github.com/KamitaniLab/slir',
      license='MIT',
      packages=['slir'],
      install_requires=['numpy', 'scikit-learn'])
