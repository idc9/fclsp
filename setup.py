from setuptools import setup, find_packages

# def readme():
#     with open('README.rst') as f:
#            return f.read()

# time, abc, numbers, copy, textwrap
# os, json, argparase, re

install_requires = ['numpy',
                    'pandas',
                    'scikit-learn'
                    ]


setup(name='fclsp',
      version='0.0.0',
      description='Folded concave Laplacian spectral penalty.',
      author='Iain Carmichael',
      author_email='idc9@cornell.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=install_requires,
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
