import os
import sys

from setuptools import setup
from setuptools.command.test import test

# Get the long description from the README file
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


class PyTest(test):
    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        errno = pytest.main(['test'])
        sys.exit(errno)


setup(
    name='slurf',
    version='0.1',
    author='T. Badings, S. Junges, M. Volk',
    author_email='thom.badings@ru.nl',
    description='SLURF - Scenario optimization for Learning Uncertain Reliability Functions',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['slurf'],
    cmdclass={
        'test': PyTest
    },
    zip_safe=False,
    install_requires=[
        'stormpy>=1.6.3',
        'numpy',
        'cvxpy',
        'matplotlib',
        'seaborn',
        'tqdm' # Progress bar
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    python_requires='>=3',
)
