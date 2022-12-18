import os.path

from setuptools import setup

import tfidftransform

with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as f:
    requirements = f.read().splitlines(keepends=False)

setup(
    name='tfidftransform',
    version=tfidftransform.__version__,
    description='A TF-IDF transformation library',
    license='MIT',
    packages=['tfidftransform'],
    package_data={'tfidftransform': ['version.txt']},
    python_requires='>=3.8',
    install_requires=requirements,
    url='https://github.com/SedatDe/tfidftransform'
)
