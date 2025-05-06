import versioneer
from setuptools import setup, find_packages

setup(
    name='laGPy',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='Reygie Macasieb',
    author_email='reygie.macasieb@research.uwa.edu.au',
    license='GNU Lesser General Public License v3 (LGPLv3)',
    description='Python implementation of local approximate GP',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/rqmacasieb/laGPy',
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.6.0",
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)