from setuptools import setup, find_packages

setup(name='oof',
      version='0.1.0',
      author='Fernando Perez-Garcia',
      author_email='fernando.perezgarcia.17@ucl.ac.uk',
      packages=find_packages(exclude=['*tests']),
      entry_points={'console_scripts': [
          'convert_trsf = oof.oof:main',
          ]},
      install_requires=[
          'nibabel',
          'scipy',
          ],
     )
