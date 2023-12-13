from setuptools import setup, find_packages

setup(
    name='od_evaluation',
      packages = find_packages(exclude=['config']),
      version='0.1.0',
      author='Lue Fan',
      install_requires=[]
    )