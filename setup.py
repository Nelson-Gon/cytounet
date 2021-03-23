from setuptools import setup, find_packages
from cytounet.version import __version__
zip_link = "https://github.com/Nelson-Gon/cytounet/archive/refs/tags/v"+__version__+".zip"

setup(name='cytounet',
      version=__version__,
      description='Deep Learning based Cell Segmentation',
      url="http://www.github.com/Nelson-Gon/cytounet",
      download_url="https://github.com/Nelson-Gon/cytounet/",
      author='Nelson Gonzabato',
      author_email='gonzabato@hotmail.com',
      license='MIT',
      keywords="keras tensorflow images image-analysis deep-learning biology",
      packages=find_packages(),
      long_description=open('README.md', encoding="UTF-8").read(),
      python_requires='>=3.6',
      long_description_content_type='text/markdown',
      zip_safe=False)
