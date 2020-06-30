from setuptools import setup, find_packages


setup(name='unet',
      version="0.1.0",
      description='A Keras-Tensorflow Segmentation Pipeline',
      url="http://www.github.com/Nelson-Gon/unet",
      download_url=None,
      author='Nelson Gonzabato',
      author_email='gonzabato@hotmail.com',
      license='MIT',
      keywords="keras tensorflow images image-analysis deep-learning",
      packages=find_packages(),
      long_description=open('README.md').read(),
      python_requires='>=3.6',
      long_description_content_type='text/markdown',
      zip_safe=False)
