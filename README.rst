

.. image:: https://www.repostatus.org/badges/latest/wip.svg
   :target: https://www.repostatus.org/badges/latest/wip.svg
   :alt: Stage
 

.. image:: https://github.com/Nelson-Gon/unet/workflows/Test%20Install/badge.svg
   :target: https://github.com/Nelson-Gon/unet/workflows/Test%20Install/badge.svg
   :alt: Test Install


.. image:: https://travis-ci.com/Nelson-Gon/unet.svg?branch=master
   :target: https://travis-ci.com/Nelson-Gon/unet.svg?branch=master
   :alt: Travis Build


.. image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg
   :target: https://GitHub.com/Nelson-Gon/unet/graphs/commit-activity
   :alt: Maintenance


.. image:: https://img.shields.io/github/last-commit/Nelson-Gon/unet.svg
   :target: https://github.com/Nelson-Gon/unet/commits/master
   :alt: GitHub last commit


.. image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
   :target: https://www.python.org/
   :alt: made-with-python


.. image:: https://img.shields.io/github/issues/Nelson-Gon/unet.svg
   :target: https://GitHub.com/Nelson-Gon/unet/issues/
   :alt: GitHub issues


.. image:: https://img.shields.io/github/issues-closed/Nelson-Gon/unet.svg
   :target: https://GitHub.com/Nelson-Gon/unet/issues?q=is%3Aissue+is%3Aclosed
   :alt: GitHub issues-closed


.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://github.com/Nelson-Gon/pyautocv/blob/master/LICENSE
   :alt: license


----

**Installation**

To install:

.. code-block::

   git clone https://github.com/Nelson-Gon/unet.git

Or:

.. code-block:: python


   pip install git+https://github.com/Nelson-Gon/unet.git

**Sample Usage**

Please see the following examples:


* 
  Typical Pipeline available `here <https://github.com/Nelson-Gon/unet/blob/master/examples/example_usage.ipynb>`_

* 
  A biological example showing segmentation of DIC images of embryos available `here <https://github.com/Nelson-Gon/unet/blob/master/examples/embryos.ipynb>`_

For more examples or to add your own, please see the 
examples `folder <https://github.com/Nelson-Gon/unet/blob/master/examples>`_.

**Is it supported?**

A checked box indicates support. You can either add more feature requests here or tackle unchecked boxes and make
a pull request to add such support. 


* 
  [x] Single class segmentation

* 
  [x] Grayscale images

* 
  [x] Model Validation

* 
  [ ] Multi-class segmentation

* 
  [ ]  Colored image input

* 
  [ ] COCO Datasets 

* 
  [ ] CSV Based Annotations

* 
  [ ] XML Based Annotations 

**To raise an issue or question**

Please raise an issue `here <https://github.com/Nelson-Gon/unet/issues>`_ if you have any discussion, criticism,

or bug reports. 

Thank you very much. 

----

**References**


* 
  Mouse Embryos `Dataset obtained <https://github.com/Nelson-Gon/unet/tree/master/examples/BBBC003_v1>`_ from Broad Bioimage Benchmark Collection.
  `Source <https://data.broadinstitute.org/bbbc/BBBC003/>`_.

* 
  Red Blood Cell `Images <https://github.com/Nelson-Gon/unet/tree/master/examples/BBBC009_v1>`_ provided by Anne 
  Carpenter and Roger Wiegand, available `here <https://data.broadinstitute.org/bbbc/BBBC009/>`_

**Credits**


* This repository started out as a clone of `zhuxihao <https://github.com/zhixuhao>`_\ 's  original 
  unet `implementation <https://github.com/zhixuhao/unet/>`_.

This repository has considerably diverged from the original implementation hence the need
to distribute it separately. 

This decision was taken in part due to the relative inactivity of the original implementation which would have made
it harder to collaborate. Please take a look at the list of changes from the original implementation
`here <https://github.com/Nelson-Gon/unet/blob/master/changelog.md>`_. 


* The Unet algorithm was introduced by Ronneberger et al. in their `paper <http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/>`_.
