

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3928919.svg
   :target: https://doi.org/10.5281/zenodo.3928919
   :alt: DOI


.. image:: https://www.repostatus.org/badges/latest/wip.svg
   :target: https://www.repostatus.org/badges/latest/wip.svg
   :alt: Stage
 

.. image:: https://github.com/Nelson-Gon/cytounet/workflows/Test%20Install/badge.svg
   :target: https://github.com/Nelson-Gon/cytounet/workflows/Test%20Install/badge.svg
   :alt: Test Install


.. image:: https://badge.fury.io/py/cytounet.svg
   :target: https://badge.fury.io/py/cytounet
   :alt: PyPI version
 

.. image:: https://img.shields.io/pypi/l/cytounet.svg
   :target: https://pypi.python.org/pypi/cytounet/
   :alt: PyPI license
 

.. image:: https://travis-ci.com/Nelson-Gon/cytounet.svg?branch=master
   :target: https://travis-ci.com/Nelson-Gon/cytounet.svg?branch=master
   :alt: Travis Build


.. image:: https://img.shields.io/badge/Maintained%3F-yes-green.svg
   :target: https://GitHub.com/Nelson-Gon/cytounet/graphs/commit-activity
   :alt: Maintenance


.. image:: https://img.shields.io/github/last-commit/Nelson-Gon/cytounet.svg
   :target: https://github.com/Nelson-Gon/cytounet/commits/master
   :alt: GitHub last commit


.. image:: https://img.shields.io/github/issues/Nelson-Gon/cytounet.svg
   :target: https://GitHub.com/Nelson-Gon/cytounet/issues/
   :alt: GitHub issues


.. image:: https://img.shields.io/github/issues-closed/Nelson-Gon/cytounet.svg
   :target: https://GitHub.com/Nelson-Gon/cytounet/issues?q=is%3Aissue+is%3Aclosed
   :alt: GitHub issues-closed


----

**Installation**

From ``PyPI``\ :

.. code-block::


   pip install cytounet

From source or to use without installing locally:

.. code-block::

   git clone https://github.com/Nelson-Gon/cytounet.git
   # proceed with usual source build procedure

Or:

.. code-block:: python


   pip install git+https://github.com/Nelson-Gon/cytounet.git

**Import**

.. code-block:: python


   from cytounet.model import *
   from cytounet.data import *
   from cytounet.augmentation import *

**Detailed Sample Usage**

Please see the following examples:


* `Predicting Embryonic DIC Image Labels with Keras <https://www.kaggle.com/gonnel/predicting-embryonic-dic-image-labels-with-keras>`_


.. image:: https://img.shields.io/badge/view%20on-nbviewer-brightgreen.svg
   :target: https://nbviewer.jupyter.org/github/Nelson-Gon/cytounet/blob/aedf8d52af4e3e9f2cd426de90b4c5dea2a4e11c/examples/embryos_dic.ipynb
   :alt: nbviewer



* Chinese Hamster Ovary Segmentation `here <https://github.com/Nelson-Gon/cytounet/blob/ff5ce0c2cc97e35baf1edacbc994661583200884/examples/example_usage.ipynb>`_


.. image:: https://img.shields.io/badge/view%20on-nbviewer-brightgreen.svg
   :target: https://nbviewer.jupyter.org/github.com/Nelson-Gon/cytounet/blob/ff5ce0c2cc97e35baf1edacbc994661583200884/examples/example_usage.ipynb
   :alt: nbviewer


Visually:


.. image:: https://raw.githubusercontent.com/Nelson-Gon/cytounet/master/examples/example_results.png
   :target: https://raw.githubusercontent.com/Nelson-Gon/cytounet/master/examples/example_results.png
   :alt: CHO


For more examples or to add your own, please see the examples `folder <https://github.com/Nelson-Gon/cytounet/blob/master/examples>`_.

**Is it supported?**

A checked box indicates support. You can either add more feature requests here or tackle unchecked boxes and make
a pull request to add such support. 


* 
  [ ] Custom Number of Layers

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

**Frequently Asked Questions**

Please read our Wiki `Pages <https://github.com/Nelson-Gon/cytounet/wiki>`_

**To raise an issue or question**

If the `wiki <https://github.com/Nelson-Gon/cytounet/wiki>`_ does not answer your question,
please raise a new issue `here <https://github.com/Nelson-Gon/cytounet/issues>`_. You can also open an issue if you have any discussion, criticism,
or bug reports. 

Thank you very much. 

----

**References**


* 
  Mouse Embryos `Dataset obtained <https://github.com/Nelson-Gon/cytounet/tree/master/examples/BBBC003_v1>`_ from Broad Bioimage Benchmark Collection.
  `Source <https://data.broadinstitute.org/bbbc/BBBC003/>`_.

* 
  Red Blood Cell `Images <https://github.com/Nelson-Gon/cytounet/tree/master/examples/BBBC009_v1>`_ provided by Anne 
  Carpenter and Roger Wiegand, available `here <https://data.broadinstitute.org/bbbc/BBBC009/>`_

* 
  Chinese Hamster Ovary `Cells <https://github.com/Nelson-Gon/cytounet/tree/master/examples/BBBC030_v1>`_ provided by 
  Koos et al.(2016), available `here <https://bbbc.broadinstitute.org/BBBC030>`_

**Credits**


* This repository started out as a clone of `zhixuhao <https://github.com/zhixuhao>`_\ 's  original 
  unet `implementation <https://github.com/zhixuhao/unet/>`_.

This repository has considerably diverged from the original implementation hence the need
to distribute it separately. 

This decision was taken in part due to the relative inactivity of the original implementation which would have made
it harder to collaborate. Please take a look at the list of changes from the original implementation
`here <https://github.com/Nelson-Gon/cytounet/blob/master/changelog.md>`_. 


* The Unet algorithm was introduced by Ronneberger et al. in their `paper <http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/>`_.
