.. role:: raw-html-m2r(raw)
   :format: html


cytounet: Deep Learning Based Cell Segmentation
===============================================


.. image:: https://badge.fury.io/py/cytounet.svg
   :target: https://badge.fury.io/py/cytounet
   :alt: PyPI version
 

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3928919.svg
   :target: https://doi.org/10.5281/zenodo.3928919
   :alt: DOI


.. image:: https://www.repostatus.org/badges/latest/active.svg
   :target: https://www.repostatus.org/badges/latest/active.svg
   :alt: Stage


.. image:: https://github.com/Nelson-Gon/cytounet/workflows/Test%20Install/badge.svg
   :target: https://github.com/Nelson-Gon/cytounet/workflows/Test%20Install/badge.svg
   :alt: Test Install


.. image:: https://img.shields.io/pypi/l/cytounet.svg
   :target: https://pypi.python.org/pypi/cytounet/
   :alt: PyPI license
 

.. image:: https://readthedocs.org/projects/cytounet/badge/?version=latest
   :target: https://cytounet.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


.. image:: https://pepy.tech/badge/cytounet
   :target: https://pepy.tech/project/cytounet
   :alt: Total Downloads


.. image:: https://pepy.tech/badge/cytounet/month
   :target: https://pepy.tech/project/cytounet
   :alt: Monthly Downloads


.. image:: https://pepy.tech/badge/cytounet/week
   :target: https://pepy.tech/project/cytounet
   :alt: Weekly Downloads


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


**Background**


.. image:: https://github.com/Nelson-Gon/cytounet/blob/master/examples/project_workflow.png?raw=true
   :target: https://github.com/Nelson-Gon/cytounet/blob/master/examples/project_workflow.png?raw=true
   :alt: Project Workflow


This project was largely done as a summer 2020 intern in Dr. Mikael `Bjorklund <https://person.zju.edu.cn/en/H118035>`_\ 's\ :raw-html-m2r:`<br>`
lab, whose guidance, resources, and time I am grateful for. 

The aim was to automate `a549 <https://en.wikipedia.org/wiki/A549_cell>`_ and `rpe <https://en.wikipedia.org/wiki/Retinal_pigment_epithelium>`_ 
cancer cell segmentation and size determination. 


.. image:: https://github.com/Nelson-Gon/cytounet/blob/master/examples/rpe_sample.png?raw=true
   :target: https://github.com/Nelson-Gon/cytounet/blob/master/examples/rpe_sample.png?raw=true
   :alt: RPE Sample


Sample data(10 random images each of train, validate, test sets) is provided in `original_data <https://github.com/Nelson-Gon/cytounet/tree/master/examples/original_data/a549>`_.

A complete a549 cancer cell `segmentation notebook <https://github.com/Nelson-Gon/cytounet/blob/20435549e6b4c3d15979c2117445c4c19ab51bdf/examples/a549_sampler.ipynb>`_ 
is also provided. 

Finally, pre-trained `weights <https://github.com/Nelson-Gon/cytounet/blob/56694553e5014e3f479807de244f5ddeabbcbf80/models/a549_scratch.hdf5>`_ 
are provided that can be used for transfer learning. 
These were trained on considerably more data and for more epochs. For more pre-trained weights and/or data, 
please `contact <https://nelson-gon.github.io/contact>`_ the author. 

**Note**


* 
  To generate masks(labels) provided here ``a549`` cancer cells were stained and imaged with fluorescence microscopy. These 
  fluorescent images were then thresholded with ``threshold_images`` with a threshold value of 83. The images were then saved 
  with ``save_images``. The original fluorescent images are not provided here mainly due to the already huge size of the 
  project. 

* 
  This project is not limited to cancer cells. The model can be trained on almost any object, living and non-living.
  More examples are given below. 

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

.. code-block::


   pip install git+https://github.com/Nelson-Gon/cytounet.git

**Import**

.. code-block::


   from cytounet.model import *
   from cytounet.data import *
   from cytounet.augmentation import *
   from cytounet.post_model import *

**Detailed Sample Usage**

**Script mode**

.. code-block:: shell

   python -m cytounet -t "examples/original_data/a549/train" -i "images" -m "masks" -v 
   "examples/original_data/a549/validation" -l "1e-8" -s 512 -ep 5 -se 250 -b 8 -tt "examples/original_data/a549/test/" 
   -w "models/a549_test/test_model" -o "Adam" -mt "dice_coef" -ls "dice_coef_loss" -sd 2 -f 0 -p 0

To get help:

.. code-block:: shell

   python -m cytounet -h

**Notebooks** 

Please see the following examples:


* `Typical Usage <https://github.com/Nelson-Gon/cytounet/blob/7fd42a27be1b5730eb05e60cb98d5b7e825a0087/examples/example_usage.ipynb>`_


.. image:: https://img.shields.io/badge/view%20on-nbviewer-brightgreen.svg
   :target: https://nbviewer.jupyter.org/github/Nelson-Gon/cytounet/blob/7fd42a27be1b5730eb05e60cb98d5b7e825a0087/examples/example_usage.ipynb
   :alt: nbviewer



* `Predicting Embryonic DIC Image Labels with Keras <https://www.kaggle.com/gonnel/predicting-embryonic-dic-image-labels-with-keras>`_


.. image:: https://img.shields.io/badge/view%20on-nbviewer-brightgreen.svg
   :target: https://nbviewer.jupyter.org/github/Nelson-Gon/cytounet/blob/aedf8d52af4e3e9f2cd426de90b4c5dea2a4e11c/examples/embryos_dic.ipynb
   :alt: nbviewer


Visually:


.. image:: https://raw.githubusercontent.com/Nelson-Gon/cytounet/master/examples/example_results.png
   :target: https://raw.githubusercontent.com/Nelson-Gon/cytounet/master/examples/example_results.png
   :alt: CHO


Sample Object Area Visualization(see the typical usage notebook above for detailed usage)


.. image:: https://raw.githubusercontent.com/Nelson-Gon/cytounet/master/examples/areas.png
   :target: https://raw.githubusercontent.com/Nelson-Gon/cytounet/master/examples/areas.png
   :alt: Area Determination


For more examples or to add your own, please see the examples `folder <https://github.com/Nelson-Gon/cytounet/blob/master/examples>`_.

**Experiments/Benchmarks**

This section shows some experimental results based on publicly available data. 


* Comparison of low vs high quality masks on the model's output

This notebook shows the effects of "filled holes"(outlines whose area is filled with some colour e.g. white)
on the model's quality. The results in general show that filled masks which are also better seen by the human eye
result in better quality output. 

The `notebook <https://github.com/Nelson-Gon/cytounet/blob/9781a45260bd8cdb82b37e07a26254ecf01af5c7/examples/example_usage.ipynb>`_ can be accessed via this `link <https://nbviewer.jupyter.org/github/Nelson-Gon/cytounet/blob/9781a45260bd8cdb82b37e07a26254ecf01af5c7/examples/example_usage.ipynb>`_.

**TODO List**

A checked box indicates support. You can either add more feature requests here or tackle unchecked boxes and make
a pull request to add such support. 


* 
  [x] Single class segmentation

* 
  [x] Grayscale images

* 
  [x] Model Validation

* 
  [x] Determining areas of objects in an image

* 
  [ ] Custom Number of Layers

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
please raise a new `issue <https://github.com/Nelson-Gon/cytounet/issues>`_. 
You can also open an issue if you have any discussion, criticism, or bug reports. 

Thank you very much. 

----

**References**


* 
  Mouse Embryos `Dataset <https://github.com/Nelson-Gon/cytounet/tree/master/examples/BBBC003_v1>`_
  from `Broad Bioimage Benchmark <https://data.broadinstitute.org/bbbc/BBBC003/>`_.

* 
  Red Blood Cell `Images <https://github.com/Nelson-Gon/cytounet/tree/master/examples/BBBC009_v1>`_ provided by Anne 
  Carpenter and Roger Wiegand, available `here <https://data.broadinstitute.org/bbbc/BBBC009/>`_.

* 
  Chinese Hamster Ovary `Cells <https://github.com/Nelson-Gon/cytounet/tree/master/examples/BBBC030_v1>`_ provided by 
  Koos et al.(\ `2016 <https://bbbc.broadinstitute.org/BBBC030>`_\ )

**Credits**


* This repository started out as a clone of `zhixuhao <https://github.com/zhixuhao>`_\ 's  original 
  unet `implementation <https://github.com/zhixuhao/unet/>`_.

This repository has considerably diverged from the original implementation hence the need
to distribute it separately. 

This decision was taken in part due to the relative inactivity of the original implementation which would have made
it harder to collaborate. 
Please take a look at the list of `changes <https://github.com/Nelson-Gon/cytounet/blob/master/changelog.md>`_ 
from the original implementation. 


* The Unet algorithm was introduced by Ronneberger et al. in their 
  `paper <https://link.springer.com/chapter/10.1007%2F978-3-319-24574-4_28>`_.

----

If you would like to cite this work, please use:

Nelson Gonzabato(2020) cytounet: Deep Learning Based Cell Segmentation, https://github.com/Nelson-Gon/cytounet

BibTex

.. code-block::


   @misc{Gonzabato2021,
     author = {Gonzabato, N},
     title = {cytounet: Deep Learning Based Cell Segmentation},
     year = {2021},
     publisher = {GitHub},
     journal = {GitHub repository},
     howpublished = {\url{https://github.com/Nelson-Gon/cytounet}},
     commit = {58bd951ef4417fc8542f8f3e277071e6cd6980ea}
   }
