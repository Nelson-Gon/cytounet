[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3928919.svg)](https://doi.org/10.5281/zenodo.3928919)
![Stage](https://www.repostatus.org/badges/latest/active.svg) 
![Test Install](https://github.com/Nelson-Gon/cytounet/workflows/Test%20Install/badge.svg)
[![PyPI version](https://badge.fury.io/py/cytounet.svg)](https://badge.fury.io/py/cytounet) 
[![PyPI Downloads Month](https://img.shields.io/pypi/dm/cytounet.svg)](https://pypi.python.org/pypi/cytounet/)
[![PyPI license](https://img.shields.io/pypi/l/cytounet.svg)](https://pypi.python.org/pypi/cytounet/) 
![Travis Build](https://travis-ci.com/Nelson-Gon/cytounet.svg?branch=master)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Nelson-Gon/cytounet/graphs/commit-activity)
[![GitHub last commit](https://img.shields.io/github/last-commit/Nelson-Gon/cytounet.svg)](https://github.com/Nelson-Gon/cytounet/commits/master)
[![GitHub issues](https://img.shields.io/github/issues/Nelson-Gon/cytounet.svg)](https://GitHub.com/Nelson-Gon/cytounet/issues/)
[![GitHub issues-closed](https://img.shields.io/github/issues-closed/Nelson-Gon/cytounet.svg)](https://GitHub.com/Nelson-Gon/cytounet/issues?q=is%3Aissue+is%3Aclosed)


---


**Installation**

From `PyPI`:

```

pip install cytounet

```


From source or to use without installing locally:

```
git clone https://github.com/Nelson-Gon/cytounet.git
# proceed with usual source build procedure

```

Or:

```python

pip install git+https://github.com/Nelson-Gon/cytounet.git

```

**Import**

```python

from cytounet.model import *
from cytounet.data import *
from cytounet.augmentation import *
from cytounet.post_model import *

```

**Detailed Sample Usage**

Please see the following examples:

* [Typical Usage](https://github.com/Nelson-Gon/cytounet/blob/7fd42a27be1b5730eb05e60cb98d5b7e825a0087/examples/example_usage.ipynb)

[![nbviewer](https://img.shields.io/badge/view%20on-nbviewer-brightgreen.svg)](https://nbviewer.jupyter.org/github/Nelson-Gon/cytounet/blob/7fd42a27be1b5730eb05e60cb98d5b7e825a0087/examples/example_usage.ipynb)


* [Predicting Embryonic DIC Image Labels with Keras](https://www.kaggle.com/gonnel/predicting-embryonic-dic-image-labels-with-keras)

[![nbviewer](https://img.shields.io/badge/view%20on-nbviewer-brightgreen.svg)](https://nbviewer.jupyter.org/github/Nelson-Gon/cytounet/blob/aedf8d52af4e3e9f2cd426de90b4c5dea2a4e11c/examples/embryos_dic.ipynb)






Visually:

![CHO](https://raw.githubusercontent.com/Nelson-Gon/cytounet/master/examples/example_results.png)

Sample Object Area Visualization(see the typical usage notebook above for detailed usage)

![Area Determination](https://raw.githubusercontent.com/Nelson-Gon/cytounet/master/examples/areas.png)

For more examples or to add your own, please see the examples [folder](https://github.com/Nelson-Gon/cytounet/blob/master/examples).

**Experiments/Benchmarks**

This section shows some experimental results based on publicly available data. 

* Comparison of low vs high quality masks on the model's output

This notebook shows the effects of "filled holes"(outlines whose area is filled with some colour e.g white)
on the model's quality. The results in general show that filled masks which are also better seen by the human eye
result in better quality output. 

The [notebook](https://github.com/Nelson-Gon/cytounet/blob/9781a45260bd8cdb82b37e07a26254ecf01af5c7/examples/example_usage.ipynb) can be accessed via this [link](https://nbviewer.jupyter.org/github/Nelson-Gon/cytounet/blob/9781a45260bd8cdb82b37e07a26254ecf01af5c7/examples/example_usage.ipynb).



**TODO List**

A checked box indicates support. You can either add more feature requests here or tackle unchecked boxes and make
a pull request to add such support. 



- [x] Single class segmentation

- [x] Grayscale images

- [x] Model Validation

- [x] Determining areas of objects in an image

- [ ] Custom Number of Layers

- [ ] Multi-class segmentation

- [ ]  Colored image input

- [ ] COCO Datasets 

- [ ] CSV Based Annotations

- [ ] XML Based Annotations 

**Frequently Asked Questions**

Please read our Wiki [Pages](https://github.com/Nelson-Gon/cytounet/wiki)

**To raise an issue or question**

If the [wiki](https://github.com/Nelson-Gon/cytounet/wiki) does not answer your question,
please raise a new [issue](https://github.com/Nelson-Gon/cytounet/issues). You can also open an issue if you have any discussion, criticism,
or bug reports. 

Thank you very much. 

---

**References**

* Mouse Embryos [Dataset obtained](https://github.com/Nelson-Gon/cytounet/tree/master/examples/BBBC003_v1) from Broad Bioimage Benchmark Collection.
[Source](https://data.broadinstitute.org/bbbc/BBBC003/).

* Red Blood Cell [Images](https://github.com/Nelson-Gon/cytounet/tree/master/examples/BBBC009_v1) provided by Anne 
Carpenter and Roger Wiegand, available [here](https://data.broadinstitute.org/bbbc/BBBC009/)

* Chinese Hamster Ovary [Cells](https://github.com/Nelson-Gon/cytounet/tree/master/examples/BBBC030_v1) provided by 
Koos et al.([2016](https://bbbc.broadinstitute.org/BBBC030))

**Credits**

* This repository started out as a clone of [zhixuhao](https://github.com/zhixuhao)'s  original 
unet [implementation](https://github.com/zhixuhao/unet/).

This repository has considerably diverged from the original implementation hence the need
to distribute it separately. 

This decision was taken in part due to the relative inactivity of the original implementation which would have made
it harder to collaborate. Please take a look at the list of [changes](https://github.com/Nelson-Gon/cytounet/blob/master/changelog.md) 
from the original implementation. 

* The Unet algorithm was introduced by Ronneberger et al. in their [paper](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).



