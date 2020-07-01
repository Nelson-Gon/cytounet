![Stage](https://www.repostatus.org/badges/latest/wip.svg) 
![Test Install](https://github.com/Nelson-Gon/unet/workflows/Test%20Install/badge.svg)
![Travis Build](https://travis-ci.com/Nelson-Gon/unet.svg?branch=master)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Nelson-Gon/unet/graphs/commit-activity)
[![GitHub last commit](https://img.shields.io/github/last-commit/Nelson-Gon/unet.svg)](https://github.com/Nelson-Gon/unet/commits/master)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![GitHub issues](https://img.shields.io/github/issues/Nelson-Gon/unet.svg)](https://GitHub.com/Nelson-Gon/unet/issues/)
[![GitHub issues-closed](https://img.shields.io/github/issues-closed/Nelson-Gon/unet.svg)](https://GitHub.com/Nelson-Gon/unet/issues?q=is%3Aissue+is%3Aclosed)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Nelson-Gon/pyautocv/blob/master/LICENSE)

---


**Installation**


To install:

```
git clone https://github.com/Nelson-Gon/unet.git

```

Or:

```python

pip install git+https://github.com/Nelson-Gon/unet.git

```

**Sample Usage**

Sample usage is shown in a Colab notebook available in the 
examples [folder](https://github.com/Nelson-Gon/unet/blob/master/examples/example_usage.ipynb).

**Is it supported?**

A checked box indicates support. You can either add more feature requests here or tackle unchecked boxes and make
a pull request to add such support. 

- [x] Single class segmentation

- [x] Grayscale images

- [ ] Multi-class segmentation

- [ ]  Colored image input

- [ ] COCO Datasets 

- [ ] CSV Based Annotations

- [ ] XML Based Annotations 


**To raise an issue or question**

Please raise an issue [here](https://github.com/Nelson-Gon/unet/issues) if you have any discussion, criticism,

or bug reports. 

Thank you very much. 

---

**References**

Mouse Embryos [Dataset obtained](https://github.com/Nelson-Gon/unet/tree/master/examples/BBBC003_v1) from Broad Bioimage Benchmark Collection.
[Source](https://data.broadinstitute.org/bbbc/BBBC003/).

**Credits**

This repository started out as a clone of [zhuxihao](https://github.com/zhixuhao)'s  original 
unet [implementation](https://github.com/zhixuhao/unet/).

This repository has considerably diverged from the original implementation hence the need
to distribute it separately. 

This decision was taken in part due to the relative inactivity of the original implementation which would have made
it harder to collaborate. Please take a look at the list of changes from the original implementation
[here](https://github.com/Nelson-Gon/unet/blob/master/changelog.md). 

The Unet algorithm was introduced by Ronneberger et al. in their [paper](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).



