# Contributing to cytounet

This document provides guidelines for contributions to `cytounet`.

**Kinds of contribution**

* Typo fixes
* Documentation enhancements
* Pull requests


**Fixing typos and enhancing documentation**

To fix typos and/or grammatical errors, please edit the corresponding `.py` or `.md` file that generates the documentation. 

Please also update the docs using `sphinx`

**Pull Requests**

* Please raise an issue for discussion and reproducibility checks at [issues](https://github.com/Nelson-Gon/cytounet/issues)

* Once the bug/enhancement is approved, please create a Git branch for the pull request.

* Make changes and ensure that builds are passing the necessary checks on Travis.

* Update `changelog.md` to reflect the changes made.

* Do the following:



```
bash scripts/mkdocs.sh 
```

**Commit messages**

Please write commit messages in the format "Extends functionality" instead of "Extended functionality".


**Releasing**

```shell
bash scripts/release.sh
```
The above does the following:

 - Makes `dist` with `python setup.py sdist` at the very minimum. Ensure everything necessary is included in
 `Manifest.in`. 
 - Uploads `dist` to test.pypi.org with `twine upload --repository-url https://test.pypi.org/legacy/ dist/*`
 - If everything looks good, asks you to upload to pypi.org with `twine upload dist/*`
 
Please note that the 'cytounet' project is released with a
[Contributor Code of Conduct](https://github/com/Nelson-Gon/cytounet/.github/CODE_OF_CONDUCT.md).
By contributing to this project, you agree to abide by its terms.

[See also](https://samnicholls.net/2016/06/15/how-to-sphinx-readthedocs/) for a guide on Sphinx documentation.