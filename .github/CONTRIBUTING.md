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

# The Makefile here is Windows specific

# root of project
python -m m2r README.md --overwrite
python -m m2r changelog.md --overwrite
# copy changelog and README or get their diff and copy it to docs/source
# TODO
cd ./README* ./changelog* docs/source
# if not done before, use sphnix-autostart and edit the corresponding files ie conf.py 
cd docs
# build docs
# Ensure that you can import modules
# Due to hardware limits, I used mock imports but that should ideally be 
# avoided
sphinx-build source build
# use make on *nix or if you have make on Windows
# Check that the batch file points to the right directories 
make.bat html
# Check built docs
sphinx-build source -W -b linkcheck -d build/doctrees build/html

```
Please note that the 'cytounet' project is released with a
[Contributor Code of Conduct](https://github/com/Nelson-Gon/cytounet/.github/CODE_OF_CONDUCT.md).
By contributing to this project, you agree to abide by its terms.

[See also](https://samnicholls.net/2016/06/15/how-to-sphinx-readthedocs/) for a guide on Sphinx documentation.

* Releasing
 - Make `dist` with `python setup.py sdist` at the very minimum. Ensure everything necessary is included in
 `Manifest.in`. 
 - Upload `dist` to test.pypi.org with `twine upload --repository-url https://test.pypi.org/legacy/ dist/*`
 - If everything looks good, upload to pypi.org with `twine upload dist/*`
 