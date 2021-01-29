#!/bin/bash
# Checks that all necessary steps have been done for pypi release
check_status (){
  read -p "Have you updated documentation with mkdocs?!" docs_answer
  case "$docs_answer" in
  [yY][eE][sS] | [yY] ) read -p "Have you updated the version number in all places?" version_answer
    case "$version_answer" in
    [yY][eE][sS] | [yY] ) echo "Perfect. Cleaning up and uploading to test.pypi"
    if [ -d dist ]
    then
      rm -r dist;
    fi;
    python setup.py sdist;python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*;;
    [nN][oO] ) echo "Please update version number first";exit 1;;
    esac;;
  [nN][oO] | [nN] ) echo "Please update docs first";;
  esac
}

upload_to_pypi (){
  read -p "Did you upload to test.pypi and are happy with the results?" checkmate
  case "$checkmate" in
  [yY][eE][sS] | [yY] ) echo "Uploading to pypi";python -m twine upload dist/*;;
  [nN][oO] | [Nn] ) echo "Please run the script again until you are happy with the results";;
  esac

}

check_status
upload_to_pypi



