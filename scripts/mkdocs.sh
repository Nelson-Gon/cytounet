#!/bin/bash 
# This script allows building of docs 
# First we create corresponding rst files and move these to docs
# Next we run make.bat to build docs
# Finally we push to GitHub following doc builds
# Need to check that we are in the correct directory

base_dir="$(realpath "$0")"
actual_directory="$(dirname "$(dirname "$base_dir")")"
echo -e "\e[0;36m Writing and moving README, contributing, and changelog to docs folder"
python -m m2r README.md changelog.md .github/CONTRIBUTING.md --overwrite && mv .github/CONTRIBUTING.rst README.rst changelog.rst docs/source

echo "Building docs for" "$actual_directory"
./make.bat html
echo  -e "\e[0;36m All done, commit latest changes if you wish."


