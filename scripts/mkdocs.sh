#!/bin/bash 
# This script allows building of docs 
# First we create corresponding rst files and move these to docs
# Next we run make.bat to build docs
# Finally we push to GitHub following doc builds
# Need to check that we are in the correct directory
base_dir="$(realpath "$0")"
file_location="$(dirname "$base_dir")"
modules_location="$(dirname "$file_location")"
echo "Doc generator is located in" "$file_location" "while modules are in" "$modules_location"
echo -e "\e[0;36m Writing and moving README, contributing, and changelog to docs folder";

if [ ! -f docs/source/modules.rst ]
    then
      echo "Creating skeletons for" "$1", "ignore if already done" && sphinx-apidoc -o docs/source "$1"

    fi;

python -m m2r README.md changelog.md .github/CONTRIBUTING.md --overwrite
mv .github/CONTRIBUTING.rst README.rst changelog.rst docs/source

echo "Building docs for" "$(dirname "$file_location")"
./make.bat html
echo  -e "\e[0;36m All done, commit latest changes if you wish."
echo  -e "\e[0;36m If there were errors or warnings, delete modules.rst and ensure index.rst exists before trying again."


