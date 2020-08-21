# These functions were originally part of pyautocv. To prevent excessive dependencies
# and because pyautocv was not yet released, I kept these functions here.
# In the future, they will be removed and imported from pyautocv instead

import matplotlib.pyplot as plt
from itertools import chain
from skimage.io import imread, imread_collection
import glob
from os import pathsep


def read_images(directory, image_suffix="tif"):
    # This currently only supports grayscale images, see pyautocv for better support

    """

    :return: Returns a multidimensional array containing arrays that represent images in a directory

    """
    # read png and jpg from current directory
    if image_suffix == "tif":
        images_list = sorted(glob.glob(directory + "/*.tif"))
        return [imread(x, plugin='pil') for x in images_list]
    else:
        if image_suffix not in ["png", "jpg"]:
            raise ValueError("Only tif, png, and jpg are currently supported")
        return list(imread_collection(directory + "/*.jpg" + pathsep + "/*.png"))


def show_images(original_images=None, processed_images=None, cmap="gray", number=None, figure_size=(20, 20)):
    """
    :param figure_size: Size of the plot shown. Defaults to (20,20)
    :param original_images: Original Images from read_images()
    :param processed_images: Images that have been converted eg from detect_edges()
    :param cmap: Color cmap from matplotlib. Defaults to gray
    :param number: optional Number of images to show
    """
    # need to figure out how any works in python
    if original_images is None or processed_images is None:
        raise ValueError("Both original and processed image lists are required.")
    if number is not None:
        original_images = original_images[:number]
        processed_images = processed_images[:number]

    image_list = list(chain(*zip(original_images, processed_images)))

    if len(image_list) % 2 == 0:
        ncols = len(image_list) / 2
    else:
        ncols = len(image_list)

    fig, axes = plt.subplots(nrows=2, ncols=int(ncols), figsize=figure_size)
    for ind, image in enumerate(image_list):
        axes.ravel()[ind].imshow(image_list[ind], cmap=cmap)
        axes.ravel()[ind].set_axis_off()
