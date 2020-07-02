from .data import *
import glob
from skimage.io import ImageCollection
import matplotlib.pyplot as plt
import numpy as np


def show_images(directory="aug/mask", image_suffix="png",number=4, cmap="gray"):
    """

    :param directory: Directory holding images. Deafults to aug/mask to plot augmented masks.
    Adjust as necessary. Can be a list of images or a numpy ndarray instead.
    :param image_suffix: Image format, defaults to png
    :param image_type: Masks or original. Defaults to masks
    :param number: Number of images to show, only even numbers are supported
    :param cmap: Plot color cmap(as provided by imshow from matplotlib). Defaults to gray
    :return: A plot of images as requested.

    """
    # should really use os and sys to join paths
    if isinstance(directory, list) or isinstance(directory, np.ndarray):
        images = [image[:,:,0] for image in directory]
    else:
        images = ImageCollection(glob.glob(directory + "/*." + image_suffix))

    if number % 2 != 0:
        raise ValueError("Only an even number of images can be shown")

    plt.figure(figsize=(10, 10))

    for i in range(number):
        plt.subplot(number / 2, number / 2, i + 1)
        plt.imshow(images[i], cmap = cmap)


plt.show()