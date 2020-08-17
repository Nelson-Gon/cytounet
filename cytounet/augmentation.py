from .data import *
import glob
from skimage.io import ImageCollection
import matplotlib.pyplot as plt
import numpy as np


def show_images(directory="aug/mask", image_suffix="png", number=4, cmap="gray"):
    """

    :param directory: Directory holding images. Deafults to aug/mask to plot augmented masks.
    Adjust as necessary. Can be a list of images or a numpy ndarray instead.
    :param image_suffix: Image format, defaults to png
    :param image_type: Masks or original. Defaults to masks
    :param number: Number of images to show
    :param cmap: Plot color cmap(as provided by imshow from matplotlib). Defaults to gray
    :return: A plot of images as requested.

    """
    # should really use os and sys to join paths
    if isinstance(directory, (list, np.ndarray)):
        # convert to viewable format by imshow, only considers len 3 for now
        images = [image[:, :, 0] if len(image.shape) == 3 else image for image in directory]
    else:
        images = ImageCollection(sorted(glob.glob(directory + "/*." + image_suffix)))

    no_cols = number / 2 if number % 2 == 0 else number / 3

    # no_rows= number / 2 if number % 2 == 0 else number / 3

    fig, axes = plt.subplots(nrows=2, ncols=int(no_cols))
    fig.set_size_inches(10, 10)
    for index, item in zip(np.arange(number), images):
        axes.ravel()[index].imshow(item, cmap=cmap)
        axes.ravel()[index].set_axis_off()