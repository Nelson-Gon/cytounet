from .data import *
import glob
from skimage.io import ImageCollection
import matplotlib.pyplot as plt




def show_augmented(directory="aug", image_suffix="png", image_type="masks", number=4):
    """

    :param directory: Directory holding augmented images
    :param image_suffix: Image format, defaults to png
    :param image_type: Masks or original. Defaults to masks
    :param number: Number of images to show, only even numbers are supported
    :return: A plot of images as requested.
    """
    masks = ImageCollection(glob.glob(directory + "/mask*." + image_suffix))
    original = ImageCollection(glob.glob(directory + "/image*." + image_suffix))
    if image_type == "masks":
        use_data = masks
    else:
        use_data = original

    plt.figure(figsize=(10, 10))
    for i in range(number):
        plt.subplot(number / 2, number / 2, i + 1)
        plt.imshow(use_data[i])


plt.show()
