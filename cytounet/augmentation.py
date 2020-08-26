# These functions were originally part of pyautocv. To prevent excessive dependencies
# and because pyautocv was not yet released, I kept these functions here.
# In the future, they will be removed and imported from pyautocv instead

import matplotlib.pyplot as plt
from itertools import chain
from skimage.io import imread, imread_collection
from skimage.transform import resize
import glob
from os import pathsep


def reshape_images(image_list):
    """

    :param image_list: A list of images to reshape for plotting
    :return: Images that can be plotted with show_images

    """
    final_list = [img[:, :, 0] if len(img.shape) == 3 and img.shape[2] != 3 else img for img in image_list]
    return final_list


def read_images(directory, image_suffix="tif", other_directory=None):
    # This currently only supports grayscale images, see pyautocv for better support
    """

    :param directory: Directory containing tiff only or mixed png/jpg images to be read
    :param image_suffix: Suffix of images in the folder. Currently only supports tif, png, and jpg
    :param other_directory: If images exist in another folder/sub folder, please provide it here. Leave blank if
    mixed file formats(jpg and png) exist in the same folder

    :return: A list containing arrays of images

    """
    # read png and jpg from current directory
    if image_suffix == "tif":
        images_list = sorted(glob.glob(directory + "/*.tif"))
        return [imread(x, plugin='pil') for x in images_list]
    else:
        if image_suffix not in ["png", "jpg"]:
            raise ValueError("Only tif, png, and jpg are currently supported")
        if other_directory is None:
            other_directory = directory
        return list(imread_collection(directory + "/*.jpg" + pathsep + other_directory + "/*.png"))


def show_images(original_images=None, processed_images=None, cmap="gray", number=None, figure_size=(20, 20),
                titles=None):
    """
    :param titles: A list of length 2 for titles to use. Defaults to ['original','processed']
    :param figure_size: Size of the plot shown. Defaults to (20,20)
    :param original_images: Original Images from read_images()
    :param processed_images: Images that have been converted eg from detect_edges()
    :param cmap: Color cmap from matplotlib. Defaults to gray
    :param number: optional Number of images to show
    :return A matplotlib plot of images

    """

    if original_images is None or processed_images is None:
        raise ValueError("Both original and processed image lists are required.")
    if number is not None:
        original_images = reshape_images(original_images[:number])
        processed_images = reshape_images(processed_images[:number])

    image_list = list(chain(*zip(original_images, processed_images)))

    if len(image_list) % 2 == 0:
        ncols = len(image_list) / 2
    else:
        ncols = len(image_list)

    fig, axes = plt.subplots(nrows=2, ncols=int(ncols), figsize=figure_size)
    if titles is None:
        titles = ['original', 'processed']

    titles = titles * len(image_list)
    for ind, image in enumerate(image_list):
        axes.ravel()[ind].imshow(image_list[ind], cmap=cmap)
        axes.ravel()[ind].set_title(f'{titles[ind]}')
        axes.ravel()[ind].set_axis_off()


def resize_images(image_list, target_size):
    """

    :param image_list: A list of images or image that needs to be resized
    :param target_size: New target image size
    :return: Image or list of images that have been resized.


    """

    return [resize(x, target_size) for x in image_list]
