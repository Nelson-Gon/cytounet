from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_uint
from PIL import Image
import cv2


# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

def generate_train_data(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                        mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                        save_to_dir=None, target_size=(256, 256), seed=1):
    """

    :param batch_size: tensorflow batch size
    :param train_path: Path to training images
    :param image_folder: Path to a folder in train_path that holds the actual images
    :param mask_folder: Path to a folder in train_path that holds the masks
    :param aug_dict: An augmentation dict(see keras ImageDataGenerator for more)
    :param image_color_mode: One of rgb or grayscale. Defaults to grayscale
    :param mask_color_mode: One of rgb or grayscale. Defaults to grayscale
    :param image_save_prefix: Prefix to to add to augmented images
    :param mask_save_prefix: Prefix to add to augmented masks
    :param save_to_dir: If you needed to save augmented images, path to the target directory
    :param target_size: Size of images(reshape to this size). Defaults to (256, 256)
    :param seed: Reproducibility. May also affect results. Defaults to 1
    :return: A generator object to use with keras fit or fit_generator
    """

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)


def generate_test_data(test_path, num_image=30, target_size=(256, 256), image_suffix="tif"):
    """

    :param test_path: Path to test images
    :param num_image: Number of test images. Defaults to 30 for legacy reasons
    :param target_size: Target size(same as trainGenerator and unet layer 1)
    :param image_suffix: Image format. Defaults to tif
    :return: A test image generator object to feed to keras' predict_generator
    """
    img_width, img_height = target_size
    for i in range(num_image):
        img = image.load_img(glob.glob(test_path + "/*." + image_suffix)[i],
                             target_size=(img_width, img_height), color_mode="grayscale")
        img = image.img_to_array(img)
        img = img / 255.
        img = np.expand_dims(img, axis=0)
        yield img


def generate_validation_data(batch_size, validation_path, image_folder, mask_folder, aug_dict,
                             image_color_mode="grayscale",
                             mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                             save_to_dir=None, target_size=(256, 256), seed=1):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        validation_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        validation_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    valid_generator = zip(image_generator, mask_generator)
    for (img, mask) in valid_generator:
        yield (img, mask)


def load_augmentations(image_path, mask_path, image_prefix="image", mask_prefix="mask"):
    image_name_arr = glob.glob(os.path.join(image_path, "{}*.png".format(image_prefix)))
    image_arr = []
    mask_arr = []
    for index, item in enumerate(image_name_arr):
        img = image.load_img(item, color_mode="grayscale")
        img = image.img_to_array(img)
        img = img[:, :, 0]
        img = np.expand_dims(img, axis=0)
        img = img.transpose(2, 1, 0)  # make channels last
        mask = image.load_img(item.replace(image_path, mask_path).replace(image_prefix, mask_prefix),
                              color_mode="grayscale")
        mask = image.img_to_array(mask)
        mask = mask / 255.
        mask = mask[:, :, 0]
        mask = np.expand_dims(mask, axis=0)
        mask = mask.transpose(2, 1, 0)  # make channels last
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr, mask_arr


def save_predictions(directory, images, image_prefix=None, image_suffix="tif"):
    """


    :param image_prefix: Optional prefix to add to images eg msk or img
    :param directory: Directory to which to save images
    :param images: A list of image arrays
    :param image_suffix: Format, defaults to tif
    :return: Saved images

    """
    for index, item in enumerate(images):
        # needed for PIL
        item = (item * 255)[:, :, 0] if len(item.shape) == 3 else (item * 255)
        read_image = Image.fromarray(item.astype(np.uint8))
        read_image.save(directory + "/" + image_prefix + str(index) + "." + image_suffix)


def save_images(directory, images, image_prefix=None, image_suffix="tif"):
    """


    :param image_prefix: Optional prefix to add to images eg msk or img
    :param directory: Directory to which to save images
    :param images: A list of image arrays
    :param image_suffix: Format, defaults to tif
    :return: Saved images

    """
    for index, item in enumerate(images):
        item = item[:, :, 0] if len(item.shape) == 3 else item
        read_image = Image.fromarray(item.astype(np.uint8))
        read_image.save(directory + "/" + image_prefix + str(index) + "." + image_suffix)


def threshold_images(image_path, image_format="tif", thresh_val=128, thresh_max=255):
    """
    This is mostly useful as a wrapper for masks(labels)

    :param image_path: Path to images to threshold
    :param image_format: Format to save images to
    :param thresh_val: Thresholding threshold, defaults to 1
    :param thresh_max: Maximum value of pixels, defaults to 255
    :return: thresholded images

    """
    masks = glob.glob(image_path + "/*." + image_format)
    masks_arrays = [cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in masks]
    thresholded = [cv2.threshold(x, thresh_val, thresh_max,
                                 cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] for x in masks_arrays]
    return thresholded


