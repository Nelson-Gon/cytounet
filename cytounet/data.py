from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import glob
from PIL import Image
import cv2


def assert_path_exists(dir_path, image_suffix=None, path_ext=None):
    if not os.path.exists(dir_path):
        raise NotADirectoryError(f"{dir_path} is not a valid directory")
    if path_ext is not None:
        dir_path = os.path.join(dir_path, path_ext)
    if image_suffix is not None:
        files = sorted(glob.glob(dir_path + "/*." + image_suffix))
        if len(files) == 0:
            raise ValueError(f"{dir_path} contains no images.")


def generate_train_data(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                        mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                        save_to_dir=None, target_size=(256, 256), seed=1, show_names=True):
    """

    :param show_names: Boolean. Should filenames be printed? Defaults to True
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

    assert_path_exists(train_path, path_ext=image_folder)

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
        shuffle=True,
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
        shuffle=True,
        save_prefix=mask_save_prefix,
        seed=seed)
    if show_names:
        print(image_generator.filenames)
        print(mask_generator.filenames)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield img, mask


def generate_test_data(test_path, train_seed, target_size=(256, 256), show_names=True):
    """

    :param show_names: Boolean. Should filenames be printed? Defaults to True
    :param test_path: Path to test images
    :param train_seed: Same seed used in generate_train_data
    :param target_size: Target size(same as generate_train_data and unet's input layer)
    :return: A test image generator object to feed to keras' predict

    """
    assert_path_exists(test_path)
    test_data_gen = ImageDataGenerator(rescale=1 / 255.)
    test_data_gen = test_data_gen.flow_from_directory(directory=test_path,
                                                      target_size=target_size,
                                                      class_mode=None,
                                                      batch_size=1,
                                                      color_mode="grayscale",
                                                      seed=train_seed,
                                                      shuffle=False)
    if show_names:
        print(test_data_gen.filenames)
    return test_data_gen


def generate_validation_data(batch_size, validation_path, image_folder, mask_folder, aug_dict,
                             image_color_mode="grayscale",
                             mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                             save_to_dir=None, target_size=(256, 256), seed=1, show_names=True):
    """
    :param show_names: Boolean. Should filenames be printed? Defaults to True
    :param batch_size: tensorflow batch size
    :param validation_path: Path to training images
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
    :return: A generator object to supply to the validation_data argument of keras fit or previously fit_generator

    """
    assert_path_exists(validation_path)
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
    if show_names:
        print(image_generator.filenames)
        print(mask_generator.filenames)

    for (img, mask) in valid_generator:
        yield img, mask


def load_augmentations(image_path, mask_path, image_prefix="image", mask_prefix="mask", image_suffix="png",
                       target_size=(512, 512)):
    """

    :param image_path: Path to augmented images
    :param mask_path: Path to augmented masks
    :param image_prefix: Image filename prefix. Defaults to image
    :param mask_prefix: Mask filename prefix. Defaults to mask
    :param image_suffix: Image format. Defaults to tif
    :param target_size: Size to set. Defaults to (512, 512). Should be the same size as that defined for the model
    input
    :return: A tuple of images and masks

    """

    assert_path_exists(image_path, image_suffix=image_suffix)
    assert_path_exists(mask_path, image_suffix=image_suffix)

    image_name_arr = glob.glob(os.path.join(image_path, "{}*.{}".format(image_prefix, image_suffix)))
    image_arr = []
    mask_arr = []
    for index, item in enumerate(image_name_arr):
        img = image.load_img(item, color_mode="grayscale", target_size=target_size)
        img = image.img_to_array(img)
        # img = img[:, :, 0]
        # img = np.expand_dims(img, axis=0)
        # img = img.transpose(2, 1, 0)  # make channels last
        mask = image.load_img(item.replace(image_path, mask_path).replace(image_prefix, mask_prefix),
                              color_mode="grayscale", target_size=target_size)
        mask = image.img_to_array(mask)
        # mask = mask / 255.
        # mask = mask[:, :, 0]
        # mask = np.expand_dims(mask, axis=0)
        # mask = mask.transpose(2, 1, 0)  # make channels last
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
