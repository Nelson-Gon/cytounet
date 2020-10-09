# Steps Taken
# Load relevant libraries
import cv2
from .augmentation import *
from copy import deepcopy


# Read an Image
# Here we read images predicted by the UNet algorithm
def read_image_spec(directory, image_suffix="png", load_format=None, verbose=True):
    """
    :param load_format If specified, only files that match the given pattern are returned. Otherwise, all images that
    end in the target format are read.
    :param directory Path to folder containing images to read
    :param image_suffix Str of image formats to read. Currently only supports reading a single format at a time
    :param verbose For debugging or aesthetic purposes, printing what is being read can be turned
    on by setting this to True
    :return Images to input to other functions
    """
    # Ideally, we would use read_images but this sometimes gives weird image shapes(4 channels for example)
    if load_format is None:
        load_format = "*"
    to_read = sorted(glob.glob(directory + "/" + load_format + "." + image_suffix))
    if verbose:
        [print("Reading {}".format(x)) for x in to_read]
    return [cv2.imread(file_name) for file_name in to_read]


def find_contours(image_list, threshold_val=20, max_threshold=255, retr_mode=None, return_edges=False):
    """
    :param image_list A list of images(from cytounet/pyautocv's read_images) whose contours are required
    :param threshold_val A threshold value to use in finding edges using canny edge detection
    :param max_threshold Maximum pixel value. Defaults to 255
    :param retr_mode Retrieval mode for opencv's findContours function. Defaults to RETR_EXTERNAL
    :param return_edges Boolean. If True, the result from canny edge detection is returned for further \
    processing/viewing.
    :return Contours and their calculated areas. Results of canny edge detection are optionally returned depending
    on the return_edges argument.

    """
    if retr_mode is None:
        retr_mode = cv2.RETR_EXTERNAL
    # detect edges for a list of images with canny edge detection
    # use one threshold to rule them all
    canny_edge_result = [cv2.Canny(x, threshold_val, max_threshold) for x in image_list]

    # iterate over these edges and find contours
    contours_list = []
    areas_list = []
    for img in canny_edge_result:
        # ignoring hierarchies for now
        calculated_contours, hierarchy = cv2.findContours(img, retr_mode, cv2.CHAIN_APPROX_NONE)
        contours_list.append(calculated_contours)
    # Get contour areas for every contour in the list
    for contour in contours_list:
        areas_list.append([cv2.contourArea(x) for x in contour])

    final_return = contours_list, areas_list
    if return_edges:
        final_return = contours_list, areas_list, canny_edge_result

    return final_return


# Draw contours that meet a minimum(or maximum) area
# Minimum chosen since it is a binary image

def draw_contours(areas_list, contours_list, original_images, min_area=0, font_size=2,
                  number=None, show_text=True, **kwargs):
    """
    :param areas_list A list/array containing areas from find_contours
    :param contours_list A list/array containing contours from find_contours
    :param images_list Images on which to draw contours
    :param original_images A copy of the same images as images_list, this is useful for visualization purposes.
    :param min_area Minimum area required for a contour to be drawn onto the image
    :param number Number of images to show, defaults to 4.
    :param font_size Font Size for the text displayed.
    :show_text Should the area be displayed on the image? Defaults to True
    :return A matplotlib plot of objects(contours) and their areas

    """
    if number is None:
        number = len(original_images)

    original_images_copy = deepcopy(original_images[:number])
    slice_images = original_images[:number]
    slice_contours = contours_list[:number]
    slice_areas = areas_list[:number]

    for img, contour, area in zip(slice_images, slice_contours, slice_areas):

        if len(img.shape) == 2 or img.shape[2] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for specific_contour, specific_area in zip(contour, area):
            if specific_area > min_area:
                cv2.drawContours(img, specific_contour, -1, (255, 140, 0), 3)
                x_value, y_value, width, height = cv2.boundingRect(specific_contour)
                if show_text:
                    cv2.putText(img, str(specific_area), (x_value, y_value), cv2.FONT_HERSHEY_SIMPLEX, int(font_size),
                            (255, 255, 255), 3)
    print("Returning {} images as requested".format(number))
    return show_images(original_images_copy,original_images, **kwargs), original_images_copy
