from .data import *
import glob
from skimage.io import ImageCollection
import matplotlib.pyplot as plt


# augment data

def show_augmented(directory="aug", image_suffix="png", image_type="masks", number=4):
    masks = ImageCollection(glob.glob(directory + "/mask*." + image_suffix))
    original = ImageCollection(glob.glob(directory + "/image/*." + image_suffix))
    use_data = original
    if image_type == "masks":
        use_data = masks

    plt.figure(figsize=(10, 10))
    for i in range(number):
        plt.subplot(number / 2, number / 2, i + 1)
        plt.imshow(use_data[i])


plt.show()
