# cytounet's changelog. 

**Version 0.2.0**

* Fixed issues with list input to `show_images`


---
* Release 0.1.0

---

* Renamed repository to `cytounet` to reflect the havy focus on biological images.

* Initiated support for validation via `validGenerator`.

* Fixed issues with `show_images` failing to load `numpy` `ndarray` images.   

---

* Initiated ability to install with `pip` and `setup.py`.

---

* `show_augmented` was renamed to `show_images` and refactored as a more general method not limited

to just augmented images. A `cmap` argument was also added for more flexibility. This replaces `labelVisualize`
which has now been dropped. 

* Introduced a separate save method for images and predictions. Use `saveImages` and `savePredictions`
respectively. 

---

* Fixed issues with information loss following saving of predictions. 

* `geneTrainNpY` was refactored and renamed `LoadAugmented`

* Added `thresholdImages` to threshold masks(mostly). Please see [pyautocv](https://github.com/Nelson-Gon/pyautocv)
for a more general and flexible way to manipulate images. 

* Added `saveImages`, a helper to save images as(by default) `.tif`. This is because biological
images are normally tiff in nature.

* Removed `savePredictions`. Use `saveImages` instead. 

---

* Updated module documentation 

* `adjustData` was removed since it had known issues. It may be restored in the future. 

* Fixed issues that resulted in blank predictions 

* Added `show_augmented` to show results of data augmentation

* Added `BatchNormarmalisation` steps

* Training made more flexible by allowing usage of different metrics and loss functions without editing source code(i.e change on the fly)

* Saving and image reading functions made more flexible to read/save any image file format.

* Made most functions compatible with Keras >= 2.0 

* Added `dice` loss and dice coefficient.


