# Changes to cytounet 

**Version 0.2.0**

* Kernel regularization can now be turned off via a boolean argument(use_regularizer)

* Added a new data set from BBBC. 

* `finetune` is a new function dedicated to the finetuning workflow. 

* Regularization is now supported. It is currently limited to L1 and L2.

* `pretrained_weights` was dropped as an argument to `unet`. Use a `callback` instead. A future
version wil include a fine tuning function. 

* `save_as` was removed from `train`. Use ModelCheckpoint instead and provide it as a callback. 

* `show_images` now shows titles. These functions will be removed later and imported from `pyautocv`
instead. 

* Fixed issues with reading mixed `jpg` and `png` images. 

* Added `reshape_images` and `resize_images`. These are helper functions that may be useful when plotting
or restoring original image size. 

* `show_images` and `read_images` are now imported from `pyautocv` >= 0.2.2

* Fixed issues with inconsistent image order in `show_images` when reading from a directory.

* Added filename printing to data generators to make it easier to show what order the files are
being read in. This can be disabled by setting `show_names` to `False`. 

* Changes to prediction generation were made. We now use `ImageDataGenerator` for
test time data generation. 

* Fixed a bug related to `load_augmentations` that led to image flipping. 

* Changed outputs to `sigmoid` instead of `ReLU`

* Updated to latest API ie `predict` vs `predict_generator`

* Added `train` to simplify model fitting.

* Added `predict` to reduce code repetition and make predicting easier. 

* `unet` was rewritten to increase complexity and solve issues with blank predictions. It now also uses `Conv2DTranspose` instead of `UpSampling2D`. 

* Initial support for a simpler model to optimise the bias-variance trade off for small(er) datasets.

* Removed `Dropout` since this is known to have no improvement over Batch Normalisation. 

* Initial support for SGD as the default optimiser

* Moved from camelCase to snake_case, now using more descriptive function names. 

* Fixed issues with list input to `show_images`


---
* Release 0.1.0

---

* Renamed repository to `cytounet` to reflect the heavy focus on biological images.

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


