# Changes to zhixuhao's original implementation

* `geneTrainNpY` was refactored and renamed `LoadAugmented`

* Added `thresholdImages` to threshold masks(mostly). Please see [pyautocv](https://github.com/Nelson-Gon/pyautocv)
for a more general and flexible way to manipulate images. 

* Added `saveImages`, a helper to save images as(by default) `.tif`. This is because biological
images are normally tiff in nature.

* Removed `savePredictions`. Use `saveImages` instead. 

* Updated module documentation 

* `adjustData` was removed since it had known issues. It may be restored in the future. 

* Fixed issues that resulted in blank predictions 

* Added `show_augmented` to show results of data augmentation

* Added `BatchNormarmalisation` steps

* Training made more flexible by allowing usage of different metrics and loss functions without editing source code(i.e change on the fly)

* Saving and image reading functions made more flexible to read/save any image file format.

* Made most functions compatible with Keras >= 2.0 

* Added `dice` loss and dice coefficient.


