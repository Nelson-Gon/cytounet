# Changes to zhixuhao's original implementation

* Updated module documentation 

* `adjustData` was removed since it had known issues. It may be restored in the future. 

* Fixed issues tha resulted in blank predictions 

* Added `show_augmented` to show results of data augmentation

* Added `BatchNormarmalisation` steps

* Training made more flexible by allowing usage of different metrics and loss functions without editing source code(i.e change on the fly)

* Saving and image reading functions made more flexible to read/save any image file format.

* Made most functions compatible with Keras >= 2.0 

* Added `dice` loss and dice coefficient.


