"""

Deep Learning based Cell Segmentation

.. author: Nelson Gonzabato

"""

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
__author__ = "Nelson Gonzabato"
__version__ = "0.2.0".encode("ascii", "ignore").decode('ascii')
__all__ = ["model", "data", "augmentation"]
