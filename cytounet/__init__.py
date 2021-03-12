"""

Deep Learning based Cell Segmentation

.. author: Nelson Gonzabato

"""

import os
import sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from version import __version__

assert isinstance(__version__, str)
__author__ = "Nelson Gonzabato"
__version__ = __version__
__all__ = ["model", "data", "augmentation", "post_model"]
