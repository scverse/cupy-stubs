# SPDX-License-Identifier: MPL-2.0
import numpy as np

from cupy._core import ndarray

def array_equal(a1: ndarray, a2: ndarray, equal_nan: bool = False) -> ndarray[tuple[()], np.dtype[np.bool]]: ...
