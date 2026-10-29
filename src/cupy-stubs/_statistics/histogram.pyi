# SPDX-License-Identifier: MPL-2.0
from cupy._core import ndarray

def bincount(x: ndarray, weights: ndarray | None = None, minlength: int | None = None) -> ndarray[tuple[int]]: ...
