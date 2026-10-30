# SPDX-License-Identifier: MPL-2.0
from cupy._core import ndarray

def bincount(
    x: ndarray[tuple[int]], weights: ndarray[tuple[int]] | None = None, minlength: int | None = None
) -> ndarray[tuple[int]]: ...
