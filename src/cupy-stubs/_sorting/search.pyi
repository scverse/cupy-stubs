# SPDX-License-Identifier: MPL-2.0
from typing import Literal, overload

import numpy as np

from cupy._core import ndarray
from cupy._core.core import _Shape

@overload
def searchsorted[ShapeT: _Shape](
    a: ndarray[tuple[int]],
    v: ndarray[ShapeT],
    side: Literal["left", "right"] = "left",
    sorter: ndarray[tuple[int]] | None = None,
) -> ndarray[ShapeT, np.dtype[np.int64]]: ...
@overload
def searchsorted(
    a: ndarray[tuple[int]],
    v: complex,
    side: Literal["left", "right"] = "left",
    sorter: ndarray[tuple[int]] | None = None,
) -> ndarray[tuple[()], np.dtype[np.int64]]: ...
def flatnonzero(a: ndarray) -> ndarray[tuple[int]]: ...
def nonzero(a: ndarray) -> tuple[ndarray[tuple[int]], ...]: ...
@overload
def where(condition: ndarray, x: None = None, y: None = None) -> tuple[ndarray[tuple[int]], ...]: ...
@overload
def where(condition: ndarray, x: ndarray | complex, y: ndarray | complex) -> ndarray: ...
