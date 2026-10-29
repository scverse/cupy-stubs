# SPDX-License-Identifier: MPL-2.0
from typing import Literal

from cupy._core import ndarray

def searchsorted(
    a: ndarray, v: ndarray | complex, side: Literal["left", "right"] = "left", sorter: ndarray | None = None
) -> ndarray: ...
def flatnonzero(a: ndarray) -> ndarray[tuple[int]]: ...
def nonzero(a: ndarray) -> tuple[ndarray[tuple[int]], ...]: ...
