# SPDX-License-Identifier: MPL-2.0
from typing import Literal, overload

from cupy._core import ndarray

def argsort(a: ndarray, axis: int | None = -1, kind: Literal["stable"] | None = None) -> ndarray: ...
@overload
def lexsort(keys: ndarray[tuple[int, int]]) -> ndarray[tuple[int]]: ...
@overload
def lexsort(keys: ndarray) -> ndarray: ...
