# SPDX-License-Identifier: MPL-2.0
from typing import Literal, overload

from numpy.typing import ArrayLike

from cupy._core import ndarray

def append(arr: ndarray, values: ndarray | ArrayLike, axis: int | None = None) -> ndarray: ...
@overload
def unique(
    ar: ndarray,
    return_index: Literal[False] = False,
    return_inverse: Literal[False] = False,
    return_counts: Literal[False] = False,
    axis: int | None = None,
    *,
    equal_nan: bool = True,
) -> ndarray: ...
@overload
def unique(
    ar: ndarray,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: int | None = None,
    *,
    equal_nan: bool = True,
) -> ndarray | tuple[ndarray, ...]: ...
