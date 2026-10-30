# SPDX-License-Identifier: MPL-2.0
from collections.abc import Iterable
from typing import overload

from cupy._core import ndarray
from cupy._core.core import _DType, _Shape

@overload
def broadcast_to[DTypeT: _DType](array: ndarray[_Shape, DTypeT], shape: int) -> ndarray[tuple[int], DTypeT]: ...
@overload
def broadcast_to[ShapeT: _Shape, DTypeT: _DType](
    array: ndarray[_Shape, DTypeT], shape: ShapeT
) -> ndarray[ShapeT, DTypeT]: ...
@overload
def broadcast_to[DTypeT: _DType](array: ndarray[_Shape, DTypeT], shape: Iterable[int]) -> ndarray[_Shape, DTypeT]: ...
