# SPDX-License-Identifier: MPL-2.0
from typing import overload

import numpy as np
from numpy.typing import DTypeLike

from cupy._core import ndarray
from cupy._core.core import _DType, _Scalar, _Shape, _ToDType

@overload
def cumsum[ScalarT: _Scalar](
    a: ndarray, axis: int | None = None, *, dtype: _ToDType[ScalarT], out: None = None
) -> ndarray[_Shape, np.dtype[ScalarT]]: ...
@overload
def cumsum[ScalarT: _Scalar](
    a: ndarray, axis: int | None, dtype: _ToDType[ScalarT], out: None = None
) -> ndarray[_Shape, np.dtype[ScalarT]]: ...
@overload
def cumsum(
    a: ndarray, axis: int | None = None, dtype: DTypeLike | None = None, out: ndarray | None = None
) -> ndarray: ...
@overload
def diff[DTypeT: _DType](
    a: ndarray[_Shape, DTypeT],
    n: int = 1,
    axis: int = -1,
    prepend: None = None,
    append: None = None,
) -> ndarray[_Shape, DTypeT]: ...
@overload
def diff(
    a: ndarray,
    n: int = 1,
    axis: int = -1,
    prepend: ndarray | complex | None = None,
    append: ndarray | complex | None = None,
) -> ndarray: ...
