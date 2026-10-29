# SPDX-License-Identifier: MPL-2.0
from collections.abc import Sequence
from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from cupy._core import ndarray
from cupy._core.core import _Scalar, _Shape

# `dtype` arguments that determine the scalar type
type _ToDType[ScalarT: _Scalar] = type[ScalarT] | np.dtype[ScalarT]

@overload
def full[ScalarT: _Scalar](
    shape: int | Sequence[int],
    fill_value: ArrayLike | ndarray,
    dtype: _ToDType[ScalarT],
    order: Literal["C", "F"] = "C",
) -> ndarray[_Shape, np.dtype[ScalarT]]: ...
@overload
def full(
    shape: int | Sequence[int],
    fill_value: ArrayLike | ndarray,
    dtype: DTypeLike | None = None,
    order: Literal["C", "F"] = "C",
) -> ndarray: ...
@overload
def ones[ScalarT: _Scalar](
    shape: int | Sequence[int],
    dtype: _ToDType[ScalarT],
    order: Literal["C", "F"] = "C",
) -> ndarray[_Shape, np.dtype[ScalarT]]: ...
@overload
def ones(
    shape: int | Sequence[int],
    dtype: DTypeLike | None = float,
    order: Literal["C", "F"] = "C",
) -> ndarray: ...
@overload
def zeros[ScalarT: _Scalar](
    shape: int | Sequence[int],
    dtype: _ToDType[ScalarT],
    order: Literal["C", "F"] = "C",
) -> ndarray[_Shape, np.dtype[ScalarT]]: ...
@overload
def zeros(
    shape: int | Sequence[int],
    dtype: DTypeLike | None = float,
    order: Literal["C", "F"] = "C",
) -> ndarray: ...
@overload
def empty[ScalarT: _Scalar](
    shape: int | Sequence[int],
    dtype: _ToDType[ScalarT],
    order: Literal["C", "F"] = "C",
) -> ndarray[_Shape, np.dtype[ScalarT]]: ...
@overload
def empty(
    shape: int | Sequence[int],
    dtype: DTypeLike | None = float,
    order: Literal["C", "F"] = "C",
) -> ndarray: ...
