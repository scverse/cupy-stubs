# SPDX-License-Identifier: MPL-2.0
from collections.abc import Sequence
from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from cupy._core import ndarray
from cupy._core.core import _Scalar, _Shape, _ToDType

@overload
def full[ScalarT: _Scalar](
    shape: int,
    fill_value: ArrayLike | ndarray,
    dtype: _ToDType[ScalarT],
    order: Literal["C", "F"] = "C",
) -> ndarray[tuple[int], np.dtype[ScalarT]]: ...
@overload
def full[ShapeT: _Shape, ScalarT: _Scalar](
    shape: ShapeT,
    fill_value: ArrayLike | ndarray,
    dtype: _ToDType[ScalarT],
    order: Literal["C", "F"] = "C",
) -> ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload
def full[ScalarT: _Scalar](
    shape: Sequence[int],
    fill_value: ArrayLike | ndarray,
    dtype: _ToDType[ScalarT],
    order: Literal["C", "F"] = "C",
) -> ndarray[_Shape, np.dtype[ScalarT]]: ...
@overload
def full(
    shape: int,
    fill_value: ArrayLike | ndarray,
    dtype: DTypeLike | None = None,
    order: Literal["C", "F"] = "C",
) -> ndarray[tuple[int]]: ...
@overload
def full[ShapeT: _Shape](
    shape: ShapeT,
    fill_value: ArrayLike | ndarray,
    dtype: DTypeLike | None = None,
    order: Literal["C", "F"] = "C",
) -> ndarray[ShapeT]: ...
@overload
def full(
    shape: Sequence[int],
    fill_value: ArrayLike | ndarray,
    dtype: DTypeLike | None = None,
    order: Literal["C", "F"] = "C",
) -> ndarray[_Shape]: ...
@overload
def ones[ScalarT: _Scalar](
    shape: int,
    dtype: _ToDType[ScalarT],
    order: Literal["C", "F"] = "C",
) -> ndarray[tuple[int], np.dtype[ScalarT]]: ...
@overload
def ones[ShapeT: _Shape, ScalarT: _Scalar](
    shape: ShapeT,
    dtype: _ToDType[ScalarT],
    order: Literal["C", "F"] = "C",
) -> ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload
def ones[ScalarT: _Scalar](
    shape: Sequence[int],
    dtype: _ToDType[ScalarT],
    order: Literal["C", "F"] = "C",
) -> ndarray[_Shape, np.dtype[ScalarT]]: ...
@overload
def ones(
    shape: int,
    dtype: DTypeLike | None = float,
    order: Literal["C", "F"] = "C",
) -> ndarray[tuple[int]]: ...
@overload
def ones[ShapeT: _Shape](
    shape: ShapeT,
    dtype: DTypeLike | None = float,
    order: Literal["C", "F"] = "C",
) -> ndarray[ShapeT]: ...
@overload
def ones(
    shape: Sequence[int],
    dtype: DTypeLike | None = float,
    order: Literal["C", "F"] = "C",
) -> ndarray[_Shape]: ...
@overload
def zeros[ScalarT: _Scalar](
    shape: int,
    dtype: _ToDType[ScalarT],
    order: Literal["C", "F"] = "C",
) -> ndarray[tuple[int], np.dtype[ScalarT]]: ...
@overload
def zeros[ShapeT: _Shape, ScalarT: _Scalar](
    shape: ShapeT,
    dtype: _ToDType[ScalarT],
    order: Literal["C", "F"] = "C",
) -> ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload
def zeros[ScalarT: _Scalar](
    shape: Sequence[int],
    dtype: _ToDType[ScalarT],
    order: Literal["C", "F"] = "C",
) -> ndarray[_Shape, np.dtype[ScalarT]]: ...
@overload
def zeros(
    shape: int,
    dtype: DTypeLike | None = float,
    order: Literal["C", "F"] = "C",
) -> ndarray[tuple[int]]: ...
@overload
def zeros[ShapeT: _Shape](
    shape: ShapeT,
    dtype: DTypeLike | None = float,
    order: Literal["C", "F"] = "C",
) -> ndarray[ShapeT]: ...
@overload
def zeros(
    shape: Sequence[int],
    dtype: DTypeLike | None = float,
    order: Literal["C", "F"] = "C",
) -> ndarray[_Shape]: ...
@overload
def empty[ScalarT: _Scalar](
    shape: int,
    dtype: _ToDType[ScalarT],
    order: Literal["C", "F"] = "C",
) -> ndarray[tuple[int], np.dtype[ScalarT]]: ...
@overload
def empty[ShapeT: _Shape, ScalarT: _Scalar](
    shape: ShapeT,
    dtype: _ToDType[ScalarT],
    order: Literal["C", "F"] = "C",
) -> ndarray[ShapeT, np.dtype[ScalarT]]: ...
@overload
def empty[ScalarT: _Scalar](
    shape: Sequence[int],
    dtype: _ToDType[ScalarT],
    order: Literal["C", "F"] = "C",
) -> ndarray[_Shape, np.dtype[ScalarT]]: ...
@overload
def empty(
    shape: int,
    dtype: DTypeLike | None = float,
    order: Literal["C", "F"] = "C",
) -> ndarray[tuple[int]]: ...
@overload
def empty[ShapeT: _Shape](
    shape: ShapeT,
    dtype: DTypeLike | None = float,
    order: Literal["C", "F"] = "C",
) -> ndarray[ShapeT]: ...
@overload
def empty(
    shape: Sequence[int],
    dtype: DTypeLike | None = float,
    order: Literal["C", "F"] = "C",
) -> ndarray[_Shape]: ...
