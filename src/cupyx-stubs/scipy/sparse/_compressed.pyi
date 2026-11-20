# SPDX-License-Identifier: MPL-2.0
from typing import Literal, overload

import numpy as np
from numpy.typing import DTypeLike

from cupy import ndarray

from ._base import (
    _1D,
    _2D,
    _spbase,
    _SpScalar,
    _SpShape,
    _ToDType,
    _ToSparse,
    _ToSparseOf,
)

class _compressed_sparse_matrix[ScalarT: _SpScalar = _SpScalar, ShapeT: _SpShape = _2D](_spbase[ScalarT, ShapeT]):
    indices: ndarray[_1D, np.dtype[np.int32 | np.int64]]
    indptr: ndarray[_1D, np.dtype[np.int32 | np.int64]]

    @overload  # keyword `dtype` determines the scalar type
    def __init__(
        self,
        arg1: _ToSparse[ShapeT] | tuple[ndarray, ndarray, ndarray],
        shape: ShapeT | None = None,
        *,
        dtype: _ToDType[ScalarT],
        copy: bool = False,
        maxprint: int | None = None,
    ) -> None: ...
    @overload  # without `dtype`, `arg1` determines the scalar type
    def __init__(
        self,
        arg1: _ToSparseOf[ScalarT, ShapeT] | tuple[ndarray[_1D, np.dtype[ScalarT]], ndarray, ndarray],
        shape: ShapeT | None = None,
        dtype: None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        arg1: _ToSparse[ShapeT] | tuple[ndarray, ndarray, ndarray],
        shape: ShapeT | None = None,
        dtype: DTypeLike | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @property
    def data(self) -> ndarray[_1D, np.dtype[ScalarT]]: ...  # property, so `ScalarT` stays covariant
    @property
    def format(self) -> Literal["csr", "csc"]: ...
    # `__getitem__` and `__setitem__` are in subclasses, as they differ between matrices and arrays
    # `astype` and `power` are in subclasses, as `Self` can’t change its type parameter
