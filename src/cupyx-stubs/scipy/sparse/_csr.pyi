# SPDX-License-Identifier: MPL-2.0
from typing import Literal, overload

import numpy as np
import scipy.sparse as sps
from numpy.typing import DTypeLike
from typing_extensions import Self, TypeIs, override

import cupy.cuda
from cupy import ndarray

from ._base import (
    _1D,
    _2D,
    _Index,
    _NonIntIndex,
    _SpScalar,
    _SpShape,
    _ToDType,
    _ToSparse,
    _ToSparse1D,
    _ToSparse1DOf,
    _ToSparseOf,
    sparray,
    spmatrix,
)
from ._compressed import _compressed_sparse_matrix

class csr_matrix[ScalarT: _SpScalar = _SpScalar](spmatrix[ScalarT], _compressed_sparse_matrix[ScalarT, _2D]):
    @property
    @override
    def format(self) -> Literal["csr"]: ...
    @override
    def get(self, stream: cupy.cuda.Stream | None = None) -> sps.csr_matrix[ScalarT]: ...
    @overload
    def astype[NewScalarT: _SpScalar](
        self, dtype: _ToDType[NewScalarT], copy: bool = True
    ) -> csr_matrix[NewScalarT]: ...
    @overload
    def astype(self, dtype: DTypeLike | None, copy: bool = True) -> csr_matrix: ...
    @overload
    def power(self, n: int, dtype: None = None) -> Self: ...
    @overload
    def power[NewScalarT: _SpScalar](self, n: int, dtype: _ToDType[NewScalarT]) -> csr_matrix[NewScalarT]: ...
    @overload
    def power(self, n: int, dtype: DTypeLike) -> csr_matrix: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> ndarray[tuple[()], np.dtype[ScalarT]]: ...
    @overload
    def __getitem__(self, key: _Index | tuple[_NonIntIndex, _Index] | tuple[int, _NonIntIndex]) -> Self: ...
    def __setitem__(self, key: _Index | tuple[_Index, _Index], x: object) -> None: ...

class csr_array[ScalarT: _SpScalar = _SpScalar, ShapeT: _SpShape = _2D](
    sparray[ScalarT, ShapeT], _compressed_sparse_matrix[ScalarT, ShapeT]
):
    # 1D overloads need `self: csr_array[…]` here, which doesn’t work on the base class
    @overload  # 1D input, before the others, which would make it 2D
    def __init__(
        self: csr_array[ScalarT, _1D],
        arg1: _ToSparse1DOf[ScalarT],
        shape: _1D | None = None,
        dtype: None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @overload
    def __init__(
        self: csr_array[_SpScalar, _1D],
        arg1: _ToSparse1D,
        shape: _1D | None = None,
        dtype: DTypeLike | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
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
    @override
    def format(self) -> Literal["csr"]: ...
    @override
    def get(self, stream: cupy.cuda.Stream | None = None) -> sps.csr_array[ScalarT, ShapeT]: ...
    @overload
    def astype[NewScalarT: _SpScalar](
        self, dtype: _ToDType[NewScalarT], copy: bool = True
    ) -> csr_array[NewScalarT, ShapeT]: ...
    @overload
    def astype(self, dtype: DTypeLike | None, copy: bool = True) -> csr_array[_SpScalar, ShapeT]: ...
    @overload
    def power(self, n: int, dtype: None = None) -> Self: ...
    @overload
    def power[NewScalarT: _SpScalar](self, n: int, dtype: _ToDType[NewScalarT]) -> csr_array[NewScalarT, ShapeT]: ...
    @overload
    def power(self, n: int, dtype: DTypeLike) -> csr_array[_SpScalar, ShapeT]: ...

def isspmatrix_csr(x: object) -> TypeIs[csr_matrix]: ...
