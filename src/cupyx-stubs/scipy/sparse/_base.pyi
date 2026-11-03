# SPDX-License-Identifier: MPL-2.0
from typing import Literal, overload

import numpy as np
import scipy.sparse as sps
from numpy.typing import DTypeLike
from typing_extensions import Self, TypeIs

import cupy.cuda
from cupy import ndarray
from cupy._core.core import _Integer, _NDArray, _Slice
from cupy._core.core import _ToDType as _ToDType

from ._coo import coo_array
from ._csc import csc_array, csc_matrix
from ._csr import csr_array, csr_matrix

# scalar types cupy’s sparse matrices support
type _SpScalar = np.bool | np.float32 | np.float64 | np.complex64 | np.complex128
# all formats are 2D, `coo_array` and `csr_array` can also be 1D
type _SpShape = tuple[int] | tuple[int, int]
type _1D = tuple[int]
type _2D = tuple[int, int]

# constructor inputs all sparse formats accept, `ShapeT` is bound by the input or the `shape` argument
# (dense input of unknown shape is 2D for all formats, 1D dense input is in `_ToSparse1D`)
type _ToSparse[ShapeT: _SpShape] = (
    ndarray
    | _spbase[_SpScalar, ShapeT]
    | sps.spmatrix[_SpScalar]
    | sps.sparray[_SpScalar, ShapeT]
    | ShapeT
    | tuple[ndarray, tuple[ndarray, ndarray]]
)
# constructor inputs all sparse formats accept, which determine the scalar type
type _ToSparseOf[ScalarT: _SpScalar, ShapeT: _SpShape] = (
    ndarray[_SpShape, np.dtype[ScalarT]]  # matrices make 1D input a row
    | _spbase[ScalarT, ShapeT]
    | sps.spmatrix[ScalarT]
    | sps.sparray[ScalarT, ShapeT]
    | tuple[ndarray[_1D, np.dtype[ScalarT]], tuple[ndarray, ndarray]]
)
# 1D input, which makes `coo_array` and `csr_array` 1D (matrices make it a row)
type _ToSparse1D = ndarray[_1D] | tuple[ndarray, tuple[ndarray]]
type _ToSparse1DOf[ScalarT: _SpScalar] = (
    ndarray[_1D, np.dtype[ScalarT]] | tuple[ndarray[_1D, np.dtype[ScalarT]], tuple[ndarray]]
)

# `list`, not `Sequence`: a `tuple` means one index per axis
type _NonIntIndex = _Slice | list[int] | list[bool] | ndarray | _NDArray[_Integer | np.bool]
type _Index = int | _NonIntIndex

class _spbase[ScalarT: _SpScalar = _SpScalar, ShapeT: _SpShape = _2D]:
    @property
    def shape(self) -> ShapeT: ...
    @property
    def ndim(self) -> int: ...
    @property
    def dtype(self) -> np.dtype[ScalarT]: ...  # property, so `ScalarT` stays covariant
    def toarray(
        self, order: Literal["C", "F"] | None = None, out: None = None
    ) -> ndarray[ShapeT, np.dtype[ScalarT]]: ...
    def copy(self) -> Self: ...
    # Doesn’t exist at runtime, but makes this `ArrayLike`,
    # so e.g. `np.sum(mat)` type checks, which works because `numpy` falls back to calling `mat.sum()`.
    def __array__(self) -> np.ndarray[ShapeT, np.dtype[ScalarT]]: ...

# The mixins are bare classes at runtime; all their concrete subclasses are `_spbase`s.
# Methods whose result is a matrix for matrices and an array for arrays live here.

class spmatrix[ScalarT: _SpScalar = _SpScalar](_spbase[ScalarT, _2D]):
    def tocsr(self, copy: bool = False) -> csr_matrix[ScalarT]: ...
    def tocsc(self, copy: bool = False) -> csc_matrix[ScalarT]: ...
    # TODO: overloads to infer the result scalar type (promotion of both operands’ scalar types)
    def __matmul__(self, other: _spbase) -> csr_matrix: ...  # sparse @ sparse is always CSR
    @overload
    def __pow__(self, other: Literal[1]) -> Self: ...
    @overload
    def __pow__(self, other: int) -> csr_matrix[ScalarT] | Self: ...  # matrix power via `@`, so CSR unless `other == 1`
    def get(self, stream: cupy.cuda.Stream | None = None) -> sps.spmatrix[ScalarT]: ...
    @overload
    def sum(self, axis: None = None, dtype: DTypeLike | None = None, out: None = None) -> ndarray[tuple[()]]: ...
    @overload
    def sum(
        self, axis: Literal[0, 1], dtype: DTypeLike | None = None, out: ndarray[_2D] | None = None
    ) -> ndarray[_2D]: ...  # keeps dimensions

class sparray[ScalarT: _SpScalar = _SpScalar, ShapeT: _SpShape = _2D](_spbase[ScalarT, ShapeT]):
    def tocsr(self, copy: bool = False) -> csr_array[ScalarT, ShapeT]: ...
    def tocsc(self: sparray[ScalarT, _2D], copy: bool = False) -> csc_array[ScalarT]: ...
    # TODO: overloads to infer the result scalar type (promotion of both operands’ scalar types)
    @overload  # sparse @ sparse is CSR
    def __matmul__(self: sparray[_SpScalar, _2D], other: _spbase[_SpScalar, _2D]) -> csr_array: ...
    @overload  # numpy semantics: a 1D operand drops a dimension, the result is COO
    def __matmul__(self: sparray[_SpScalar, _2D], other: sparray[_SpScalar, _1D]) -> coo_array[_SpScalar, _1D]: ...
    @overload
    def __matmul__(self: sparray[_SpScalar, _1D], other: _spbase[_SpScalar, _2D]) -> coo_array[_SpScalar, _1D]: ...
    @overload
    def __matmul__(self: sparray[_SpScalar, _1D], other: sparray[_SpScalar, _1D]) -> ndarray[tuple[()]]: ...  # dot
    def __pow__(self, other: int) -> Self: ...  # element-wise
    def get(self, stream: cupy.cuda.Stream | None = None) -> sps.sparray[ScalarT, ShapeT]: ...
    @overload
    def sum(self, axis: None = None, dtype: DTypeLike | None = None, out: None = None) -> ndarray[tuple[()]]: ...
    @overload
    def sum(
        self: sparray[_SpScalar, _1D], axis: Literal[0, -1], dtype: DTypeLike | None = None, out: None = None
    ) -> ndarray[tuple[()]]: ...
    @overload
    def sum(
        self: sparray[_SpScalar, _2D],
        axis: Literal[0, 1],
        dtype: DTypeLike | None = None,
        out: ndarray[_1D] | None = None,
    ) -> ndarray[_1D]: ...  # drops the reduced dimension
    # numpy semantics: each integer index drops a dimension
    @overload
    def __getitem__(self: sparray[ScalarT, _1D], key: int) -> ndarray[tuple[()], np.dtype[ScalarT]]: ...
    @overload
    def __getitem__[A: sparray[_SpScalar, _1D]](self: A, key: _NonIntIndex) -> A: ...
    @overload
    def __getitem__(self: sparray[ScalarT, _2D], key: tuple[int, int]) -> ndarray[tuple[()], np.dtype[ScalarT]]: ...
    @overload  # a row or column, in every format, as only COO and CSR can be 1D
    def __getitem__(
        self: sparray[ScalarT, _2D], key: int | tuple[int, _NonIntIndex] | tuple[_NonIntIndex, int]
    ) -> coo_array[ScalarT, _1D]: ...
    @overload
    def __getitem__[A: sparray[_SpScalar, _2D]](
        self: A, key: _NonIntIndex | tuple[_NonIntIndex, _NonIntIndex]
    ) -> A: ...
    def __setitem__(self, key: _Index | tuple[_Index, _Index], x: object) -> None: ...

def issparse(x: object) -> TypeIs[spmatrix | sparray]: ...
def isspmatrix(x: object) -> TypeIs[spmatrix]: ...
