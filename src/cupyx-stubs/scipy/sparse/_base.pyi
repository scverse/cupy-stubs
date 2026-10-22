# SPDX-License-Identifier: MPL-2.0
from typing import Literal, overload

import numpy as np
import scipy.sparse as sps
from typing_extensions import Self

import cupy.cuda
from cupy import ndarray
from cupy._core.core import _DType

from ._csc import csc_matrix
from ._csr import csr_matrix

# scalar types cupy’s sparse matrices support
type _SpScalar = np.bool | np.float32 | np.float64 | np.complex64 | np.complex128

# constructor inputs all sparse formats accept
type _ToSparse = (
    ndarray
    | _spbase
    | sps.spmatrix[_SpScalar]
    | sps.sparray[_SpScalar, tuple[int, int]]
    | tuple[int, int]
    | tuple[ndarray, tuple[ndarray, ndarray]]
)

class _spbase:
    dtype: _DType
    shape: tuple[int, int]
    ndim: int
    def toarray(self, order: Literal["C", "F"] | None = None, out: None = None) -> ndarray[tuple[int, int]]: ...
    def copy(self) -> Self: ...
    def tocsr(self, copy: bool = False) -> csr_matrix: ...
    def tocsc(self, copy: bool = False) -> csc_matrix: ...
    def __matmul__(self, other: _spbase) -> csr_matrix: ...  # sparse @ sparse is always CSR
    # Doesn’t exist at runtime, but makes this `ArrayLike`,
    # so e.g. `np.sum(mat)` type checks, which works because `numpy` falls back to calling `mat.sum()`.
    def __array__(self) -> np.ndarray[tuple[int, int], _DType]: ...
    @overload
    def __pow__(self, other: Literal[1]) -> Self: ...
    @overload
    def __pow__(self, other: int) -> csr_matrix | Self: ...  # matrix power via `@`, so CSR unless `other == 1`
    def get(self, stream: cupy.cuda.Stream | None = None) -> sps.spmatrix[_SpScalar]: ...

# at runtime a mixin without base class, but all its concrete subclasses are `_spbase`s
class spmatrix(_spbase): ...
