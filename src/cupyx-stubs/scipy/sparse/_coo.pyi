# SPDX-License-Identifier: MPL-2.0
from typing import Literal, overload

import scipy.sparse as sps
from numpy.typing import DTypeLike
from typing_extensions import TypeIs, override

import cupy.cuda

from ._base import (
    _1D,
    _2D,
    _spbase,
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

class _coo_base[ScalarT: _SpScalar = _SpScalar, ShapeT: _SpShape = _2D](_spbase[ScalarT, ShapeT]):
    @overload  # keyword `dtype` determines the scalar type
    def __init__(
        self,
        arg1: _ToSparse[ShapeT],
        shape: ShapeT | None = None,
        *,
        dtype: _ToDType[ScalarT],
        copy: bool = False,
        maxprint: int = 50,
    ) -> None: ...
    @overload  # without `dtype`, `arg1` determines the scalar type
    def __init__(
        self,
        arg1: _ToSparseOf[ScalarT, ShapeT],
        shape: ShapeT | None = None,
        dtype: None = None,
        copy: bool = False,
        *,
        maxprint: int = 50,
    ) -> None: ...
    @overload
    def __init__(
        self,
        arg1: _ToSparse[ShapeT],
        shape: ShapeT | None = None,
        dtype: DTypeLike | None = None,
        copy: bool = False,
        *,
        maxprint: int = 50,
    ) -> None: ...
    @property
    def format(self) -> Literal["coo"]: ...

class coo_matrix[ScalarT: _SpScalar = _SpScalar](spmatrix[ScalarT], _coo_base[ScalarT, _2D]):
    @override
    def get(self, stream: cupy.cuda.Stream | None = None) -> sps.coo_matrix[ScalarT]: ...

class coo_array[ScalarT: _SpScalar = _SpScalar, ShapeT: _SpShape = _2D](
    sparray[ScalarT, ShapeT], _coo_base[ScalarT, ShapeT]
):
    # 1D overloads need `self: coo_array[…]` here, which doesn’t work on the base class
    @overload  # 1D input, before the others, which would make it 2D
    def __init__(
        self: coo_array[ScalarT, _1D],
        arg1: _ToSparse1DOf[ScalarT],
        shape: _1D | None = None,
        dtype: None = None,
        copy: bool = False,
        *,
        maxprint: int = 50,
    ) -> None: ...
    @overload
    def __init__(
        self: coo_array[_SpScalar, _1D],
        arg1: _ToSparse1D,
        shape: _1D | None = None,
        dtype: DTypeLike | None = None,
        copy: bool = False,
        *,
        maxprint: int = 50,
    ) -> None: ...
    @overload  # keyword `dtype` determines the scalar type
    def __init__(
        self,
        arg1: _ToSparse[ShapeT],
        shape: ShapeT | None = None,
        *,
        dtype: _ToDType[ScalarT],
        copy: bool = False,
        maxprint: int = 50,
    ) -> None: ...
    @overload  # without `dtype`, `arg1` determines the scalar type
    def __init__(
        self,
        arg1: _ToSparseOf[ScalarT, ShapeT],
        shape: ShapeT | None = None,
        dtype: None = None,
        copy: bool = False,
        *,
        maxprint: int = 50,
    ) -> None: ...
    @overload
    def __init__(
        self,
        arg1: _ToSparse[ShapeT],
        shape: ShapeT | None = None,
        dtype: DTypeLike | None = None,
        copy: bool = False,
        *,
        maxprint: int = 50,
    ) -> None: ...
    @override
    def get(self, stream: cupy.cuda.Stream | None = None) -> sps.coo_array[ScalarT, ShapeT]: ...

def isspmatrix_coo(x: object) -> TypeIs[coo_matrix]: ...
