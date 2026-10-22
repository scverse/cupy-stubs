# SPDX-License-Identifier: MPL-2.0
from typing import Literal, overload

import scipy.sparse as sps
from numpy.typing import DTypeLike
from typing_extensions import override

import cupy.cuda

from ._base import _SpScalar, _ToDType, _ToSparse, _ToSparseOf, spmatrix

class coo_matrix[ScalarT: _SpScalar = _SpScalar](spmatrix[ScalarT]):
    @overload  # keyword `dtype` determines the scalar type
    def __init__(
        self,
        arg1: _ToSparse,
        shape: tuple[int, int] | None = None,
        *,
        dtype: _ToDType[ScalarT],
        copy: bool = False,
        maxprint: int = 50,
    ) -> None: ...
    @overload  # without `dtype`, `arg1` determines the scalar type
    def __init__(
        self,
        arg1: _ToSparseOf[ScalarT],
        shape: tuple[int, int] | None = None,
        dtype: None = None,
        copy: bool = False,
        *,
        maxprint: int = 50,
    ) -> None: ...
    @overload
    def __init__(
        self,
        arg1: _ToSparse,
        shape: tuple[int, int] | None = None,
        dtype: DTypeLike | None = None,
        copy: bool = False,
        *,
        maxprint: int = 50,
    ) -> None: ...
    @property
    def format(self) -> Literal["coo"]: ...
    @override
    def get(self, stream: cupy.cuda.Stream | None = None) -> sps.coo_matrix[ScalarT]: ...
