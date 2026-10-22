# SPDX-License-Identifier: MPL-2.0
from typing import Literal, overload

import scipy.sparse as sps
from numpy.typing import DTypeLike
from typing_extensions import Self, override

import cupy.cuda

from ._base import _SpScalar, _ToDType, spmatrix
from ._compressed import _compressed_sparse_matrix

class csc_matrix[ScalarT: _SpScalar = _SpScalar](spmatrix[ScalarT], _compressed_sparse_matrix[ScalarT]):
    @property
    @override
    def format(self) -> Literal["csc"]: ...
    @override
    def get(self, stream: cupy.cuda.Stream | None = None) -> sps.csc_matrix[ScalarT]: ...
    @overload
    def astype[NewScalarT: _SpScalar](
        self, dtype: _ToDType[NewScalarT], copy: bool = True
    ) -> csc_matrix[NewScalarT]: ...
    @overload
    def astype(self, dtype: DTypeLike | None, copy: bool = True) -> csc_matrix: ...
    @overload
    def power(self, n: int, dtype: None = None) -> Self: ...
    @overload
    def power[NewScalarT: _SpScalar](self, n: int, dtype: _ToDType[NewScalarT]) -> csc_matrix[NewScalarT]: ...
    @overload
    def power(self, n: int, dtype: DTypeLike) -> csc_matrix: ...
