# SPDX-License-Identifier: MPL-2.0
from typing import Literal, overload

import numpy as np
from numpy.typing import DTypeLike
from typing_extensions import Self

from cupy import ndarray
from cupy._core.core import _Integer, _NDArray, _Slice

from ._base import _spbase, _ToSparse

# `list`, not `Sequence`: a `tuple` means one index per axis
type _NonIntIndex = _Slice | list[int] | list[bool] | ndarray | _NDArray[_Integer | np.bool]
type _Index = int | _NonIntIndex

class _compressed_sparse_matrix(_spbase):
    data: ndarray[tuple[int]]
    indices: ndarray[tuple[int]]
    indptr: ndarray[tuple[int]]

    def __init__(
        self,
        arg1: _ToSparse | tuple[ndarray, ndarray, ndarray],
        shape: tuple[int, int] | None = None,
        dtype: DTypeLike | None = None,
        copy: bool = False,
        *,
        maxprint: int | None = None,
    ) -> None: ...
    @property
    def format(self) -> Literal["csr", "csc"]: ...
    @overload
    def __getitem__(self, key: tuple[int, int]) -> ndarray[tuple[()]]: ...
    @overload
    def __getitem__(self, key: _Index | tuple[_NonIntIndex, _Index] | tuple[int, _NonIntIndex]) -> Self: ...
    def __setitem__(self, key: _Index | tuple[_Index, _Index], x: object) -> None: ...

    # methods
    def astype(self, dtype: DTypeLike | None, copy: bool = True) -> Self: ...
    def power(self, n: int, dtype: DTypeLike | None = None) -> Self: ...
    @overload
    def sum(self, axis: None = None, dtype: DTypeLike | None = None, out: None = None) -> ndarray[tuple[()]]: ...
    @overload
    def sum(
        self, axis: Literal[0, 1], dtype: DTypeLike | None = None, out: ndarray[tuple[int, int]] | None = None
    ) -> ndarray[tuple[int, int]]: ...  # keeps dimensions
