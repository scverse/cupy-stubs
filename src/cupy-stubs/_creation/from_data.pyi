# SPDX-License-Identifier: MPL-2.0
from typing import Literal, overload

import h5py
import numpy as np
import zarr
from numpy.typing import ArrayLike, DTypeLike
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata

from cupy._core import ndarray
from cupy._core.core import _DType, _Scalar, _Shape, _ToDType

def asarray(
    a: ArrayLike | h5py.Dataset | zarr.Array[ArrayV2Metadata] | zarr.Array[ArrayV3Metadata],
    dtype: DTypeLike | None = None,
    order: Literal["C", "F", "A", "K"] | None = None,
    *,
    copy: bool | None = None,
    blocking: bool = False,
) -> ndarray: ...
def array(
    obj: ArrayLike | ndarray,
    dtype: DTypeLike | None = None,
    copy: bool = True,
    order: Literal["C", "F", "A", "K"] = "K",
    subok: bool = False,
    ndmin: int = 0,
    *,
    blocking: bool = False,
) -> ndarray: ...
@overload
def ascontiguousarray[DTypeT: _DType](a: ndarray[_Shape, DTypeT], dtype: None = None) -> ndarray[_Shape, DTypeT]: ...
@overload
def ascontiguousarray[ScalarT: _Scalar](a: ndarray, dtype: _ToDType[ScalarT]) -> ndarray[_Shape, np.dtype[ScalarT]]: ...
@overload
def ascontiguousarray(a: ndarray, dtype: DTypeLike | None = None) -> ndarray: ...
