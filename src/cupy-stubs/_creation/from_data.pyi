# SPDX-License-Identifier: MPL-2.0
from typing import Literal

import h5py
import zarr
from numpy.typing import ArrayLike, DTypeLike
from zarr.core.metadata import ArrayV2Metadata, ArrayV3Metadata

from cupy._core import ndarray

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
def ascontiguousarray(a: ndarray, dtype: DTypeLike | None = None) -> ndarray: ...
