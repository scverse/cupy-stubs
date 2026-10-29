# SPDX-License-Identifier: MPL-2.0
from typing import Literal, overload

from numpy import bool_ as bool_
from numpy import complex64 as complex64
from numpy import complex128 as complex128
from numpy import float16 as float16
from numpy import float32 as float32
from numpy import float64 as float64
from numpy import int8 as int8
from numpy import int16 as int16
from numpy import int32 as int32
from numpy import int64 as int64
from numpy import nan as nan
from numpy import ndarray as _np_ndarray
from numpy import result_type as result_type
from numpy import uint8 as uint8
from numpy import uint16 as uint16
from numpy import uint32 as uint32
from numpy import uint64 as uint64
from numpy.typing import ArrayLike

from . import random as random
from ._core import ndarray as ndarray
from ._core._kernel import ufunc as ufunc
from ._core.core import _DType, _Generic, _NDArray, _Shape
from ._creation.basic import empty as empty
from ._creation.basic import full as full
from ._creation.basic import ones as ones
from ._creation.basic import zeros as zeros
from ._creation.from_data import array as array
from ._creation.from_data import asarray as asarray
from ._creation.from_data import ascontiguousarray as ascontiguousarray
from ._creation.ranges import arange as arange
from ._logic.comparison import array_equal as array_equal
from ._manipulation.add_remove import append as append
from ._manipulation.add_remove import unique as unique
from ._manipulation.dims import broadcast_to as broadcast_to
from ._manipulation.join import concatenate as concatenate
from ._manipulation.join import stack as stack
from ._manipulation.tiling import repeat as repeat
from ._math.explog import log2 as log2
from ._math.sumprod import cumsum as cumsum
from ._math.sumprod import diff as diff
from ._sorting.search import flatnonzero as flatnonzero
from ._sorting.search import nonzero as nonzero
from ._sorting.search import searchsorted as searchsorted
from ._sorting.search import where as where
from ._sorting.sort import argsort as argsort
from ._sorting.sort import lexsort as lexsort
from ._statistics.histogram import bincount as bincount
from .cuda import Stream

@overload
def asnumpy[ShapeT: _Shape, DTypeT: _DType](
    a: ndarray[ShapeT, DTypeT],
    stream: Stream | None = None,
    order: Literal["C", "F", "A"] = "C",
    out: _NDArray[_Generic] | None = None,
    *,
    blocking: bool = True,
) -> _np_ndarray[ShapeT, DTypeT]: ...
@overload
def asnumpy(
    a: ArrayLike,
    stream: Stream | None = None,
    order: Literal["C", "F", "A"] = "C",
    out: _NDArray[_Generic] | None = None,
    *,
    blocking: bool = True,
) -> _NDArray[_Generic]: ...
