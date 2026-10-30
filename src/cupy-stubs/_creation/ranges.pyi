# SPDX-License-Identifier: MPL-2.0
from typing import overload

import numpy as np
from numpy.typing import DTypeLike

from cupy._core import ndarray
from cupy._core.core import _Scalar, _ToDType

@overload
def arange[ScalarT: _Scalar](
    start: float,
    stop: float | None = None,
    step: float = 1,
    *,
    dtype: _ToDType[ScalarT],
) -> ndarray[tuple[int], np.dtype[ScalarT]]: ...
@overload
def arange[ScalarT: _Scalar](
    start: float,
    stop: float | None,
    step: float,
    dtype: _ToDType[ScalarT],
) -> ndarray[tuple[int], np.dtype[ScalarT]]: ...
@overload
def arange(
    start: float,
    stop: float | None = None,
    step: float = 1,
    dtype: DTypeLike | None = None,
) -> ndarray[tuple[int]]: ...
