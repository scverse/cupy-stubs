# SPDX-License-Identifier: MPL-2.0
from collections.abc import Sequence
from typing import overload

import numpy as np

from cupy._core import ndarray
from cupy._core.core import _Shape, _ToDType

class Generator:
    @overload
    def standard_normal(
        self,
        size: int | Sequence[int] | None = None,
        dtype: _ToDType[np.float64] = np.float64,
        out: ndarray | None = None,
    ) -> ndarray[_Shape, np.dtype[np.float64]]: ...
    @overload
    def standard_normal(
        self,
        size: int | Sequence[int] | None = None,
        dtype: _ToDType[np.float32] = ...,
        out: ndarray | None = None,
    ) -> ndarray[_Shape, np.dtype[np.float32]]: ...
