# SPDX-License-Identifier: MPL-2.0
from collections.abc import Sequence

import numpy as np

from cupy._core import ndarray
from cupy._core.core import _Shape, _ToDType

class Generator:
    def standard_normal[ScalarT: np.float32 | np.float64 = np.float64](
        self,
        size: int | Sequence[int] | None = None,
        dtype: _ToDType[ScalarT] = ...,  # `np.float64`, see `ScalarT`’s default
        out: ndarray | None = None,
    ) -> ndarray[_Shape, np.dtype[ScalarT]]: ...
