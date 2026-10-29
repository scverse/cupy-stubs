# SPDX-License-Identifier: MPL-2.0
from collections.abc import Sequence

import numpy as np

from cupy._core import ndarray
from cupy._core.core import _ToDType

class Generator:
    def standard_normal(
        self,
        size: int | Sequence[int] | None = None,
        dtype: _ToDType[np.float32 | np.float64] = np.float64,
        out: ndarray | None = None,
    ) -> ndarray: ...
