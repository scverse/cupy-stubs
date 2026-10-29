# SPDX-License-Identifier: MPL-2.0
from collections.abc import Sequence

import numpy as np
from numpy.typing import DTypeLike

from cupy._core import ndarray

class Generator:
    def standard_normal(
        self, size: int | Sequence[int] | None = None, dtype: DTypeLike = np.float64, out: ndarray | None = None
    ) -> ndarray: ...
