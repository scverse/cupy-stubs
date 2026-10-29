# SPDX-License-Identifier: MPL-2.0
from numpy.typing import DTypeLike

from cupy._core import ndarray

def arange(
    start: float,
    stop: float | None = None,
    step: float = 1,
    dtype: DTypeLike | None = None,
) -> ndarray[tuple[int]]: ...
