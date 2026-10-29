# SPDX-License-Identifier: MPL-2.0
from numpy.typing import DTypeLike

from cupy._core import ndarray

def cumsum(a: ndarray, axis: int | None = None, dtype: DTypeLike | None = None, out: ndarray | None = None) -> ndarray: ...
def diff(
    a: ndarray,
    n: int = 1,
    axis: int = -1,
    prepend: ndarray | complex | None = None,
    append: ndarray | complex | None = None,
) -> ndarray: ...
