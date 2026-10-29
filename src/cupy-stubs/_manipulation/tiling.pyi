# SPDX-License-Identifier: MPL-2.0
from collections.abc import Sequence

from cupy._core import ndarray

def repeat(a: ndarray, repeats: int | Sequence[int] | ndarray, axis: int | None = None) -> ndarray: ...
