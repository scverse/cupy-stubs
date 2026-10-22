# SPDX-License-Identifier: MPL-2.0
from collections.abc import Sequence

from cupy._core import ndarray

def broadcast_to(array: ndarray, shape: int | Sequence[int]) -> ndarray: ...
