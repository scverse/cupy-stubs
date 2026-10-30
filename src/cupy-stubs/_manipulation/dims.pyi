# SPDX-License-Identifier: MPL-2.0
from collections.abc import Iterable

from cupy._core import ndarray

def broadcast_to(array: ndarray, shape: int | Iterable[int]) -> ndarray: ...
