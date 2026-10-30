# SPDX-License-Identifier: MPL-2.0
from collections.abc import Sequence

from cupy._core import ndarray
from cupy._core.core import _DType, _Shape

def repeat[DTypeT: _DType](
    a: ndarray[_Shape, DTypeT],
    repeats: int | Sequence[int] | ndarray[tuple[()] | tuple[int]],
    axis: int | None = None,
) -> ndarray[_Shape, DTypeT]: ...
