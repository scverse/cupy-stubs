# SPDX-License-Identifier: MPL-2.0
from collections.abc import Iterable
from typing import Literal, overload

import numpy as np
from numpy.typing import DTypeLike

from cupy._core import ndarray
from cupy._core.core import _Scalar, _Shape, _ToDType

@overload
def concatenate[ScalarT: _Scalar](
    tup: Iterable[ndarray],
    axis: int | None = 0,
    out: None = None,
    *,
    dtype: _ToDType[ScalarT],
    casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
) -> ndarray[_Shape, np.dtype[ScalarT]]: ...
@overload
def concatenate(
    tup: Iterable[ndarray],
    axis: int | None = 0,
    out: ndarray | None = None,
    *,
    dtype: DTypeLike | None = None,
    casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
) -> ndarray: ...
@overload
def stack[ScalarT: _Scalar](
    tup: Iterable[ndarray],
    axis: int = 0,
    out: None = None,
    *,
    dtype: _ToDType[ScalarT],
    casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
) -> ndarray[_Shape, np.dtype[ScalarT]]: ...
@overload
def stack(
    tup: Iterable[ndarray],
    axis: int = 0,
    out: ndarray | None = None,
    *,
    dtype: DTypeLike | None = None,
    casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
) -> ndarray: ...
