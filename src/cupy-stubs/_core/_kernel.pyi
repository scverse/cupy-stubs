# SPDX-License-Identifier: MPL-2.0
from typing import Literal, overload

import numpy as np
from numpy.typing import DTypeLike
from typing_extensions import disjoint_base

from .core import _Scalar, _Shape, _ToDType, ndarray

@disjoint_base
class ufunc:
    @property
    def nin(self) -> int: ...
    @property
    def nout(self) -> int: ...

class _UFunc_Nin1_Nout1(ufunc):
    @overload
    def __call__[ShapeT: _Shape, ScalarT: _Scalar](
        self,
        x: ndarray[ShapeT],
        /,
        out: None = None,
        *,
        dtype: _ToDType[ScalarT],
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
    ) -> ndarray[ShapeT, np.dtype[ScalarT]]: ...
    @overload
    def __call__[ScalarT: _Scalar](
        self,
        x: _Scalar | complex,
        /,
        out: None = None,
        *,
        dtype: _ToDType[ScalarT],
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
    ) -> ndarray[tuple[()], np.dtype[ScalarT]]: ...
    @overload
    def __call__[ShapeT: _Shape](
        self,
        x: ndarray[ShapeT],
        /,
        out: None = None,
        *,
        dtype: DTypeLike | None = None,
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
    ) -> ndarray[ShapeT]: ...
    @overload
    def __call__(
        self,
        x: _Scalar | complex,
        /,
        out: None = None,
        *,
        dtype: DTypeLike | None = None,
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
    ) -> ndarray[tuple[()]]: ...
    # `out` can have a bigger (broadcast) shape than `x`, and is returned as is
    @overload
    def __call__[OutT: ndarray](
        self,
        x: ndarray | _Scalar | complex,
        /,
        out: OutT,
        *,
        dtype: DTypeLike | None = None,
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
    ) -> OutT: ...

# the result shape is broadcast from both inputs, so it isn’t tracked
class _UFunc_Nin2_Nout1(ufunc):
    @overload
    def __call__[ScalarT: _Scalar](
        self,
        x1: ndarray | _Scalar | complex,
        x2: ndarray | _Scalar | complex,
        /,
        out: None = None,
        *,
        dtype: _ToDType[ScalarT],
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
    ) -> ndarray[_Shape, np.dtype[ScalarT]]: ...
    @overload
    def __call__(
        self,
        x1: ndarray | _Scalar | complex,
        x2: ndarray | _Scalar | complex,
        /,
        out: None = None,
        *,
        dtype: DTypeLike | None = None,
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
    ) -> ndarray: ...
    @overload
    def __call__[OutT: ndarray](
        self,
        x1: ndarray | _Scalar | complex,
        x2: ndarray | _Scalar | complex,
        /,
        out: OutT,
        *,
        dtype: DTypeLike | None = None,
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
    ) -> OutT: ...
