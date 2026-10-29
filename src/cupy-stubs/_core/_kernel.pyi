# SPDX-License-Identifier: MPL-2.0
from typing import Literal

from numpy.typing import DTypeLike

from .core import _Scalar, ndarray

class ufunc:
    @property
    def nin(self) -> int: ...
    @property
    def nout(self) -> int: ...

class _UFunc_Nin1_Nout1(ufunc):
    def __call__(
        self,
        x: ndarray | _Scalar | complex,
        /,
        out: ndarray | None = None,
        *,
        dtype: DTypeLike | None = None,
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
    ) -> ndarray: ...
