# SPDX-License-Identifier: MPL-2.0
from typing import Literal

from numpy.typing import ArrayLike, DTypeLike

from .core import ndarray

class ufunc:
    @property
    def nin(self) -> int: ...
    @property
    def nout(self) -> int: ...

# like `numpy`’s stubs, one class per number of inputs and outputs
class _UFunc_Nin1_Nout1(ufunc):
    def __call__(
        self,
        x: ndarray | ArrayLike,
        /,
        out: ndarray | None = None,
        *,
        dtype: DTypeLike | None = None,
        casting: Literal["no", "equiv", "safe", "same_kind", "unsafe"] = "same_kind",
    ) -> ndarray: ...
