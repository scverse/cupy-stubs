# SPDX-License-Identifier: MPL-2.0
from typing import Literal

import scipy.sparse as sps
from numpy.typing import DTypeLike
from typing_extensions import override

import cupy.cuda

from ._base import _SpScalar, _ToSparse, spmatrix

class coo_matrix(spmatrix):
    def __init__(
        self,
        arg1: _ToSparse,
        shape: tuple[int, int] | None = None,
        dtype: DTypeLike | None = None,
        copy: bool = False,
        *,
        maxprint: int = 50,
    ) -> None: ...
    @property
    def format(self) -> Literal["coo"]: ...
    @override
    def get(self, stream: cupy.cuda.Stream | None = None) -> sps.coo_matrix[_SpScalar]: ...
