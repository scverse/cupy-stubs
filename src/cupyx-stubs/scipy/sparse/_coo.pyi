# SPDX-License-Identifier: MPL-2.0
from typing import Literal

import scipy.sparse as sps
from numpy.typing import DTypeLike

import cupy.cuda

from ._base import _ToSparse, spmatrix

class coo_matrix(spmatrix):
    format: Literal["coo"] = "coo"
    def __init__(
        self,
        arg1: _ToSparse,
        shape: tuple[int, int] | None = None,
        dtype: DTypeLike | None = None,
        copy: bool = False,
    ) -> None: ...
    def get(self, stream: cupy.cuda.Stream | None = None) -> sps.coo_matrix: ...
