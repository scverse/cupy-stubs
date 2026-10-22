# SPDX-License-Identifier: MPL-2.0
from typing import Literal

import scipy.sparse as sps
from typing_extensions import override

import cupy.cuda

from ._base import _SpScalar, spmatrix
from ._compressed import _compressed_sparse_matrix

class csr_matrix(spmatrix, _compressed_sparse_matrix):
    @property
    @override
    def format(self) -> Literal["csr"]: ...
    @override
    def get(self, stream: cupy.cuda.Stream | None = None) -> sps.csr_matrix[_SpScalar]: ...
