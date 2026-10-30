# SPDX-License-Identifier: MPL-2.0
from collections.abc import Iterable
from typing import Literal

from numpy.typing import DTypeLike

from ._base import spmatrix

def hstack(
    blocks: Iterable[spmatrix],
    format: Literal["csr", "csc", "coo"] | None = None,
    dtype: DTypeLike | None = None,
) -> spmatrix: ...
def vstack(
    blocks: Iterable[spmatrix],
    format: Literal["csr", "csc", "coo"] | None = None,
    dtype: DTypeLike | None = None,
) -> spmatrix: ...
