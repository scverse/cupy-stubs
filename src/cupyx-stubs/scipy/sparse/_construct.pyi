# SPDX-License-Identifier: MPL-2.0
from collections.abc import Iterable
from typing import Literal, overload

from numpy.typing import DTypeLike

from ._base import _spbase, sparray, spmatrix

# the result is an array if any block is one
@overload
def hstack(
    blocks: Iterable[spmatrix],
    format: Literal["csr", "csc", "coo"] | None = None,
    dtype: DTypeLike | None = None,
) -> spmatrix: ...
@overload
def hstack(
    blocks: Iterable[sparray],
    format: Literal["csr", "csc", "coo"] | None = None,
    dtype: DTypeLike | None = None,
) -> sparray: ...
@overload
def hstack(
    blocks: Iterable[_spbase],
    format: Literal["csr", "csc", "coo"] | None = None,
    dtype: DTypeLike | None = None,
) -> spmatrix | sparray: ...
@overload
def vstack(
    blocks: Iterable[spmatrix],
    format: Literal["csr", "csc", "coo"] | None = None,
    dtype: DTypeLike | None = None,
) -> spmatrix: ...
@overload
def vstack(
    blocks: Iterable[sparray],
    format: Literal["csr", "csc", "coo"] | None = None,
    dtype: DTypeLike | None = None,
) -> sparray: ...
@overload
def vstack(
    blocks: Iterable[_spbase],
    format: Literal["csr", "csc", "coo"] | None = None,
    dtype: DTypeLike | None = None,
) -> spmatrix | sparray: ...
