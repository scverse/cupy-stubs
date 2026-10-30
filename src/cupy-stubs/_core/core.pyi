# SPDX-License-Identifier: MPL-2.0
from collections.abc import Iterator, Sequence
from types import EllipsisType
from typing import Literal, overload

import numpy as np
from numpy.typing import DTypeLike
from typing_extensions import Self, override

from cupy.cuda import MemoryPointer, Stream

from .flags import Flags

# bare `np.generic`, `np.integer`, … have `Any` type parameters
type _Generic = np.generic[object]
type _Integer = np.int8 | np.int16 | np.int32 | np.int64 | np.uint8 | np.uint16 | np.uint32 | np.uint64
type _Slice = slice[int | None, int | None, int | None]
type _Shape = tuple[int, ...]
# the scalar types cupy supports
type _Scalar = np.bool | _Integer | np.float16 | np.float32 | np.float64 | np.complex64 | np.complex128
type _DType = np.dtype[_Scalar]

# `numpy.typing.NDArray` has `tuple[Any, ...]` as shape
type _NDArray[ScalarT: _Generic] = np.ndarray[_Shape, np.dtype[ScalarT]]

type _Index1D = int | _Slice | EllipsisType | _NDArray[_Integer | np.bool] | ndarray | None
type _Index = _Index1D | tuple[_Index1D, ...]

# like `numpy.ndarray`’s type parameters, but without `Any` defaults
class ndarray[ShapeT: _Shape = _Shape, DTypeT: _DType = _DType]:
    size: int
    ndim: int
    flags: Flags
    nbytes: int

    def __init__(
        self,
        shape: int | Sequence[int],
        dtype: DTypeLike | None = ...,
        memptr: MemoryPointer | None = None,
        strides: Sequence[int] | None = None,
        order: Literal["C", "F"] = "C",
    ) -> None: ...
    @property
    def shape(self) -> ShapeT: ...
    @property
    def dtype(self) -> DTypeT: ...
    @property
    def data(self) -> MemoryPointer: ...

    # cupy-specific
    def get(
        self,
        stream: Stream | None = None,
        order: Literal["C", "F", "A"] = "C",
        out: _NDArray[_Generic] | None = None,  # not `ShapeT`/`DTypeT`, which would make them invariant
        blocking: bool = True,
    ) -> np.ndarray[ShapeT, DTypeT]: ...

    # operators
    # Raises at runtime (no implicit conversion), but makes this `ArrayLike`,
    # so e.g. `np.all(arr)` type checks, which works by dispatching to `cupy.all` via `__array_function__`.
    def __array__(self) -> np.ndarray[ShapeT, DTypeT]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: _Index) -> ndarray[_Shape, DTypeT]: ...  # never returns scalars
    def __setitem__(self, index: _Index, value: object) -> None: ...
    def __iter__(self) -> Iterator[ndarray[_Shape, DTypeT]]: ...
    def __add__(self, other: object, /) -> ndarray: ...
    def __sub__(self, other: object, /) -> ndarray: ...
    @override
    def __eq__(self, value: object, /) -> ndarray[_Shape, np.dtype[np.bool]]: ...  # type: ignore[override]
    def __pow__(self, value: int, mod: None = None, /) -> ndarray: ...

    # methods
    def copy(self, order: Literal["C", "F", "A", "K"] = "C") -> ndarray[ShapeT, DTypeT]: ...
    def item(self) -> bool | int | float | complex: ...
    @overload
    def view(self, dtype: DTypeLike | None = None, type: None = None) -> ndarray[ShapeT]: ...
    @overload
    def view[ArrayT: ndarray](self, dtype: DTypeLike | None = None, *, type: type[ArrayT]) -> ArrayT: ...
    def astype(
        self,
        dtype: DTypeLike | None,
        order: Literal["C", "F", "A", "K"] = "K",
        casting: None = None,
        subok: None = None,
        copy: bool = True,
    ) -> ndarray[ShapeT]: ...
    @property
    def T(self) -> Self: ...
    @overload
    def all(self, axis: None = None) -> ndarray[tuple[()], np.dtype[np.bool]]: ...
    @overload
    def all(self, axis: int) -> ndarray[_Shape, np.dtype[np.bool]]: ...
    @overload
    def reshape(self, shape: int) -> ndarray[tuple[int], DTypeT]: ...
    @overload
    def reshape[NewShapeT: _Shape](self, shape: NewShapeT) -> ndarray[NewShapeT, DTypeT]: ...
    @overload
    def reshape(self, shape: Sequence[int]) -> ndarray[_Shape, DTypeT]: ...
    def squeeze(self, axis: int | None = None) -> ndarray[_Shape, DTypeT]: ...
    def ravel(self, order: Literal["C", "F", "A", "K"] = "C") -> ndarray[tuple[int], DTypeT]: ...
    def flatten(self, order: Literal["C", "F", "A", "K"] = "C") -> ndarray[tuple[int], DTypeT]: ...
    @property
    def flat(self) -> _FlatIter[DTypeT]: ...
    @overload
    def sum(
        self,
        axis: None = None,
        dtype: DTypeLike | None = None,
        out: None = None,
        keepdims: Literal[False] = False,
    ) -> ndarray[tuple[()]]: ...
    @overload
    def sum(
        self,
        axis: int | None = None,
        dtype: DTypeLike | None = None,
        out: ndarray | None = None,
        keepdims: bool = False,
    ) -> ndarray: ...

class _FlatIter[DTypeT: _DType = _DType]:
    def __next__(self) -> ndarray[tuple[()], DTypeT]: ...
    def __iter__(self) -> Self: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> ndarray[tuple[()], DTypeT]: ...
