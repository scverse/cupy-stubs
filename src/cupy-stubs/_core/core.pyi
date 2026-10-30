# SPDX-License-Identifier: MPL-2.0
from collections.abc import Iterable, Iterator, Sequence
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
type _Axis = int | tuple[int, ...]
type _Order = Literal["C", "F", "A"]
# the scalar types cupy supports
type _Scalar = np.bool | _Integer | np.float16 | np.float32 | np.float64 | np.complex64 | np.complex128
type _DType = np.dtype[_Scalar]
# `dtype` arguments that determine the scalar type
type _ToDType[ScalarT: _Scalar] = type[ScalarT] | np.dtype[ScalarT]

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

    @overload
    def __init__(
        self: ndarray[tuple[int]],
        shape: int,
        dtype: DTypeLike | None = ...,
        memptr: MemoryPointer | None = None,
        strides: Iterable[int] | None = None,
        order: Literal["C", "F"] = "C",
    ) -> None: ...
    @overload
    def __init__(
        self,
        shape: ShapeT,
        dtype: DTypeLike | None = ...,
        memptr: MemoryPointer | None = None,
        strides: Iterable[int] | None = None,
        order: Literal["C", "F"] = "C",
    ) -> None: ...
    @overload
    def __init__(
        self,
        shape: int | Sequence[int],
        dtype: DTypeLike | None = ...,
        memptr: MemoryPointer | None = None,
        strides: Iterable[int] | None = None,
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
    def __array__(self, dtype: None = None) -> np.ndarray[ShapeT, DTypeT]: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: _Index) -> ndarray[_Shape, DTypeT]: ...  # never returns scalars
    def __setitem__(self, index: _Index, value: object) -> None: ...
    def __iter__(self) -> Iterator[ndarray[_Shape, DTypeT]]: ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __invert__(self) -> ndarray[ShapeT, DTypeT]: ...
    def __add__(self, other: object, /) -> ndarray: ...
    def __radd__(self, other: object, /) -> ndarray: ...
    def __sub__(self, other: object, /) -> ndarray: ...
    def __floordiv__(self, other: object, /) -> ndarray: ...
    def __mod__(self, other: object, /) -> ndarray: ...
    def __lshift__(self, other: object, /) -> ndarray: ...
    def __rshift__(self, other: object, /) -> ndarray: ...
    @overload
    def __and__(
        self: ndarray[_Shape, np.dtype[np.bool]], other: ndarray[_Shape, np.dtype[np.bool]] | bool, /
    ) -> ndarray[_Shape, np.dtype[np.bool]]: ...
    @overload
    def __and__(self, other: object, /) -> ndarray: ...
    @overload
    def __or__(
        self: ndarray[_Shape, np.dtype[np.bool]], other: ndarray[_Shape, np.dtype[np.bool]] | bool, /
    ) -> ndarray[_Shape, np.dtype[np.bool]]: ...
    @overload
    def __or__(self, other: object, /) -> ndarray: ...
    @override
    def __eq__(self, value: object, /) -> ndarray[_Shape, np.dtype[np.bool]]: ...  # type: ignore[override]
    @override
    def __ne__(self, value: object, /) -> ndarray[_Shape, np.dtype[np.bool]]: ...  # type: ignore[override]
    def __lt__(self, other: object, /) -> ndarray[_Shape, np.dtype[np.bool]]: ...
    def __le__(self, other: object, /) -> ndarray[_Shape, np.dtype[np.bool]]: ...
    def __gt__(self, other: object, /) -> ndarray[_Shape, np.dtype[np.bool]]: ...
    def __ge__(self, other: object, /) -> ndarray[_Shape, np.dtype[np.bool]]: ...
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
    # one `axis: int` overload per ndim, for the same reason as in `reshape`
    @overload
    def all[OutT: ndarray](self, axis: _Axis | None, out: OutT, keepdims: bool = False) -> OutT: ...
    @overload
    def all[OutT: ndarray](self, axis: _Axis | None = None, *, out: OutT, keepdims: bool = False) -> OutT: ...
    @overload
    def all(
        self, axis: None = None, out: None = None, keepdims: Literal[False] = False
    ) -> ndarray[tuple[()], np.dtype[np.bool]]: ...
    @overload
    def all(
        self, axis: _Axis | None = None, out: None = None, *, keepdims: Literal[True]
    ) -> ndarray[ShapeT, np.dtype[np.bool]]: ...
    @overload
    def all(
        self: ndarray[tuple[int]], axis: int, out: None = None, keepdims: Literal[False] = False
    ) -> ndarray[tuple[()], np.dtype[np.bool]]: ...
    @overload
    def all(
        self: ndarray[tuple[int, int]], axis: int, out: None = None, keepdims: Literal[False] = False
    ) -> ndarray[tuple[int], np.dtype[np.bool]]: ...
    @overload
    def all(
        self: ndarray[tuple[int, int, int]], axis: int, out: None = None, keepdims: Literal[False] = False
    ) -> ndarray[tuple[int, int], np.dtype[np.bool]]: ...
    @overload
    def all(
        self, axis: _Axis | None = None, out: None = None, keepdims: bool = False
    ) -> ndarray[_Shape, np.dtype[np.bool]]: ...
    @overload
    def any[OutT: ndarray](self, axis: _Axis | None, out: OutT, keepdims: bool = False) -> OutT: ...
    @overload
    def any[OutT: ndarray](self, axis: _Axis | None = None, *, out: OutT, keepdims: bool = False) -> OutT: ...
    @overload
    def any(
        self, axis: None = None, out: None = None, keepdims: Literal[False] = False
    ) -> ndarray[tuple[()], np.dtype[np.bool]]: ...
    @overload
    def any(
        self, axis: _Axis | None = None, out: None = None, *, keepdims: Literal[True]
    ) -> ndarray[ShapeT, np.dtype[np.bool]]: ...
    @overload
    def any(
        self: ndarray[tuple[int]], axis: int, out: None = None, keepdims: Literal[False] = False
    ) -> ndarray[tuple[()], np.dtype[np.bool]]: ...
    @overload
    def any(
        self: ndarray[tuple[int, int]], axis: int, out: None = None, keepdims: Literal[False] = False
    ) -> ndarray[tuple[int], np.dtype[np.bool]]: ...
    @overload
    def any(
        self: ndarray[tuple[int, int, int]], axis: int, out: None = None, keepdims: Literal[False] = False
    ) -> ndarray[tuple[int, int], np.dtype[np.bool]]: ...
    @overload
    def any(
        self, axis: _Axis | None = None, out: None = None, keepdims: bool = False
    ) -> ndarray[_Shape, np.dtype[np.bool]]: ...
    @overload
    def max[OutT: ndarray](self, axis: _Axis | None, out: OutT, keepdims: bool = False) -> OutT: ...
    @overload
    def max[OutT: ndarray](self, axis: _Axis | None = None, *, out: OutT, keepdims: bool = False) -> OutT: ...
    @overload
    def max(
        self, axis: None = None, out: None = None, keepdims: Literal[False] = False
    ) -> ndarray[tuple[()], DTypeT]: ...
    @overload
    def max(
        self, axis: _Axis | None = None, out: None = None, *, keepdims: Literal[True]
    ) -> ndarray[ShapeT, DTypeT]: ...
    @overload
    def max(
        self: ndarray[tuple[int], DTypeT], axis: int, out: None = None, keepdims: Literal[False] = False
    ) -> ndarray[tuple[()], DTypeT]: ...
    @overload
    def max(
        self: ndarray[tuple[int, int], DTypeT], axis: int, out: None = None, keepdims: Literal[False] = False
    ) -> ndarray[tuple[int], DTypeT]: ...
    @overload
    def max(
        self: ndarray[tuple[int, int, int], DTypeT], axis: int, out: None = None, keepdims: Literal[False] = False
    ) -> ndarray[tuple[int, int], DTypeT]: ...
    @overload
    def max(self, axis: _Axis | None = None, out: None = None, keepdims: bool = False) -> ndarray[_Shape, DTypeT]: ...
    @overload
    def argmax[OutT: ndarray](
        self, axis: _Axis | None, out: OutT, dtype: DTypeLike | None = None, keepdims: bool = False
    ) -> OutT: ...
    @overload
    def argmax[OutT: ndarray](
        self, axis: _Axis | None = None, *, out: OutT, dtype: DTypeLike | None = None, keepdims: bool = False
    ) -> OutT: ...
    @overload
    def argmax[ScalarT: _Scalar = np.int64](
        self,
        axis: None = None,
        out: None = None,
        dtype: _ToDType[ScalarT] | None = None,
        keepdims: Literal[False] = False,
    ) -> ndarray[tuple[()], np.dtype[ScalarT]]: ...
    @overload
    def argmax[ScalarT: _Scalar = np.int64](
        self,
        axis: _Axis | None = None,
        out: None = None,
        dtype: _ToDType[ScalarT] | None = None,
        *,
        keepdims: Literal[True],
    ) -> ndarray[ShapeT, np.dtype[ScalarT]]: ...
    @overload
    def argmax[ScalarT: _Scalar = np.int64](
        self: ndarray[tuple[int]],
        axis: int,
        out: None = None,
        dtype: _ToDType[ScalarT] | None = None,
        keepdims: Literal[False] = False,
    ) -> ndarray[tuple[()], np.dtype[ScalarT]]: ...
    @overload
    def argmax[ScalarT: _Scalar = np.int64](
        self: ndarray[tuple[int, int]],
        axis: int,
        out: None = None,
        dtype: _ToDType[ScalarT] | None = None,
        keepdims: Literal[False] = False,
    ) -> ndarray[tuple[int], np.dtype[ScalarT]]: ...
    @overload
    def argmax[ScalarT: _Scalar = np.int64](
        self: ndarray[tuple[int, int, int]],
        axis: int,
        out: None = None,
        dtype: _ToDType[ScalarT] | None = None,
        keepdims: Literal[False] = False,
    ) -> ndarray[tuple[int, int], np.dtype[ScalarT]]: ...
    @overload
    def argmax[ScalarT: _Scalar = np.int64](
        self,
        axis: _Axis | None = None,
        out: None = None,
        dtype: _ToDType[ScalarT] | None = None,
        keepdims: bool = False,
    ) -> ndarray[_Shape, np.dtype[ScalarT]]: ...
    @overload
    def argmax(
        self, axis: _Axis | None = None, out: None = None, dtype: DTypeLike | None = None, keepdims: bool = False
    ) -> ndarray: ...
    @overload
    def min[OutT: ndarray](self, axis: _Axis | None, out: OutT, keepdims: bool = False) -> OutT: ...
    @overload
    def min[OutT: ndarray](self, axis: _Axis | None = None, *, out: OutT, keepdims: bool = False) -> OutT: ...
    @overload
    def min(
        self, axis: None = None, out: None = None, keepdims: Literal[False] = False
    ) -> ndarray[tuple[()], DTypeT]: ...
    @overload
    def min(
        self, axis: _Axis | None = None, out: None = None, *, keepdims: Literal[True]
    ) -> ndarray[ShapeT, DTypeT]: ...
    @overload
    def min(
        self: ndarray[tuple[int], DTypeT], axis: int, out: None = None, keepdims: Literal[False] = False
    ) -> ndarray[tuple[()], DTypeT]: ...
    @overload
    def min(
        self: ndarray[tuple[int, int], DTypeT], axis: int, out: None = None, keepdims: Literal[False] = False
    ) -> ndarray[tuple[int], DTypeT]: ...
    @overload
    def min(
        self: ndarray[tuple[int, int, int], DTypeT], axis: int, out: None = None, keepdims: Literal[False] = False
    ) -> ndarray[tuple[int, int], DTypeT]: ...
    @overload
    def min(self, axis: _Axis | None = None, out: None = None, keepdims: bool = False) -> ndarray[_Shape, DTypeT]: ...
    @overload
    def argmin[OutT: ndarray](
        self, axis: _Axis | None, out: OutT, dtype: DTypeLike | None = None, keepdims: bool = False
    ) -> OutT: ...
    @overload
    def argmin[OutT: ndarray](
        self, axis: _Axis | None = None, *, out: OutT, dtype: DTypeLike | None = None, keepdims: bool = False
    ) -> OutT: ...
    @overload
    def argmin[ScalarT: _Scalar = np.int64](
        self,
        axis: None = None,
        out: None = None,
        dtype: _ToDType[ScalarT] | None = None,
        keepdims: Literal[False] = False,
    ) -> ndarray[tuple[()], np.dtype[ScalarT]]: ...
    @overload
    def argmin[ScalarT: _Scalar = np.int64](
        self,
        axis: _Axis | None = None,
        out: None = None,
        dtype: _ToDType[ScalarT] | None = None,
        *,
        keepdims: Literal[True],
    ) -> ndarray[ShapeT, np.dtype[ScalarT]]: ...
    @overload
    def argmin[ScalarT: _Scalar = np.int64](
        self: ndarray[tuple[int]],
        axis: int,
        out: None = None,
        dtype: _ToDType[ScalarT] | None = None,
        keepdims: Literal[False] = False,
    ) -> ndarray[tuple[()], np.dtype[ScalarT]]: ...
    @overload
    def argmin[ScalarT: _Scalar = np.int64](
        self: ndarray[tuple[int, int]],
        axis: int,
        out: None = None,
        dtype: _ToDType[ScalarT] | None = None,
        keepdims: Literal[False] = False,
    ) -> ndarray[tuple[int], np.dtype[ScalarT]]: ...
    @overload
    def argmin[ScalarT: _Scalar = np.int64](
        self: ndarray[tuple[int, int, int]],
        axis: int,
        out: None = None,
        dtype: _ToDType[ScalarT] | None = None,
        keepdims: Literal[False] = False,
    ) -> ndarray[tuple[int, int], np.dtype[ScalarT]]: ...
    @overload
    def argmin[ScalarT: _Scalar = np.int64](
        self,
        axis: _Axis | None = None,
        out: None = None,
        dtype: _ToDType[ScalarT] | None = None,
        keepdims: bool = False,
    ) -> ndarray[_Shape, np.dtype[ScalarT]]: ...
    @overload
    def argmin(
        self, axis: _Axis | None = None, out: None = None, dtype: DTypeLike | None = None, keepdims: bool = False
    ) -> ndarray: ...
    # can’t just use a single overload for `def reshape[*Ts: int]` and `Unpack[*Ts]` here
    # since bounds for variadic type vars are not supported
    @overload
    def reshape(
        self, shape: int, /, *, order: _Order = "C", copy: bool | None = None
    ) -> ndarray[tuple[int], DTypeT]: ...
    @overload
    def reshape[NewShapeT: _Shape](
        self, shape: NewShapeT, /, *, order: _Order = "C", copy: bool | None = None
    ) -> ndarray[NewShapeT, DTypeT]: ...
    @overload
    def reshape(
        self, size1: int, size2: int, /, *, order: _Order = "C", copy: bool | None = None
    ) -> ndarray[tuple[int, int], DTypeT]: ...
    @overload
    def reshape(
        self, size1: int, size2: int, size3: int, /, *, order: _Order = "C", copy: bool | None = None
    ) -> ndarray[tuple[int, int, int], DTypeT]: ...
    @overload
    def reshape(
        self,
        size1: int,
        size2: int,
        size3: int,
        size4: int,
        /,
        *shape: int,
        order: _Order = "C",
        copy: bool | None = None,
    ) -> ndarray[tuple[int, int, int, int, *tuple[int, ...]], DTypeT]: ...
    @overload
    def squeeze(self: ndarray[tuple[int], DTypeT], axis: int) -> ndarray[tuple[()], DTypeT]: ...
    @overload
    def squeeze(self: ndarray[tuple[int, int], DTypeT], axis: int) -> ndarray[tuple[int], DTypeT]: ...
    @overload
    def squeeze(self: ndarray[tuple[int, int, int], DTypeT], axis: int) -> ndarray[tuple[int, int], DTypeT]: ...
    @overload
    def squeeze(self, axis: int | None = None) -> ndarray[_Shape, DTypeT]: ...
    def ravel(self, order: Literal["C", "F", "A", "K"] = "C") -> ndarray[tuple[int], DTypeT]: ...
    def flatten(self, order: Literal["C", "F", "A", "K"] = "C") -> ndarray[tuple[int], DTypeT]: ...
    @property
    def flat(self) -> _FlatIter[DTypeT]: ...
    @overload
    def sum[OutT: ndarray](
        self, axis: _Axis | None, dtype: DTypeLike | None, out: OutT, keepdims: bool = False
    ) -> OutT: ...
    @overload
    def sum[OutT: ndarray](
        self, axis: _Axis | None = None, dtype: DTypeLike | None = None, *, out: OutT, keepdims: bool = False
    ) -> OutT: ...
    @overload
    def sum[ScalarT: _Scalar = _Scalar](
        self,
        axis: None = None,
        dtype: _ToDType[ScalarT] | None = None,
        out: None = None,
        keepdims: Literal[False] = False,
    ) -> ndarray[tuple[()], np.dtype[ScalarT]]: ...
    @overload
    def sum[ScalarT: _Scalar = _Scalar](
        self,
        axis: _Axis | None = None,
        dtype: _ToDType[ScalarT] | None = None,
        out: None = None,
        *,
        keepdims: Literal[True],
    ) -> ndarray[ShapeT, np.dtype[ScalarT]]: ...
    @overload
    def sum[ScalarT: _Scalar = _Scalar](
        self: ndarray[tuple[int]],
        axis: int,
        dtype: _ToDType[ScalarT] | None = None,
        out: None = None,
        keepdims: Literal[False] = False,
    ) -> ndarray[tuple[()], np.dtype[ScalarT]]: ...
    @overload
    def sum[ScalarT: _Scalar = _Scalar](
        self: ndarray[tuple[int, int]],
        axis: int,
        dtype: _ToDType[ScalarT] | None = None,
        out: None = None,
        keepdims: Literal[False] = False,
    ) -> ndarray[tuple[int], np.dtype[ScalarT]]: ...
    @overload
    def sum[ScalarT: _Scalar = _Scalar](
        self: ndarray[tuple[int, int, int]],
        axis: int,
        dtype: _ToDType[ScalarT] | None = None,
        out: None = None,
        keepdims: Literal[False] = False,
    ) -> ndarray[tuple[int, int], np.dtype[ScalarT]]: ...
    @overload
    def sum[ScalarT: _Scalar = _Scalar](
        self,
        axis: _Axis | None = None,
        dtype: _ToDType[ScalarT] | None = None,
        out: None = None,
        keepdims: bool = False,
    ) -> ndarray[_Shape, np.dtype[ScalarT]]: ...
    @overload
    def sum(
        self, axis: _Axis | None = None, dtype: DTypeLike | None = None, out: None = None, keepdims: bool = False
    ) -> ndarray: ...

class _FlatIter[DTypeT: _DType = _DType]:
    def __next__(self) -> ndarray[tuple[()], DTypeT]: ...
    def __iter__(self) -> Self: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> ndarray[tuple[()], DTypeT]: ...
