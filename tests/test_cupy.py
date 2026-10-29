# SPDX-License-Identifier: MPL-2.0
import sys
from typing import TypeAlias

import h5py
import numpy as np
import pytest
import zarr

import cupy as cp
from cupy._core.flags import Flags

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type
import cupy.cuda

# `np.generic` etc. have `Any` type parameters
Scalar: TypeAlias = (  # strings, as `np.dtype[A | B]` isn’t subscriptable at runtime
    "np.bool | np.int8 | np.int16 | np.int32 | np.int64 | np.uint8 | np.uint16 | np.uint32 | np.uint64"
    " | np.float16 | np.float32 | np.float64 | np.complex64 | np.complex128"
)


@pytest.fixture
def a() -> cp.ndarray:
    return cp.asarray(np.arange(6, dtype=np.float32).reshape(2, 3))


def test_ndarray_init(a: cp.ndarray) -> None:
    x = cp.ndarray((2, 3), dtype=np.float32, strides=None, order="F")
    assert_type(x, cp.ndarray)
    assert isinstance(x, cp.ndarray)
    assert x.shape == (2, 3)

    ptr = a.data
    assert_type(ptr, cupy.cuda.MemoryPointer)
    y = cp.ndarray(a.shape, dtype=a.dtype, memptr=ptr)
    assert isinstance(ptr, cupy.cuda.MemoryPointer)
    assert isinstance(y, cp.ndarray)


def test_ndarray_attrs(a: cp.ndarray) -> None:
    assert_type(a.dtype, np.dtype[Scalar])
    assert isinstance(a.dtype, np.dtype)
    assert_type(a.shape, tuple[int, ...])
    assert a.shape == (2, 3)
    assert_type(a.size, int)
    assert a.size == 6
    assert_type(a.ndim, int)
    assert a.ndim == 2
    assert_type(a.flags, Flags)
    assert isinstance(a.flags, Flags)
    assert a.flags.c_contiguous
    assert not a.flags.f_contiguous
    assert a.flags.owndata
    assert_type(a.nbytes, int)
    assert a.nbytes == 24


def test_ndarray_get(a: cp.ndarray) -> None:
    stream = cupy.cuda.Stream()
    assert isinstance(stream, cupy.cuda.Stream)
    x = a.get(stream=stream, order="C", out=None, blocking=True)
    assert_type(x, np.ndarray[tuple[int, ...], np.dtype[Scalar]])
    assert isinstance(x, np.ndarray)


def test_ndarray_array_function(a: cp.ndarray) -> None:
    assert np.all(a == a)  # dispatched to `cupy.all`


def test_ndarray_dunders(a: cp.ndarray) -> None:
    assert_type(len(a), int)
    assert len(a) == 2

    row = a[0]
    assert_type(row, cp.ndarray)
    assert isinstance(row, cp.ndarray)
    assert isinstance(a[..., None, 1:], cp.ndarray)
    assert isinstance(a[np.array([0, 1])], cp.ndarray)
    assert isinstance(a[cp.asarray([True, False])], cp.ndarray)
    a[0, 0] = 5
    assert a[0, 0].item() == 5

    rows = list(iter(a))
    assert_type(rows, list[cp.ndarray])
    assert all(isinstance(r, cp.ndarray) for r in rows)

    assert_type(a + 1, cp.ndarray)
    assert isinstance(a + 1, cp.ndarray)
    assert_type(a - a, cp.ndarray)
    assert isinstance(a - a, cp.ndarray)
    assert_type(a == a, cp.ndarray[tuple[int, ...], np.dtype[np.bool]])
    assert isinstance(a == a, cp.ndarray)
    assert_type(a**2, cp.ndarray)
    assert isinstance(a**2, cp.ndarray)
    assert_type(1 + a, cp.ndarray)
    assert isinstance(1 + a, cp.ndarray)
    assert_type(a // 2, cp.ndarray)
    assert isinstance(a // 2, cp.ndarray)
    assert_type(a % 2, cp.ndarray)
    assert isinstance(a % 2, cp.ndarray)
    assert_type(int(a[0, 1]), int)
    assert int(a[0, 1]) == 1
    assert_type(float(a[0, 1]), float)
    assert float(a[0, 1]) == 1.0
    i = a.astype(np.int64)
    assert_type(~i, cp.ndarray)
    assert (~i).dtype == np.int64


def test_ndarray_comparisons(a: cp.ndarray) -> None:
    for x in (a != a, a < 1, a <= 1, a > 1, a >= 1):
        assert_type(x, cp.ndarray[tuple[int, ...], np.dtype[np.bool]])
        assert x.dtype == np.bool_


def test_ndarray_bitwise(a: cp.ndarray) -> None:
    i = a.astype(np.int64)
    for x in (i << 1, i >> 1, i & 1, i | 1):
        assert_type(x, cp.ndarray)
        assert isinstance(x, cp.ndarray)
    b = a > 1
    assert_type(b & b, cp.ndarray[tuple[int, ...], np.dtype[np.bool]])
    assert (b & b).dtype == np.bool_
    assert_type(b | True, cp.ndarray[tuple[int, ...], np.dtype[np.bool]])
    assert (b | True).dtype == np.bool_


def test_ndarray_copy(a: cp.ndarray) -> None:
    assert_type(a.copy(order="F"), cp.ndarray)
    assert isinstance(a.copy(order="F"), cp.ndarray)


def test_ndarray_item(a: cp.ndarray) -> None:
    x = a[0, 0].item()
    assert_type(x, bool | int | float | complex)
    assert isinstance(x, bool | int | float | complex)


def test_ndarray_view(a: cp.ndarray) -> None:
    v = a.view(np.int32)
    assert_type(v, cp.ndarray)
    assert v.dtype == np.int32

    class Sub(cp.ndarray): ...

    s = a.view(type=Sub)
    assert_type(s, Sub)
    assert isinstance(s, Sub)


def test_ndarray_astype(a: cp.ndarray) -> None:
    x = a.astype(np.float64, order="C", casting=None, subok=None, copy=False)
    assert_type(x, cp.ndarray)
    assert x.dtype == np.float64


def test_ndarray_T(a: cp.ndarray) -> None:
    assert_type(a.T, cp.ndarray)
    assert a.T.shape == (3, 2)


def test_ndarray_all(a: cp.ndarray) -> None:
    x = a.all()
    assert_type(x, cp.ndarray[tuple[()], np.dtype[np.bool]])
    assert x.ndim == 0
    y = a.all(axis=0)
    assert_type(y, cp.ndarray[tuple[int, ...], np.dtype[np.bool]])
    assert isinstance(y, cp.ndarray)


def test_ndarray_any(a: cp.ndarray) -> None:
    x = a.any()
    assert_type(x, cp.ndarray[tuple[()], np.dtype[np.bool]])
    assert x.ndim == 0
    y = a.any(axis=1)
    assert_type(y, cp.ndarray[tuple[int, ...], np.dtype[np.bool]])
    assert y.shape == (2,)


def test_ndarray_max(a: cp.ndarray) -> None:
    x = a.max()
    assert_type(x, cp.ndarray[tuple[()], np.dtype[Scalar]])
    assert x.item() == 5
    y = a.max(axis=0)
    assert_type(y, cp.ndarray)
    assert y.shape == (3,)


def test_ndarray_argmax(a: cp.ndarray) -> None:
    x = a.argmax()
    assert_type(x, cp.ndarray[tuple[()], np.dtype[np.int64]])
    assert x.item() == 5
    y = a.argmax(axis=1)
    assert_type(y, cp.ndarray[tuple[int, ...], np.dtype[np.int64]])
    assert y.dtype == np.int64


def test_ndarray_shape_methods(a: cp.ndarray) -> None:
    assert_type(a.reshape((3, 2)), cp.ndarray[tuple[int, int], np.dtype[Scalar]])
    assert a.reshape((3, 2)).shape == (3, 2)
    assert_type(a.reshape(6), cp.ndarray[tuple[int], np.dtype[Scalar]])
    assert_type(a.reshape([6]), cp.ndarray[tuple[int, ...], np.dtype[Scalar]])
    assert a.reshape([6]).shape == (6,)
    assert a.reshape(6).shape == (6,)
    assert_type(a[None].squeeze(axis=0), cp.ndarray)
    assert a[None].squeeze(axis=0).shape == (2, 3)
    assert_type(a.ravel(order="F"), cp.ndarray[tuple[int], np.dtype[Scalar]])
    assert a.ravel(order="F").shape == (6,)
    assert_type(a.flatten(order="F"), cp.ndarray[tuple[int], np.dtype[Scalar]])
    assert a.flatten(order="F").shape == (6,)


def test_ndarray_flat(a: cp.ndarray) -> None:
    it = a.flat
    assert_type(len(it), int)
    assert len(it) == 6
    assert_type(it[0], cp.ndarray[tuple[()], np.dtype[Scalar]])
    assert it[0].ndim == 0
    x = next(iter(it))
    assert_type(x, cp.ndarray[tuple[()], np.dtype[Scalar]])
    assert x.ndim == 0


def test_ndarray_sum(a: cp.ndarray) -> None:
    out = cp.ndarray((3,), dtype=np.float64)
    x = a.sum(axis=0, dtype=np.float64, out=out, keepdims=False)
    assert_type(x, cp.ndarray)
    assert isinstance(x, cp.ndarray)
    total = a.sum()
    assert_type(total, cp.ndarray[tuple[()], np.dtype[Scalar]])
    assert total.shape == ()


def test_full() -> None:
    x = cp.full((2, 3), 1.5, dtype=np.float32, order="F")
    assert_type(x, cp.ndarray[tuple[int, ...], np.dtype[np.float32]])
    assert x.dtype == np.float32
    y = cp.full(3, cp.asarray(2))
    assert_type(y, cp.ndarray)
    assert isinstance(y, cp.ndarray)
    assert_type(cp.full(3, 1, dtype="int32"), cp.ndarray)  # string `dtype` isn’t inferred


def test_ones() -> None:
    x = cp.ones((2, 3), dtype=np.float32, order="F")
    assert_type(x, cp.ndarray[tuple[int, ...], np.dtype[np.float32]])
    assert x.dtype == np.float32
    assert_type(cp.ones(3), cp.ndarray)
    assert isinstance(cp.ones(3), cp.ndarray)


def test_zeros() -> None:
    x = cp.zeros((2, 3), dtype=np.dtype(np.int64), order="F")
    assert_type(x, cp.ndarray[tuple[int, ...], np.dtype[np.int64]])
    assert x.dtype == np.int64
    assert_type(cp.zeros(3), cp.ndarray)
    assert cp.zeros(3).dtype == np.float64


def test_empty() -> None:
    x = cp.empty((2, 3), dtype=np.uint32, order="F")
    assert_type(x, cp.ndarray[tuple[int, ...], np.dtype[np.uint32]])
    assert x.dtype == np.uint32
    assert_type(cp.empty(3), cp.ndarray)
    assert cp.empty(3).dtype == np.float64


def test_arange() -> None:
    x = cp.arange(0, 6, 2, dtype=np.float32)
    assert_type(x, cp.ndarray[tuple[int], np.dtype[Scalar]])
    assert x.shape == (3,)
    assert cp.arange(3).shape == (3,)


def test_scalar_types() -> None:
    for name in (
        "bool_", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
        "float16", "float32", "float64", "complex64", "complex128",
    ):  # fmt: skip
        assert getattr(cp, name) is getattr(np, name)
    assert_type(cp.float32, type[np.float32])
    assert_type(cp.nan, float)
    assert np.isnan(cp.nan)


def test_result_type(a: cp.ndarray) -> None:
    dt = cp.result_type(a.dtype, np.int64)
    assert_type(dt, np.dtype)
    assert dt == np.float64


def test_asarray() -> None:
    x = cp.asarray([1, 2], dtype=np.int64, order="C", copy=True, blocking=True)
    assert_type(x, cp.ndarray)
    assert isinstance(x, cp.ndarray)


def test_asarray_h5py() -> None:
    with h5py.File("mem.h5", "w", driver="core", backing_store=False) as f:
        d = f.create_dataset("x", data=np.arange(3))
        x = cp.asarray(d)
    assert isinstance(x, cp.ndarray)


def test_asarray_zarr() -> None:
    z = zarr.zeros(shape=(2, 3))
    x = cp.asarray(z)
    assert isinstance(x, cp.ndarray)


def test_ascontiguousarray(a: cp.ndarray) -> None:
    x = cp.ascontiguousarray(a.T, dtype=np.float64)
    assert_type(x, cp.ndarray)
    assert x.flags.c_contiguous


def test_asnumpy(a: cp.ndarray) -> None:
    stream = cupy.cuda.Stream()
    x = cp.asnumpy(a, stream=stream, order="C", out=None, blocking=True)
    assert_type(x, np.ndarray[tuple[int, ...], np.dtype[Scalar]])
    assert isinstance(x, np.ndarray)
    f32 = cp.zeros(3, dtype=np.float32)
    assert_type(cp.asnumpy(f32), np.ndarray[tuple[int, ...], np.dtype[np.float32]])
    y = cp.asnumpy([1, 2])
    assert_type(y, "np.ndarray[tuple[int, ...], np.dtype[np.generic[object]]]")
    assert isinstance(y, np.ndarray)


def test_array(a: cp.ndarray) -> None:
    x = cp.array(a, dtype=np.float64, copy=True, order="K", subok=False, ndmin=3, blocking=True)
    assert_type(x, cp.ndarray)
    assert x.ndim == 3


def test_array_equal(a: cp.ndarray) -> None:
    x = cp.array_equal(a, a, equal_nan=True)
    assert_type(x, cp.ndarray[tuple[()], np.dtype[np.bool]])
    assert isinstance(x, cp.ndarray)


def test_broadcast_to(a: cp.ndarray) -> None:
    x = cp.broadcast_to(a, (4, 2, 3))
    assert_type(x, cp.ndarray)
    assert x.shape == (4, 2, 3)


def test_concatenate(a: cp.ndarray) -> None:
    x = cp.concatenate([a, a], axis=None, dtype=np.float64, casting="same_kind")
    assert_type(x, cp.ndarray)
    assert x.shape == (12,)
    out = cp.ndarray((4, 3), dtype=np.float32)
    y = cp.concatenate((a, a), out=out)
    assert isinstance(y, cp.ndarray)


def test_append(a: cp.ndarray) -> None:
    x = cp.append(a, [1.0, 2.0], axis=None)
    assert_type(x, cp.ndarray)
    assert x.shape == (8,)


def test_unique() -> None:
    x = cp.asarray([3, 1, 3])
    u = cp.unique(x)
    assert_type(u, cp.ndarray)
    assert u.shape == (2,)
    r = cp.unique(x, return_index=True, return_inverse=True, return_counts=True, axis=None, equal_nan=False)
    assert_type(r, cp.ndarray | tuple[cp.ndarray, ...])
    assert isinstance(r, tuple)
    assert len(r) == 4


def test_stack(a: cp.ndarray) -> None:
    x = cp.stack([a, a], axis=1, dtype=np.float64, casting="same_kind")
    assert_type(x, cp.ndarray)
    assert x.shape == (2, 2, 3)
    out = cp.ndarray((2, 2, 3), dtype=np.float32)
    y = cp.stack((a, a), out=out)
    assert isinstance(y, cp.ndarray)


def test_repeat(a: cp.ndarray) -> None:
    x = cp.repeat(a, 2, axis=0)
    assert_type(x, cp.ndarray)
    assert x.shape == (4, 3)
    assert cp.repeat(cp.arange(2), cp.asarray([1, 2])).shape == (3,)
    assert cp.repeat(cp.arange(2), [1, 2]).shape == (3,)


def test_cumsum(a: cp.ndarray) -> None:
    out = cp.ndarray((2, 3), dtype=np.float64)
    x = cp.cumsum(a, axis=1, dtype=np.float64, out=out)
    assert_type(x, cp.ndarray)
    assert x.shape == (2, 3)
    assert cp.cumsum(a, axis=None).shape == (6,)


def test_diff(a: cp.ndarray) -> None:
    x = cp.diff(a, n=1, axis=1, prepend=0, append=cp.zeros((2, 1), dtype=np.float32))
    assert_type(x, cp.ndarray)
    assert x.shape == (2, 4)


def test_log2(a: cp.ndarray) -> None:
    x = cp.log2(a + 1, out=None, dtype=np.float64, casting="same_kind")
    assert_type(x, cp.ndarray)
    assert x.dtype == np.float64
    assert isinstance(cp.log2(2.0), cp.ndarray)
    assert isinstance(cp.log2, cp.ufunc)
    assert_type(cp.log2.nin, int)
    assert cp.log2.nin == 1
    assert_type(cp.log2.nout, int)
    assert cp.log2.nout == 1


def test_argsort(a: cp.ndarray) -> None:
    x = cp.argsort(a, axis=None, kind="stable")
    assert_type(x, cp.ndarray)
    assert x.shape == (6,)


def test_lexsort() -> None:
    x = cp.lexsort(cp.asarray([[1, 0, 1], [0, 0, 1]]))
    assert_type(x, cp.ndarray[tuple[int], np.dtype[Scalar]])
    assert x.get().tolist() == [1, 0, 2]


def test_searchsorted() -> None:
    x = cp.searchsorted(cp.arange(5), cp.asarray([1, 3]), side="right", sorter=cp.arange(5))
    assert_type(x, cp.ndarray)
    assert x.get().tolist() == [2, 4]
    assert cp.searchsorted(cp.arange(5), 2).item() == 2


def test_nonzero(a: cp.ndarray) -> None:
    x = cp.flatnonzero(a)
    assert_type(x, cp.ndarray[tuple[int], np.dtype[Scalar]])
    assert x.shape == (5,)
    rows, cols = cp.nonzero(a)
    assert_type(cp.nonzero(a), tuple[cp.ndarray[tuple[int], np.dtype[Scalar]], ...])
    assert rows.shape == cols.shape == (5,)


def test_where(a: cp.ndarray) -> None:
    rows, cols = cp.where(a > 1, x=None, y=None)
    assert_type(cp.where(a > 1), tuple[cp.ndarray[tuple[int], np.dtype[Scalar]], ...])
    assert rows.shape == cols.shape == (4,)
    x = cp.where(a > 1, a, 0)
    assert_type(x, cp.ndarray)
    assert x.shape == (2, 3)


def test_bincount() -> None:
    x = cp.bincount(cp.asarray([0, 1, 1]), weights=cp.asarray([1.0, 2.0, 3.0]), minlength=4)
    assert_type(x, cp.ndarray[tuple[int], np.dtype[Scalar]])
    assert x.get().tolist() == [1.0, 5.0, 0.0, 0.0]


def test_default_rng() -> None:
    rng = cp.random.default_rng(seed=0)
    assert_type(rng, cp.random.Generator)
    assert isinstance(rng, cp.random.Generator)
    x = rng.standard_normal(size=(2, 3), dtype=np.float32, out=None)
    assert_type(x, cp.ndarray)
    assert x.dtype == np.float32


def test_device() -> None:
    attributes = cupy.cuda.Device(0).attributes
    assert_type(attributes, dict[str, int])
    assert attributes["MaxGridDimY"] > 0


def test_get_current_stream() -> None:
    stream = cupy.cuda.get_current_stream(device_id=0)
    assert_type(stream, cupy.cuda.Stream)
    assert isinstance(stream, cupy.cuda.Stream)
    assert_type(stream.ptr, int)
    assert isinstance(stream.ptr, int)
