# SPDX-License-Identifier: MPL-2.0
import sys
from typing import TypeAlias

import numpy as np
import pytest
import scipy.sparse as sps

import cupy as cp
import cupy.cuda
import cupyx.scipy.sparse as cps

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

# `np.generic` etc. have `Any` type parameters
Scalar: TypeAlias = (  # strings, as `np.dtype[A | B]` isn’t subscriptable at runtime
    "np.bool | np.int8 | np.int16 | np.int32 | np.int64 | np.uint8 | np.uint16 | np.uint32 | np.uint64"
    " | np.float16 | np.float32 | np.float64 | np.complex64 | np.complex128"
)
SpScalar: TypeAlias = np.bool | np.float32 | np.float64 | np.complex64 | np.complex128


@pytest.fixture
def dense() -> cp.ndarray:
    return cp.asarray(np.array([[1, 0, 2], [0, 3, 0]], dtype=np.float32))


@pytest.fixture
def csr(dense: cp.ndarray) -> cps.csr_matrix:
    return cps.csr_matrix(dense)


@pytest.fixture
def csc(dense: cp.ndarray) -> cps.csc_matrix:
    return cps.csc_matrix(dense)


@pytest.fixture
def coo(dense: cp.ndarray) -> cps.coo_matrix:
    return cps.coo_matrix(dense)


def test_init(dense: cp.ndarray, csr: cps.csr_matrix) -> None:
    from_shape = cps.csr_matrix((2, 3), dtype=np.float64)
    assert from_shape.shape == (2, 3)
    from_scipy = cps.csc_matrix(sps.random(2, 3, format="csr", dtype=np.float32), copy=True)
    assert from_scipy.shape == (2, 3)
    from_cupy = cps.csc_matrix(csr)
    assert from_cupy.shape == (2, 3)
    from_parts = cps.csr_matrix((csr.data, csr.indices, csr.indptr), shape=(2, 3), maxprint=10)
    assert from_parts.shape == (2, 3)
    row, col = cp.asarray([0, 1]), cp.asarray([0, 1])
    from_coo = cps.coo_matrix(
        (cp.asarray([1.0, 2.0]), (row, col)), shape=(2, 3), dtype=np.float32, copy=False, maxprint=10
    )
    assert from_coo.shape == (2, 3)


def test_spmatrix_attrs(coo: cps.coo_matrix) -> None:
    m: cps.spmatrix = coo
    assert isinstance(m, cps.spmatrix)
    assert_type(m.dtype, np.dtype[SpScalar])
    assert m.dtype == np.float32
    assert_type(m.shape, tuple[int, int])
    assert m.shape == (2, 3)
    assert_type(m.ndim, int)
    assert m.ndim == 2


def test_formats(csr: cps.csr_matrix, csc: cps.csc_matrix, coo: cps.coo_matrix) -> None:
    assert csr.format == "csr"
    assert csc.format == "csc"
    assert coo.format == "coo"


def test_compressed_attrs(csr: cps.csr_matrix) -> None:
    assert_type(csr.data, cp.ndarray[tuple[int], np.dtype[SpScalar]])
    assert isinstance(csr.data, cp.ndarray)
    for arr in (csr.indices, csr.indptr):
        assert_type(arr, cp.ndarray[tuple[int], np.dtype[Scalar]])
        assert isinstance(arr, cp.ndarray)


def test_toarray(coo: cps.coo_matrix) -> None:
    x = coo.toarray(order="F", out=None)
    assert_type(x, cp.ndarray[tuple[int, int], np.dtype[SpScalar]])
    assert isinstance(x, cp.ndarray)


def test_copy(coo: cps.coo_matrix, csr: cps.csr_matrix, csc: cps.csc_matrix) -> None:
    assert_type(coo.copy(), cps.coo_matrix)
    assert isinstance(coo.copy(), cps.coo_matrix)
    assert_type(csr.copy(), cps.csr_matrix)
    assert isinstance(csr.copy(), cps.csr_matrix)
    assert_type(csc.copy(), cps.csc_matrix)
    assert isinstance(csc.copy(), cps.csc_matrix)


def test_convert(coo: cps.coo_matrix) -> None:
    assert_type(coo.tocsr(copy=True), cps.csr_matrix)
    assert isinstance(coo.tocsr(copy=True), cps.csr_matrix)
    assert_type(coo.tocsc(copy=True), cps.csc_matrix)
    assert isinstance(coo.tocsc(copy=True), cps.csc_matrix)


@pytest.fixture
def square() -> cps.csc_matrix:
    return cps.csc_matrix(cp.full((2, 2), 1.0, dtype=np.float32))


def test_matmul(square: cps.csc_matrix) -> None:
    x = square @ square
    assert_type(x, cps.csr_matrix)
    assert isinstance(x, cps.csr_matrix)


def test_pow(square: cps.csc_matrix) -> None:
    assert_type(square**1, cps.csc_matrix)
    assert isinstance(square**1, cps.csc_matrix)
    n: int = 2
    assert_type(square**n, cps.csr_matrix | cps.csc_matrix)
    assert isinstance(square**n, cps.csr_matrix)


def test_array_function(csr: cps.csr_matrix) -> None:
    assert np.sum(csr) == 6  # dispatched to `csr_matrix.sum`


# scipy’s classes are only subscriptable at runtime in newer versions
def test_get(coo: cps.coo_matrix, csr: cps.csr_matrix, csc: cps.csc_matrix) -> None:
    stream = cupy.cuda.Stream()
    m: cps.spmatrix = coo
    assert_type(m.get(stream), "sps.spmatrix[SpScalar]")
    assert isinstance(m.get(stream), sps.spmatrix)
    assert_type(coo.get(stream=stream), "sps.coo_matrix[SpScalar]")
    assert isinstance(coo.get(stream=stream), sps.coo_matrix)
    assert_type(csr.get(stream=stream), "sps.csr_matrix[SpScalar]")
    assert isinstance(csr.get(stream=stream), sps.csr_matrix)
    assert_type(csc.get(stream=stream), "sps.csc_matrix[SpScalar]")
    assert isinstance(csc.get(stream=stream), sps.csc_matrix)


def test_getitem(csr: cps.csr_matrix) -> None:
    x = csr[0, 2]
    assert_type(x, cp.ndarray[tuple[()], np.dtype[SpScalar]])
    assert isinstance(x, cp.ndarray)
    assert x.ndim == 0
    for key in (0, slice(None), [0, 1], [True, False], cp.asarray([0]), np.array([0])):
        assert_type(csr[key], cps.csr_matrix)
        assert isinstance(csr[key], cps.csr_matrix)
    assert isinstance(csr[0, 1:], cps.csr_matrix)
    assert isinstance(csr[1:, 0], cps.csr_matrix)


@pytest.mark.filterwarnings("ignore::cupyx.scipy.sparse.SparseEfficiencyWarning")  # `[0, 1:]` inserts entries
def test_setitem(csr: cps.csr_matrix) -> None:
    csr[0, 0] = 5
    assert csr[0, 0].item() == 5
    csr[0] = 0
    csr[0, 1:] = 1


def test_astype(csr: cps.csr_matrix) -> None:
    x = csr.astype(np.float64, copy=False)
    assert_type(x, "cps.csr_matrix[np.float64]")
    assert x.dtype == np.float64
    y = csr.astype(np.dtype(np.float32))
    assert_type(y, "cps.csr_matrix[np.float32]")
    assert_type(y.dtype, np.dtype[np.float32])
    assert_type(y.data, cp.ndarray[tuple[int], np.dtype[np.float32]])
    assert_type(y[0, 0], cp.ndarray[tuple[()], np.dtype[np.float32]])
    assert_type(y.tocsc(), "cps.csc_matrix[np.float32]")
    assert_type(y.get(), "sps.csr_matrix[np.float32]")
    assert_type(y.copy(), "cps.csr_matrix[np.float32]")
    m: cps.spmatrix = y  # covariant
    assert isinstance(m, cps.csr_matrix)
    assert_type(csr.astype("float64"), cps.csr_matrix)


def test_power(csc: cps.csc_matrix) -> None:
    x = csc.power(2, dtype=np.float64)
    assert_type(x, "cps.csc_matrix[np.float64]")
    assert isinstance(x, cps.csc_matrix)
    assert_type(csc.power(2), cps.csc_matrix)


def test_sum(csr: cps.csr_matrix) -> None:
    x = csr.sum(axis=0, dtype=np.float64, out=None)
    assert_type(x, cp.ndarray[tuple[int, int], np.dtype[Scalar]])
    assert x.shape == (1, 3)
    total = csr.sum()
    assert_type(total, cp.ndarray[tuple[()], np.dtype[Scalar]])
    assert total.shape == ()


def test_hstack(csr: cps.csr_matrix, csc: cps.csc_matrix) -> None:
    x = cps.hstack([csr, csc], format="csr", dtype=np.float64)
    assert_type(x, cps.spmatrix)
    assert isinstance(x, cps.csr_matrix)


def test_vstack(csr: cps.csr_matrix, csc: cps.csc_matrix) -> None:
    x = cps.vstack([csr, csc], format="csc", dtype=np.float64)
    assert_type(x, cps.spmatrix)
    assert isinstance(x, cps.csc_matrix)
