# cupy-stubs

[![Tests][badge-tests]][tests]
[![PyPI][badge-pypi]][pypi]
[![Python versions][badge-python]][pypi]
[![License][badge-license]][license]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/scverse/cupy-stubs/test.yaml?branch=main
[badge-pypi]: https://img.shields.io/pypi/v/cupy-stubs
[badge-python]: https://img.shields.io/pypi/pyversions/cupy-stubs
[badge-license]: https://img.shields.io/github/license/scverse/cupy-stubs

Type stubs for CuPy

## Installation

Add `cupy-stubs` to your type checking dependencies.

## Array shapes

`cupy.ndarray` has the same type parameters as `numpy.ndarray`: a shape and a dtype.
Most functions return an unknown shape (`ndarray[tuple[int, ...], ...]`),
but some parameters require a specific number of dimensions,
e.g. `cupy.bincount` only accepts 1D arrays.

Checking `arr.shape` or `arr.ndim` doesn't change the type of `arr`.
To get a 1D array, flatten it (no copy for contiguous arrays):

```python
counts = cp.bincount(labels.ravel())
```

or narrow it with a type guard:

```python
from typing import TypeIs


def is_1d(x: cp.ndarray) -> TypeIs[cp.ndarray[tuple[int]]]:
    return x.ndim == 1


if not is_1d(labels):
    msg = f"Expected 1D labels, got {labels.ndim}D"
    raise ValueError(msg)
counts = cp.bincount(labels)
```

## Release notes

See the [GitHub releases][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/scverse/cupy-stubs/issues
[tests]: https://github.com/scverse/cupy-stubs/actions/workflows/test.yaml
[github releases]: https://github.com/scverse/cupy-stubs/releases
[pypi]: https://pypi.org/project/cupy-stubs
[license]: https://github.com/scverse/cupy-stubs/blob/main/LICENSE
[venv]: https://docs.python.org/3/tutorial/venv.html
