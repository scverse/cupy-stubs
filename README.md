# cupy-stubs

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/scverse/cupy-stubs/test.yaml?branch=main
[badge-docs]: https://app.readthedocs.org/projects/cupy-stubs/badge/

Type stubs for CuPy

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.12 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

We recommend managing dependencies in project-specific virtual environments to avoid dependency conflicts.
This is most convenient using package managers such as [uv][].
Choose from the options below to install cupy-stubs:

<!--
1. Add the latest release of `cupy-stubs` from [PyPI][] to your `uv` project:

   ```bash
   uv add cupy-stubs
   ```

1. Install the latest release into a [standard virtual environment][venv]:

   ```bash
   (after activating your venv)
   pip install cupy-stubs
   ```

-->

1. Install the latest development version:

   ```bash
   pip install git+https://github.com/scverse/cupy-stubs.git  # (or `uv add`)
   ```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/scverse/cupy-stubs/issues
[tests]: https://github.com/scverse/cupy-stubs/actions/workflows/test.yaml
[documentation]: https://cupy-stubs.readthedocs.io
[changelog]: https://cupy-stubs.readthedocs.io/page/changelog.html
[api documentation]: https://cupy-stubs.readthedocs.io/page/api.html
[pypi]: https://pypi.org/project/cupy-stubs
[venv]: https://docs.python.org/3/tutorial/venv.html
