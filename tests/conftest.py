# SPDX-License-Identifier: MPL-2.0
import sys

# stubs use type parameter defaults, which `ast` can only parse on Python ≥3.13
collect_ignore = [] if sys.version_info >= (3, 13) else ["test_meta.py"]
