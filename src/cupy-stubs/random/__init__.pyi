# SPDX-License-Identifier: MPL-2.0
from collections.abc import Sequence

import numpy as np

from ._generator_api import Generator as Generator

def default_rng(seed: int | Sequence[int] | np.random.SeedSequence | Generator | None = None) -> Generator: ...
