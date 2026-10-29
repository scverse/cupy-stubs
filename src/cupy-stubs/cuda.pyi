# SPDX-License-Identifier: MPL-2.0
from typing_extensions import disjoint_base

class Stream:
    def __init__(
        self, null: bool = False, non_blocking: bool = False, ptds: bool = False, priority: int | None = None
    ) -> None: ...

@disjoint_base
class MemoryPointer: ...
