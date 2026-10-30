# SPDX-License-Identifier: MPL-2.0
from typing_extensions import disjoint_base

class _BaseStream:
    ptr: int

class Stream(_BaseStream):
    def __init__(
        self, null: bool = False, non_blocking: bool = False, ptds: bool = False, priority: int | None = None
    ) -> None: ...

@disjoint_base
class MemoryPointer: ...

@disjoint_base
class Device:
    def __init__(self, device: int | None = None) -> None: ...
    @property
    def attributes(self) -> dict[str, int]: ...

def get_current_stream(device_id: int = -1) -> _BaseStream: ...
