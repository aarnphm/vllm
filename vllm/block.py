"""Token blocks."""
import dataclasses
from typing import TYPE_CHECKING, Iterator, List, Optional

from vllm.utils import Device

DEFAULT_LAST_ACCESSED_TIME: float = -1

@dataclasses.dataclass
class PhysicalTokenBlock:
    """Represents the state of a block in the KV cache."""
    device: Device
    block_number: int
    block_size: int
    block_hash: int
    num_hashed_tokens: int
    ref_count: int = dataclasses.field(default=0)
    last_accessed: float = dataclasses.field(default=DEFAULT_LAST_ACCESSED_TIME)
    computed: bool = False


class BlockTable:
    """Holds a list of blocks with caching of their associated block_ids
    """

    def __init__(self, blocks: Optional[List[PhysicalTokenBlock]] = None):
        self._blocks: List[PhysicalTokenBlock] = []
        self._block_ids: List[int] = []

        if blocks is not None:
            for block in blocks: self.append(block)

    def __len__(self) -> int: return len(self._blocks)
    def __getitem__(self, key): return self._blocks[key]

    if TYPE_CHECKING:
        def __iter__(self) -> Iterator[PhysicalTokenBlock]: ...

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            blocks = value
            self._blocks[key] = blocks
            self._block_ids[key] = [b.block_number for b in blocks]
        else:
            block = value
            self._blocks[key] = block
            self._block_ids[key] = block.block_number

    def reset(self): self._blocks = []; self._block_ids = [];
    def append(self, block: PhysicalTokenBlock) -> None: self._blocks.append(block); self._block_ids.append(block.block_number);
    def copy(self) -> "BlockTable": return BlockTable(self._blocks)
    def list(self) -> List[PhysicalTokenBlock]: return self._blocks
    def ids(self) -> List[int]: return self._block_ids
