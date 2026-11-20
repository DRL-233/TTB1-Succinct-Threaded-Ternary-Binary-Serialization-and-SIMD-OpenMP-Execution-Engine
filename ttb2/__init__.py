import struct
from typing import Any, List, Tuple, Optional
from . import _ttb2_core

SKIP_SENTINEL = 999999999

def from_json(
    obj: Any,
    skip_key: str = "status",
    skip_val: Any = 0,
    embed_values: bool = True,
    shell_id: int = 0,
) -> Tuple[bytes, Optional[List[int]]]:
    """
    Encodes a hierarchical Python/JSON tree into a TTB2 Cycloidal-Hub Succinct Packet.
    - Emits predicate-sparse cross-links ONLY on prunable nodes (~2.84 bits/node).
    - Records O(1) cycloidal geodesic spokes for top-level branches.
    - Embeds int64 leaf values directly into the wire packet for zero-copy execution.
    """
    degrees: List[int] = []
    dirs: List[int] = []
    values: List[int] = []
    jumps: List[Tuple[int, int]] = []
    spokes: List[Tuple[int, int]] = []

    def emit_node(deg: int, d_bit: int, val: int) -> int:
        idx = len(degrees)
        degrees.append(deg)
        dirs.append(d_bit)
        values.append(val)
        return idx

    def walk(node: Any, is_spoke_root: bool = False) -> None:
        spoke_start = len(degrees)

        if isinstance(node, dict):
            is_prunable = (skip_key in node and node[skip_key] == skip_val)
            items = list(node.items())
            if not items:
                emit_node(0, 0, 0)
            else:
                for i, (k, v) in enumerate(items):
                    has_next = 1 if (i + 1 < len(items)) else 0
                    deg = 2 if has_next else 1
                    cur_idx = emit_node(deg, 1, SKIP_SENTINEL if (i == 0 and is_prunable) else 0)
                    if i == 0 and is_prunable:
                        jump_slot = len(jumps)
                        jumps.append((cur_idx, 0))
                    walk(v, is_spoke_root=False)
                if is_prunable:
                    jumps[jump_slot] = (jumps[jump_slot][0], len(degrees))

        elif isinstance(node, list):
            if not node:
                emit_node(0, 0, 0)
            else:
                for i, item in enumerate(node):
                    has_next = 1 if (i + 1 < len(node)) else 0
                    emit_node(2 if has_next else 1, 0, 0)
                    walk(item, is_spoke_root=is_spoke_root)

        elif isinstance(node, bool):
            emit_node(0, 0, int(node))
        elif isinstance(node, int):
            emit_node(0, 0, int(node))
        else:
            emit_node(0, 0, 1)

        if is_spoke_root:
            spokes.append((spoke_start, len(degrees) - spoke_start))

    if isinstance(obj, dict):
        list_keys = [k for k, v in obj.items() if isinstance(v, list) and len(v) > 0]
        if list_keys:
            primary_key = max(list_keys, key=lambda k: len(obj[k]))
            for k, v in obj.items():
                if k == primary_key:
                    for item in v:
                        walk(item, is_spoke_root=True)
                else:
                    walk(v, is_spoke_root=False)
        else:
            for v in obj.values():
                walk(v, is_spoke_root=True)
    elif isinstance(obj, list):
        for item in obj:
            walk(item, is_spoke_root=True)
    else:
        walk(obj, is_spoke_root=True)

    n_nodes = len(degrees)
    n_spokes = len(spokes)
    n_jumps = len(jumps)

    deg_packed = bytearray((n_nodes + 4) // 5)
    for i in range(0, n_nodes, 5):
        acc = 0
        mult = 1
        for k in range(5):
            if i + k < n_nodes:
                acc += degrees[i + k] * mult
            mult *= 3
        deg_packed[i // 5] = acc

    dir_packed = bytearray((n_nodes + 7) // 8)
    for i, d in enumerate(dirs):
        if d:
            dir_packed[i >> 3] |= (1 << (i & 7))

    flags = 0x01 | (0x02 if embed_values else 0x00)
    header = struct.pack("<4sBBHIII", b"TTB2", flags, shell_id & 0xFF, 0, n_nodes, n_spokes, n_jumps)

    spoke_buf = bytearray()
    for off, cnt in spokes:
        spoke_buf.extend(struct.pack("<II", off, cnt))

    jump_buf = bytearray()
    for src, dst in jumps:
        jump_buf.extend(struct.pack("<II", src, dst))

    packet = bytearray(header) + spoke_buf + deg_packed + dir_packed + jump_buf
    if embed_values:
        val_buf = struct.pack(f"<{n_nodes}q", *values)
        packet.extend(val_buf)
        return bytes(packet), None

    return bytes(packet), values


class TTB2Engine:
    """Zero-copy C/OpenMP/SIMD execution engine for TTB2 Cycloidal-Hub packets."""
    def __init__(self, blob: bytes, values: Optional[List[int]] = None):
        self._ptr = _ttb2_core.compile_ttb2(blob, values)

    def sum_leaves(self) -> int:
        return _ttb2_core.sum_leaves(self._ptr)

    def query_skip(self, sentinel: int = SKIP_SENTINEL) -> int:
        return _ttb2_core.query_skip(self._ptr, sentinel)

    def broadcast_center(self) -> int:
        """Executes simultaneous one-stroke dispatch from center across all geodesic spokes."""
        return _ttb2_core.broadcast_center(self._ptr)

    def jump_sibling(self, src_spoke: int, dst_spoke: int) -> Tuple[int, int]:
        """Jumps between siblings: 1 hop if adjacent, 2 hops via dynamic center if distant."""
        return _ttb2_core.jump_sibling(self._ptr, src_spoke, dst_spoke)

    def ascent_to_center(self, node_idx: int) -> Tuple[int, int]:
        """O(1) periphery-to-center flight returning (spoke_id, hub_entry_offset)."""
        return _ttb2_core.ascent_to_center(self._ptr, node_idx)

    def close(self) -> None:
        if getattr(self, "_ptr", None):
            _ttb2_core.free_ttb2(self._ptr)
            self._ptr = None

    def __enter__(self) -> "TTB2Engine":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
