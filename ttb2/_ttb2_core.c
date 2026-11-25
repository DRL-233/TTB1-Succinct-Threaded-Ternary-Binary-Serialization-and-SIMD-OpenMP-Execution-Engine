"""
TTB v0.3.0 (TTB3-Bridge): Cycloidal Hub-Entangled Succinct Ternary Execution Engine
& Lean-4-Verified OZIP Z3-Triality Contraction Runtime.
"""
import struct
import _ttb2_core

__version__ = "0.3.0"


def from_json(workload, embed_values=True, embed_width="auto"):
    """
    Encodes a hierarchical Python dict/list tree into a TTB2/TTB3 wire packet.
    - embed_width="auto" (v0.3.0 default): packs leaf scalars at minimal int8/16/32/64 width
      with container elision (flags=0x04), inflated on load via AVX2 SIMD sign-extension.
    - embed_width=64: legacy v0.2.0 raw 64-bit layout (flags=0x02).
    """
    degs, dirs, leaves, all_vals, spokes = [], [], [], [], []

    def walk(obj):
        if isinstance(obj, dict):
            items = list(obj.values())
            if not items:
                degs.append(0)
                dirs.append(0)
                leaves.append(0)
                all_vals.append(0)
                return
            for k_i, v in enumerate(items):
                if k_i < len(items) - 1:
                    degs.append(2)
                    dirs.append(0)
                    all_vals.append(0)
                    walk(v)
                else:
                    walk(v)
        elif isinstance(obj, list):
            for e_i, elem in enumerate(obj):
                if e_i < len(obj) - 1:
                    degs.append(2)
                    dirs.append(0)
                    all_vals.append(0)
                s_start = len(degs)
                walk(elem)
                spokes.append((s_start, len(degs) - s_start))
        else:
            val = int(obj)
            degs.append(0)
            dirs.append(0)
            leaves.append(val)
            all_vals.append(val)

    walk(workload)
    n_nodes = len(degs)
    n_spokes = len(spokes)

    deg_bytes = bytearray((n_nodes + 4) // 5)
    for i in range(0, n_nodes, 5):
        acc = 0
        for k in range(5):
            if i + k < n_nodes:
                acc += degs[i + k] * (3 ** k)
        deg_bytes[i // 5] = acc

    dir_bytes = bytearray((n_nodes + 7) // 8)
    for i, d in enumerate(dirs):
        if d:
            dir_bytes[i >> 3] |= 1 << (i & 7)

    spoke_bytes = b"".join(struct.pack("<II", off, cnt) for off, cnt in spokes)

    if not embed_values:
        header = b"TTB2" + bytes([0x00, 0, 0, 0]) + struct.pack("<III", n_nodes, n_spokes, 0)
        return header + spoke_bytes + bytes(deg_bytes) + bytes(dir_bytes), all_vals

    if embed_width == 64:
        header = b"TTB2" + bytes([0x02, 8, 0, 0]) + struct.pack("<III", n_nodes, n_spokes, 0)
        val_bytes = b"".join(struct.pack("<q", v) for v in all_vals)
        return header + spoke_bytes + bytes(deg_bytes) + bytes(dir_bytes) + val_bytes, all_vals

    max_abs = max((abs(v) for v in leaves), default=0)
    if max_abs <= 127:
        w_code, fmt = 1, "<b"
    elif max_abs <= 32767:
        w_code, fmt = 2, "<h"
    elif max_abs <= 2147483647:
        w_code, fmt = 4, "<i"
    else:
        w_code, fmt = 8, "<q"

    header = b"TTB2" + bytes([0x04, w_code, 0, 0]) + struct.pack("<III", n_nodes, n_spokes, 0)
    val_bytes = b"".join(struct.pack(fmt, v) for v in leaves)
    return header + spoke_bytes + bytes(deg_bytes) + bytes(dir_bytes) + val_bytes, all_vals


def ozip_psi1(s: int, g: int, i: int) -> int:
    """OZIP v2.0.0 mixed-radix coordinate bijection: psi1(s, g, i) = 9s + 3g + i in Fin 18."""
    return _ttb2_core.ozip_psi1(s, g, i)


def ozip_contract_3x3x2(A_flat, B_flat):
    """Contracts <3, 3, 2> Z3-triality sectors in a single native C/AVX2 pass."""
    a_bytes = struct.pack("<9q", *[int(x) for x in A_flat])
    b_bytes = struct.pack("<6q", *[int(x) for x in B_flat])
    c_bytes = _ttb2_core.ozip_contract_3x3x2(a_bytes, b_bytes)
    return list(struct.unpack("<6q", c_bytes))


class TTB2Engine:
    """Context-managed handle to the compiled C/AVX2/OpenMP cycloidal manifold."""

    def __init__(self, packet: bytes, values=None, skip_sentinel: int = 999999999):
        self._skip = skip_sentinel
        if values is not None:
            self._handle = _ttb2_core.compile_ttb2(packet, values, skip_sentinel)
        else:
            self._handle = _ttb2_core.compile_ttb2(packet, None, skip_sentinel)

    def sum_leaves(self) -> int:
        return _ttb2_core.sum_leaves(self._handle)

    def query_skip(self, skip_sentinel=None) -> int:
        s = self._skip if skip_sentinel is None else skip_sentinel
        return _ttb2_core.query_skip(self._handle, s)

    def broadcast_center(self) -> int:
        return _ttb2_core.broadcast_center(self._handle)

    def jump_sibling(self, src_spoke: int, dst_spoke: int):
        return _ttb2_core.jump_sibling(self._handle, src_spoke, dst_spoke)

    def ascent_to_center(self, node_idx: int):
        return _ttb2_core.ascent_to_center(self._handle, node_idx)

    def update_leaf(self, node_idx: int, new_val: int, skip_sentinel=None) -> bool:
        s = self._skip if skip_sentinel is None else skip_sentinel
        return _ttb2_core.update_leaf(self._handle, node_idx, new_val, s)

    def set_spoke_status(self, spoke_idx: int, active: bool) -> bool:
        return _ttb2_core.set_spoke_status(self._handle, spoke_idx, 1 if active else 0)

    def close(self):
        if getattr(self, "_handle", None):
            _ttb2_core.free_ttb2(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
