import struct
import ctypes
import importlib.util

__version__ = "0.1.0"
__all__ = ["encode", "decode", "from_json", "TTB1Engine"]

_spec = importlib.util.find_spec("ttb1._ttb1_core")
if _spec is None or _spec.origin is None:
    raise ImportError("Could not locate compiled ttb1._ttb1_core C extension.")

_lib = ctypes.CDLL(_spec.origin)
_lib.ttb1_compile.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_int32)]
_lib.ttb1_compile.restype = ctypes.c_void_p
_lib.ttb1_sum_leaves.argtypes = [ctypes.c_void_p]
_lib.ttb1_sum_leaves.restype = ctypes.c_int64
_lib.ttb1_exec_fused.argtypes = [ctypes.c_void_p]
_lib.ttb1_exec_fused.restype = ctypes.c_int64
_lib.ttb1_query_skip.argtypes = [ctypes.c_void_p, ctypes.c_int32]
_lib.ttb1_query_skip.restype = ctypes.c_int64
_lib.ttb1_free.argtypes = [ctypes.c_void_p]


def encode(degrees, dirs, threads=None):
    threads = threads or {}
    n_nodes = len(degrees)
    n_threads = len(threads)

    deg_bytes = bytearray(
        sum(t * (3 ** (len(degrees[i:i + 5]) - 1 - p)) for p, t in enumerate(degrees[i:i + 5]))
        for i in range(0, n_nodes, 5)
    )
    dir_bytes = bytearray(
        sum(b << (7 - p) for p, b in enumerate(dirs[i:i + 8]))
        for i in range(0, n_nodes, 8)
    )
    thr_bytes = bytearray()
    for src, dst in sorted(threads.items()):
        thr_bytes.extend(struct.pack("<HH", src, dst))

    return bytes(struct.pack("<4sII", b"TTB1", n_nodes, n_threads) + deg_bytes + dir_bytes + thr_bytes)


def decode(blob):
    magic, n_nodes, n_threads = struct.unpack("<4sII", blob[:12])
    if magic != b"TTB1":
        raise ValueError("Invalid TTB1 magic header.")

    deg_len = (n_nodes + 4) // 5
    dir_len = (n_nodes + 7) // 8
    deg_raw = blob[12 : 12 + deg_len]
    dir_raw = blob[12 + deg_len : 12 + deg_len + dir_len]
    thr_raw = blob[12 + deg_len + dir_len :]

    degrees = []
    for idx_b, val in enumerate(deg_raw):
        count = min(5, n_nodes - idx_b * 5)
        chunk = []
        for _ in range(count):
            chunk.append(val % 3)
            val //= 3
        degrees.extend(reversed(chunk))

    dirs = [(dir_raw[i >> 3] >> (7 - (i & 7))) & 1 for i in range(n_nodes)]
    threads = dict(struct.unpack("<HH", thr_raw[i : i + 4]) for i in range(0, len(thr_raw), 4))
    return degrees, dirs, threads


def from_json(obj, skip_key="status", skip_val=0):
    degrees, dirs, values, threads = [], [], [], {}

    def visit(item, tag=0):
        idx = len(degrees)
        if isinstance(item, dict):
            degrees.append(2)
            dirs.append(1)
            values.append(tag)
            items = list(item.items())
            for k_i, (k, v) in enumerate(items):
                child_tag = 999 if (k == skip_key and v == skip_val) else (hash(k) & 0xFFFF)
                if k == skip_key and v == skip_val:
                    values[idx] = 999
                if k_i < len(items) - 1:
                    degrees.append(2)
                    dirs.append(1)
                    values.append(child_tag)
                    visit(v, child_tag)
                else:
                    visit(v, child_tag)
            if idx < 65535 and len(degrees) < 65535:
                threads[idx] = len(degrees)
        elif isinstance(item, list):
            degrees.append(2)
            dirs.append(1)
            values.append(tag)
            for elem in item:
                visit(elem, tag)
            if idx < 65535 and len(degrees) < 65535:
                threads[idx] = len(degrees)
        else:
            degrees.append(0)
            dirs.append(0)
            values.append(int(item) if isinstance(item, (int, bool)) else 1)

    visit(obj)
    return encode(degrees, dirs, threads), values


class TTB1Engine:
    def __init__(self, blob, values=None):
        n_nodes = struct.unpack("<I", blob[4:8])[0]
        c_vals = (ctypes.c_int32 * n_nodes)(*values) if values is not None else None
        self._ptr = _lib.ttb1_compile(blob, c_vals)

    def sum_leaves(self):
        return _lib.ttb1_sum_leaves(self._ptr)

    def exec_fused(self):
        return _lib.ttb1_exec_fused(self._ptr)

    def query_skip(self, skip_tag=999):
        return _lib.ttb1_query_skip(self._ptr, skip_tag)

    def close(self):
        if getattr(self, "_ptr", None):
            _lib.ttb1_free(self._ptr)
            self._ptr = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()
