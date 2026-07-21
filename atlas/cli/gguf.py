"""GGUF header metadata reading shared by the CLI commands.

Model-agnostic on purpose: callers key off `general.architecture` and
per-arch keys from the header, never a model name.
"""

import struct


def read_gguf_kv(f):
    """Yield (key, value) for each metadata pair in a GGUF file. Large arrays
    we never need (tokenizer tables) are not materialized: fixed-size scalar
    arrays are seeked over in one hop, string arrays are walked by their
    length prefixes without decoding."""
    def u32():
        return struct.unpack("<I", f.read(4))[0]

    def u64():
        return struct.unpack("<Q", f.read(8))[0]

    def rd_str():
        return f.read(u64()).decode("utf-8", "replace")

    SCALAR = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
              4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
              10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}

    def rd_val(t):
        if t == 8:
            return rd_str()
        if t == 9:
            et = u32()
            n = u64()
            if et == 8:
                if n > 4096:
                    for _ in range(n):       # tokenizer tables: skip each
                        f.seek(u64(), 1)     # string by its length prefix
                    return []
                return [rd_str() for _ in range(n)]
            if et == 9:
                return [rd_val(9) for _ in range(n)]
            fmt, size = SCALAR[et]
            if n > 4096:
                f.seek(n * size, 1)   # bulk data we never need (e.g. scores)
                return []
            return [struct.unpack(fmt, f.read(size))[0] for _ in range(n)]
        fmt, size = SCALAR[t]
        return struct.unpack(fmt, f.read(size))[0]

    magic = f.read(4)
    if magic != b"GGUF":
        raise ValueError("not a GGUF file")
    version = u32()
    if version not in (2, 3):
        # v1 used u32 counts/lengths; misreading them as u64 produces
        # absurd allocation sizes, so refuse cleanly instead.
        raise ValueError(f"unsupported GGUF version {version}")
    u64()                      # tensor count
    n_kv = u64()
    for _ in range(n_kv):
        key = rd_str()
        t = u32()
        yield key, rd_val(t)
