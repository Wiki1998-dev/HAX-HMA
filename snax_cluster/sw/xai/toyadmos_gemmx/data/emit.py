"""C header emission for launch schedules: initial TCDM image, launch table
and expected output regions (from the emulator)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from gemmx_sched import Launch, emit_launches


def c_bytes(name: str, data: np.ndarray, ctype: str = "uint8_t", per_line: int = 24) -> str:
    b = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
    lines = [", ".join(f"0x{x:02x}" for x in b[i:i + per_line]) for i in range(0, b.size, per_line)]
    return (f"static const {ctype} {name}[{b.size}] __attribute__((aligned(64))) = {{\n  "
            + ",\n  ".join(lines) + "\n};\n")


def emit_schedule_header(prefix: str, image: np.ndarray, launches: List[Launch],
                         checks: Sequence[Tuple[str, int, np.ndarray]]) -> str:
    """Header with <prefix>_image, <prefix>_launches and check regions.

    checks: (name, tcdm_offset, expected bytes) tuples.
    """
    out = [f"// {prefix}: {len(launches)} launches, image {image.size} B, {len(checks)} checks",
           c_bytes(f"{prefix}_image", image),
           f"#define {prefix.upper()}_IMAGE_BYTES {image.size}",
           f"#define {prefix.upper()}_N_LAUNCHES {len(launches)}",
           emit_launches(f"{prefix}_launches", launches)]
    exp = np.concatenate([np.ascontiguousarray(e).view(np.uint8).reshape(-1) for _, _, e in checks])
    out.append(c_bytes(f"{prefix}_expected", exp))
    offs, lens, pos = [], [], []
    p = 0
    for _, o, e in checks:
        n = np.ascontiguousarray(e).view(np.uint8).size
        offs.append(o); lens.append(n); pos.append(p); p += n
    out.append(f"#define {prefix.upper()}_N_CHECKS {len(checks)}")
    out.append(f"static const char *{prefix}_check_names[] = {{"
               + ", ".join(f'"{n}"' for n, _, _ in checks) + "};")
    for nm, arr in (("off", offs), ("len", lens), ("pos", pos)):
        out.append(f"static const int32_t {prefix}_check_{nm}[] = {{"
                   + ", ".join(map(str, arr)) + "};")
    return "\n".join(out) + "\n"
