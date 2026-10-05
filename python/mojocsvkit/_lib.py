"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJOCSVKIT_LIB") or os.path.join(ROOT, "dist", "libmojo-csvkit.so")

I = ctypes.c_int64
F = ctypes.c_double

# name -> (argtypes, restype)
_SIGNATURES: dict[str, tuple[list, object]] = {
    "agg_count": ([I, I], I),
    "agg_has_nulls": ([I, I], I),
    "agg_max": ([I, I, I, I, I, I, I, I], I),
    "agg_max_length": ([I, I, I, I, I, I, I], I),
    "agg_max_precision": ([I, I, I, I, I], I),
    "agg_mean": ([I, I, I], F),
    "agg_median": ([I, I, I, I, I, I, I, I, I, I], I),
    "agg_min": ([I, I, I, I, I, I, I, I], I),
    "agg_stdev": ([I, I, I], F),
    "agg_sum": ([I, I, I], F),
    "agg_variance": ([I, I, I], F),
    "column_cast_number": ([I, I, I, I, I, I, I, I, I, I, I, I, I, I, I], I),
    "column_values_distinct": ([I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I], I),
    "csv_reader": ([I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I], I),
    "csv_writerow": ([I, I, I, I, I, I, I, I, I, I, I, I, I, I], I),
    "csvcut": ([I, I, I, I, I, I, I, I, I, I, I, I], I),
    "csvstack_headers": ([I, I, I, I, I, I, I, I], I),
    "csvstack_project": ([I, I, I, I, I, I, I, I, I, I], I),
    "join_rows": ([I, I, I, I, I, I, I, I, I, I, I, I, I], I),
    "make_default_headers": ([I, I], I),
    "match_column_identifier": ([I, I, I, I, I, I, I, I], I),
    "order_by": ([I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I], I),
    "parse_column_identifiers": ([I, I, I, I, I, I, I, I, I, I, I, I, I, I], I),
    "row_empty_counts": ([I, I, I, I, I, I], I),
    "test_row": ([I, I, I, I, I, I, I, I, I, I, I, I, I, I], I),
    "type_tester_run": ([I, I, I, I, I, I, I, I, I, I, I, I], I),
    "values_most_common": ([I, I, I, I, I, I], I),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJOCSVKIT_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJOCSVKIT_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `dist/libmojo-csvkit.so` if it is stale."""
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib", "-I", SRC,
        os.path.join(SRC, "capi.mojo"), "-o", LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def addr(a) -> int:
    return a.ctypes.data


def main() -> int:
    """`python -m mojocsvkit._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())