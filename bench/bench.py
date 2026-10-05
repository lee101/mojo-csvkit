"""mojo-csvkit against upstream csvkit on the same input.

    pixi run bench

Every row is one utility on one dataset, run through both implementations with
the same flags and the same bytes. The numbers are the best of three
iterations. Where upstream wins -- it is mature, and its type inference and
aggregation live in C extensions with a long history -- the table says so.
"""

from __future__ import annotations

import io
import math
import os
import sys
import time

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

from csvkit.utilities.csvcut import CSVCut as UpCut  # noqa: E402
from csvkit.utilities.csvformat import CSVFormat as UpFormat  # noqa: E402
from csvkit.utilities.csvgrep import CSVGrep as UpGrep  # noqa: E402
from csvkit.utilities.csvsort import CSVSort as UpSort  # noqa: E402
from csvkit.utilities.csvstack import CSVStack as UpStack  # noqa: E402
from csvkit.utilities.csvstat import CSVStat as UpStat  # noqa: E402

from mojocsvkit.utilities.csvcut import CSVCut  # noqa: E402
from mojocsvkit.utilities.csvformat import CSVFormat  # noqa: E402
from mojocsvkit.utilities.csvgrep import CSVGrep  # noqa: E402
from mojocsvkit.utilities.csvsort import CSVSort  # noqa: E402
from mojocsvkit.utilities.csvstack import CSVStack  # noqa: E402
from mojocsvkit.utilities.csvstat import CSVStat  # noqa: E402

import numpy as np  # noqa: E402


def timeit(fn, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def run(cls, argv, data, paths=()):
    out, err = io.StringIO(), io.StringIO()
    u = cls(list(argv) + list(paths), output_file=out, error_file=err)
    u.input_file = io.StringIO(data)
    try:
        u.main()
    except SystemExit:
        pass
    return out.getvalue()


def table(nrows: int, ncols: int, seed: int = 0) -> str:
    """A table with the column mix csvstat cares about: text, numbers, dates."""
    rng = np.random.default_rng(seed)
    header = [f"col{i}" for i in range(ncols)]
    lines = [",".join(header)]
    for r in range(nrows):
        cells = []
        for c in range(ncols):
            if c % 5 == 0:
                cells.append(f"text{r}_{c}")
            elif c % 5 == 1:
                cells.append(str(int(rng.integers(0, 1000))))
            elif c % 5 == 2:
                cells.append(f"{rng.normal():.4f}")
            elif c % 5 == 3:
                cells.append("yes" if r % 2 else "no")
            else:
                cells.append(str(int(rng.integers(0, 1_000_000))))
        lines.append(",".join(cells))
    return "\n".join(lines) + "\n"


SMALL = table(2_000, 10)
MEDIUM = table(100_000, 10)
WIDE = table(20_000, 60)
STACK_A = table(50_000, 6, seed=1)
STACK_B = table(50_000, 6, seed=2)

CASES = [
    ("csvcut -c 1,3,5 (100k x 10)", (CSVCut, UpCut), ["-c", "1,3,5"], MEDIUM, ()),
    ("csvgrep -c 2 -m 42 (100k x 10)", (CSVGrep, UpGrep), ["-c", "2", "-m", "42"],
     MEDIUM, ()),
    ("csvsort -c 3 (100k x 10)", (CSVSort, UpSort), ["-c", "3"], MEDIUM, ()),
    ("csvsort -c 3 -r (100k x 10)", (CSVSort, UpSort), ["-c", "3", "-r"], MEDIUM, ()),
    ("csvstat --csv (100k x 10)", (CSVStat, UpStat), ["--csv"], MEDIUM, ()),
    ("csvstat --mean (100k x 10)", (CSVStat, UpStat), ["--mean"], MEDIUM, ()),
    ("csvstat --median (100k x 10)", (CSVStat, UpStat), ["--median"], MEDIUM, ()),
    ("csvstat --stdev (100k x 10)", (CSVStat, UpStat), ["--stdev"], MEDIUM, ()),
    ("csvstat --freq (100k x 10)", (CSVStat, UpStat), ["--freq"], MEDIUM, ()),
    ("csvstat --csv (20k x 60)", (CSVStat, UpStat), ["--csv"], WIDE, ()),
    ("csvformat -T (100k x 10)", (CSVFormat, UpFormat), ["-T"], MEDIUM, ()),
    ("csvformat -U 1 (100k x 10)", (CSVFormat, UpFormat), ["-U", "1"], MEDIUM, ()),
    ("csvstack (2 x 50k x 6)", (CSVStack, UpStack), [], "", ("/tmp/_bench_a.csv",
                                                           "/tmp/_bench_b.csv")),
]


def main() -> None:
    with open("/tmp/_bench_a.csv", "w") as f:
        f.write(STACK_A)
    with open("/tmp/_bench_b.csv", "w") as f:
        f.write(STACK_B)

    print(f"{'case':<34}{'mojo-csvkit':>13}{'csvkit':>12}{'ratio':>9}")
    print("-" * 68)
    for name, (ours, theirs), argv, data, paths in CASES:
        a = timeit(lambda: run(ours, argv, data, paths))
        b = timeit(lambda: run(theirs, argv, data, paths))
        flag = "faster" if a < b else "slower"
        print(f"{name:<34}{a * 1e3:>10.1f}ms{b * 1e3:>9.1f}ms{b / a:>8.2f}x  {flag}")


if __name__ == "__main__":
    main()