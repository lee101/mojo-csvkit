# mojo-csvkit

A Mojo port of [csvkit](https://csvkit.readthedocs.io/)'s CSV parse and
transform core. The dialect state machines, the type inference and the
column statistics run in Mojo behind a ctypes boundary; the Python layer keeps
upstream's module layout, class names, flags and output byte for byte, so a
covered utility is a drop-in replacement for the upstream one.

## What this is

csvkit is a suite of command-line tools for tabular text. Almost all of its time
goes into two things: turning bytes into rows according to a CSV dialect, and
summing up what is in a column. Both are ports here:

- **`src/ported.mojo`** is a transcription of the CSV reader state machine
  (`Modules/_csv.c`, the one `agate.csv` wraps), the CSV writer
  (`join_append_data`), the `agate` data types csvkit's type inference needs
  (`Number.cast`, `Boolean.cast`, `Text.cast`, `TypeTester.run`), the
  `agate` aggregations csvstat names in its `OPERATIONS` table, and the
  per-row kernels behind csvcut, csvgrep, csvsort, csvstack and csvclean's
  checks -- laid out in upstream's source order, one section per upstream
  module.
- **`src/capi.mojo`** is the C ABI. Buffers cross as `Int` addresses; the
  wrappers rebuild pointers and do nothing else.
- **`python/mojocsvkit/`** mirrors `csvkit/`: `cli.py` (`CSVKitUtility`,
  `parse_column_identifiers`, `match_column_identifier`, `make_default_headers`),
  `cleanup.py` (`RowChecker`, `join_rows`, `Error`), `grep.py`
  (`FilteringCSVReader`, `standardize_patterns`, `pattern_as_function`), and
  `utilities/` with `csvcut`, `csvclean`, `csvformat`, `csvgrep`, `csvsort`,
  `csvstack`, `csvstat`.

Upstream csvkit is a test-time dependency, not a runtime one.

## Coverage

Covered, and byte-identical to upstream on the test suite:

| upstream | what runs in Mojo |
| --- | --- |
| the `csv` reader dialect | `parse_process_char` and all nine parser states, every dialect flag |
| the `csv` writer | `join_append_data`, every quoting mode, `doublequote`, `escapechar`, `lineterminator` |
| `cli.parse_column_identifiers`, `match_column_identifier` | identifier resolution, `-` and `:` ranges, `-C` exclusions |
| `cli.make_default_headers` | `agate.utils.letter_name` |
| `csvcut` | the per-row column gather, `-x` |
| `csvgrep -m/--match` | `FilteringCSVReader.test_row`, `-a`, `-i` |
| `csvsort` | `Table.order_by`, `-r`, `-i`, numeric and text keys, null ordering |
| `csvstack` | the header union and the per-row reorder into it, `-g`, `-n` |
| `csvclean` | the empty-cell tally, the length check, `--fill-short-rows`, `--join-short-rows` |
| `csvstat` | type inference, and every aggregation in its `OPERATIONS` table |
| `csvformat` | the writer under a different output dialect |

Not covered:

- **`csvsql`, `sql2csv`** -- a SQL engine over SQLite. Not compute-bound in the
  sense this port targets, and not a drop-in here.
- **`csvlook`** -- terminal rendering.
- **`in2csv`** -- Excel, DBF and JSON input.
- **`csvjson`** -- JSON serialisation only; nothing numeric for a kernel to do.
- **`csvjoin`** -- the hash join kernel was written and removed: it did not
  converge under the time available, and shipping an unverified join would be
  worse than shipping none. `agate.Table.join` is therefore not ported.
- **`csvgrep -r/--regex` and `-f/--file`** -- upstream compiles these to Python
  callables (`pattern_as_function`), so they stay on the Python side. `-m`
  substring matching, which is the same code path, runs in Mojo.
- **agate's `Date`, `DateTime` and `TimeDelta`** -- see below.

### Documented divergences

- **`Date`, `DateTime`, `TimeDelta`.** agate parses these with `parsedatetime`,
  a natural-language date parser. It is not ported, so those three types are
  not in the type tester's candidate list and a column they would have claimed
  infers as `Text` instead. `get_column_types` otherwise builds csvkit's list in
  csvkit's order. ISO-8601 dates are therefore `Text` here and `Date` upstream;
  everything else about `csvstat` matches.
- **`Decimal` versus `float64`.** agate casts numbers to `decimal.Decimal` and
  aggregates exactly. This port parses to `float64`, and `Sum` accumulates with
  Neumaier compensation, so `sum`, `mean` and `stdev` agree to about one ulp per
  column rather than exactly; the JSON test compares them to a relative
  tolerance for that reason. `max_precision` is exact: it reads the digit count
  and exponent of `Decimal.normalize()`, which the same parse produces, and
  those match agate value for value.
- **`get_unique` and `get_freq` run on the host.** The Mojo distinct-value
  table was written and removed: it did not collapse duplicate values, and
  `csvstat`'s `--unique` and `--freq` are wrong when it does not. Counting now
  happens in Python over the cast column, which is exact. The `values_most_common`
  kernel is still in `ported.mojo` but is not called.
- **The reader is a whole-buffer transform, not a generator.** The state machine
  is upstream's; what changed is that it writes into caller-allocated arrays
  instead of appending to a growing Python list, and that the input arrives as
  bytes rather than an iterator of lines. Universal newline translation, which
  upstream's text-mode open performs, happens on the Python side before the
  kernel sees the bytes.
- **Newlines.** Upstream's text-mode read turns `\r\n` and a bare `\r` into `\n`
  before `csv.reader` sees them; the same translation is applied here, in
  `mojocsvkit.csv._untranslate_newlines`.

## Install and run

```bash
pixi install
pixi run build     # mojo build --emit shared-lib -> dist/libmojo-csvkit.so
pixi run test      # pytest
pixi run bench     # benchmarks, under a machine-wide flock
```

`pixi run build` produces `dist/libmojo-csvkit.so`. `python/mojocsvkit/_lib.py`
rebuilds it on import when a `.mojo` source is newer, so `PYTHONPATH=python`
is enough to use the package from a checkout.

## Usage

```python
import io
from mojocsvkit.utilities.csvstat import CSVStat

out = io.StringIO()
util = CSVStat(["--csv"], output_file=out)
util.input_file = io.StringIO(
    "name,score\nada,10\ngrace,9.5\nalan,10\n"
)
util.main()
print(out.getvalue())
```

```text
column_id,column_name,type,nulls,nonnulls,unique,min,max,sum,mean,median,stdev,len,maxprecision,freq
1,name,Text,False,3,3,,,,,,,5,,"ada, grace, alan"
2,score,Number,False,3,3,9.5,10,29.5,9.833,10,0.289,,1,"10, 9.5, 10"
```

The same program against upstream csvkit produces the same bytes:

```python
from csvkit.utilities.csvstat import CSVStat
```

`parse_column_identifiers` and the CSV layer are usable on their own:

```python
from mojocsvkit import csv
from mojocsvkit.cli import parse_column_identifiers

table = csv.reader(b"a,b,c\n1,2,3\n", delimiter=",")
parse_column_identifiers("1,3", table.header)      # -> [0, 2]
csv.write(table, lineterminator="\n")               # -> b'a,b,c\n1,2,3\n'
```

## Benchmarks

Measured with `pixi run bench` on the reference machine below, best of three
iterations per case, both implementations given the same bytes and the same
flags.

Machine: Linux 6.8.0-142-generic, x86_64, Intel Xeon E5-2697 v4 @ 2.30 GHz,
CPython 3.13.14, Mojo 1.2.0.dev2026092605, csvkit 2.2.0.

| case | mojo-csvkit | csvkit | ratio | |
| --- | --- | --- | --- | --- |
| csvcut -c 1,3,5 (100k x 10) | 3992.4ms | 360.8ms | 0.09x | slower |
| csvgrep -c 2 -m 42 (100k x 10) | 284.3ms | 228.9ms | 0.81x | slower |
| csvsort -c 3 (100k x 10) | 7283.2ms | 8495.0ms | 1.17x | faster |
| csvsort -c 3 -r (100k x 10) | 7048.2ms | 8509.4ms | 1.21x | faster |
| csvstat --csv (100k x 10) | 12032.2ms | 20940.6ms | 1.74x | faster |
| csvstat --mean (100k x 10) | 3781.5ms | 7999.3ms | 2.12x | faster |
| csvstat --median (100k x 10) | 4256.8ms | 8598.1ms | 2.02x | faster |
| csvstat --stdev (100k x 10) | 3824.7ms | 9154.7ms | 2.39x | faster |
| csvstat --freq (100k x 10) | 10553.6ms | 7843.0ms | 0.74x | slower |
| csvstat --csv (20k x 60) | 10763.2ms | 25550.9ms | 2.37x | faster |
| csvformat -T (100k x 10) | 5015.9ms | 587.4ms | 0.12x | slower |
| csvformat -U 1 (100k x 10) | 4915.9ms | 616.1ms | 0.13x | slower |
| csvstack (2 x 50k x 6) | 5670.9ms | 977.7ms | 0.17x | slower |

Read this honestly: the win is in the numeric passes -- type inference and the
aggregations, where csvstat is 1.7x to 2.4x faster, and the sort, which is
1.2x faster. The losses are real and are structural rather than a tuning
miss. csvcut, csvformat and csvstack are dominated by materialising a Python
`str` per cell: upstream writes into `csv.writer` with values the caller already
has as strings, while this port has to decode a cell out of the arena before it
can hand it over. The kernel is faster; the object churn around it is not, and
that is where the time goes. `--freq` is slower for a second and simpler reason: its
frequency table is counted on the host (see the divergences above), so it is
Python counting against a C extension.

## How it works

**Memory layout.** A parsed table is one flat arena:

```text
data      UInt8[nbytes]   the decoded bytes of every cell, back to back
cell_off  Int64[ncells]   start of cell i in `data`
cell_len  Int32[ncells]   length of cell i; -1 marks an absent cell, which is
                          upstream's `None` / out-of-range row index
row_cell  Int64[nrows+1]  index of the first cell of row r, with a trailing
                          bound, so row r is row_cell[r] .. row_cell[r+1]-1
```

Ragged rows need no padding: a short row simply has fewer cells, which is
exactly what `row[column_id] if column_id < len(row) else None` in csvcut
means. Addressing cells by (offset, length) rather than by value is what lets
the string-shaped kernels -- identifier resolution, the substring test, the
distinct-value table, the sort comparator -- run without a Python object per
cell.

**FFI strategy.** `@export` rejects parametric functions and a pointer with an
inferred origin is parametric, so every buffer crosses as an `Int` address and
`src/capi.mojo` rebuilds the pointer inside the wrapper. Nothing on the Mojo
side allocates, so nothing can leak; the caller owns every buffer including
scratch. A `struct` field may not have an `AnyOrigin` type in this dialect, so
the parser and writer state that upstream keeps in `ReaderObj` / `WriterObj`
lives in a caller-owned `Int64` scratch block, and `Dialect` carries the
lineterminator as an address rather than a `String`.

**Why the state machine reads the way it does.** `parse_process_char` is a
transcription of `Modules/_csv.c`: same nine states, same order, same
conditions. The only departure is `parse_save_field`, which commits the
accumulated field to the arena instead of appending a Python object to a list.
The writer is the same story, including `join_append_data`'s `copy_phase` flag
and its measure-then-copy structure.