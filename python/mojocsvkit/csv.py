"""Mojo-backed stand-in for the `csv` module csvkit reads and writes through.

Upstream csvkit never talks to CPython's `csv` directly; it goes through
`agate.csv`, which is a thin wrapper (`agate/csv_py3.py`) around it. This
module keeps that shape: `reader` and `write` take the same dialect keywords
and the same `line_numbers` / `header` switches as `agate.csv_py3`.

The one structural difference, documented in the README, is that upstream is a
generator over a text stream and this is a whole-buffer transform. Callers that
only need rows get the same rows.
"""

from __future__ import annotations

import csv as _csv
from typing import Iterable, Iterator

import numpy as np

from ._lib import lib

QUOTE_MINIMAL = _csv.QUOTE_MINIMAL
QUOTE_ALL = _csv.QUOTE_ALL
QUOTE_NONNUMERIC = _csv.QUOTE_NONNUMERIC
QUOTE_NONE = _csv.QUOTE_NONE

# _csv.c's NOT_SET, which is what a `None` dialect character becomes.
NOT_SET = -1

_SCRATCH_WORDS = 8
_WRITER_WORDS = 4


def _as_bytes(source) -> bytes:
    """The bytes a text stream would hand to `csv.reader`."""
    if isinstance(source, bytes):
        return source
    if isinstance(source, str):
        return source.encode("utf-8")
    if hasattr(source, "read"):
        chunks = []
        while True:
            chunk = source.read()
            if not chunk:
                break
            chunks.append(chunk)
        source = chunks
    parts = []
    for chunk in source:
        parts.append(chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk)
    return "".join(parts).encode("utf-8")


def _untranslate_newlines(raw: bytes) -> bytes:
    """Universal newlines, which is what upstream's text-mode open does.

    `\\r\\n` and a bare `\\r` both reach `csv.reader` as `\\n`; that translation
    belongs to the text layer, so it happens here and not in the kernel.
    """
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _char(c) -> int:
    return NOT_SET if c is None else ord(c)


class Table:
    """A parsed CSV: every cell's bytes in one arena, addressed by index.

    Mirrors what `csv.reader` yields -- a list of lists of strings -- with an
    offset/length pair per cell so the kernels can work on it directly.
    """

    __slots__ = ("data", "cell_off", "cell_len", "row_cell", "row_line", "nrows",
                 "_stats")

    def __init__(self, data, cell_off, cell_len, row_cell, row_line, nrows):
        self.data = data
        self.cell_off = cell_off
        self.cell_len = cell_len
        self.row_cell = row_cell
        self.row_line = row_line
        self.nrows = nrows

    @property
    def ncells(self) -> int:
        return int(self.row_cell[self.nrows])

    def __len__(self) -> int:
        return self.nrows

    def cell(self, i: int):
        """Cell `i`, or `None` for an absent one -- upstream's `None`."""
        if i < 0:
            return None
        n = int(self.cell_len[i])
        if n < 0:
            return None
        off = int(self.cell_off[i])
        return self.data[off:off + n].tobytes().decode("utf-8")

    def row(self, r: int) -> list:
        start = int(self.row_cell[r])
        end = int(self.row_cell[r + 1])
        return [self.cell(i) for i in range(start, end)]

    def rows(self) -> Iterator[list]:
        for r in range(self.nrows):
            yield self.row(r)

    def __iter__(self) -> Iterator[list]:
        return self.rows()

    @property
    def ncols(self) -> int:
        """The widest row, which is what the header row's length is."""
        if self.nrows == 0:
            return 0
        return int(self.row_cell[1]) - int(self.row_cell[0])

    def column(self, c: int, row_start: int = 1) -> list:
        """Column `c` of the data rows, with `None` past a short row."""
        out = []
        for r in range(row_start, self.nrows):
            base = int(self.row_cell[r])
            end = int(self.row_cell[r + 1])
            out.append(self.cell(base + c) if c < end - base else None)
        return out

    @property
    def header(self) -> list:
        """The first row, or `[]` for an empty document."""
        return self.row(0) if self.nrows else []


def from_rows(rows: Iterable[Iterable]) -> Table:
    """Build a Table from Python rows, so the kernels see one layout."""
    rows = [list(r) for r in rows]
    cell_off = np.zeros(sum(len(r) for r in rows), dtype=np.int64)
    cell_len = np.zeros(cell_off.shape[0], dtype=np.int32)
    row_cell = np.zeros(len(rows) + 1, dtype=np.int64)
    blobs: list[bytes] = []
    pos = 0
    i = 0
    for r, row in enumerate(rows):
        row_cell[r] = i
        for value in row:
            cell_off[i] = pos
            if value is None:
                cell_len[i] = -1
            else:
                if isinstance(value, bytes):
                    blob = value
                else:
                    blob = str(value).encode("utf-8")
                cell_len[i] = len(blob)
                blobs.append(blob)
                pos += len(blob)
            i += 1
    row_cell[len(rows)] = i
    row_line = np.arange(1, len(rows) + 1, dtype=np.int64)
    data = np.frombuffer(b"".join(blobs), dtype=np.uint8) if blobs else np.zeros(0, dtype=np.uint8)
    return Table(data, cell_off, cell_len, row_cell, row_line, len(rows))


def reader(
    source,
    delimiter: str = ",",
    quotechar: str | None = '"',
    escapechar: str | None = None,
    doublequote: bool = True,
    skipinitialspace: bool = False,
    quoting: int = QUOTE_MINIMAL,
    strict: bool = False,
    lineterminator: str = "\r\n",
    line_numbers: bool = False,
    header: bool = True,
    **kwargs,
) -> Table:
    """`agate.csv.reader` over a whole document."""
    raw = _untranslate_newlines(_as_bytes(source))
    n = len(raw)
    src = np.frombuffer(raw, dtype=np.uint8)
    # Every cell consumes at least one input byte -- the delimiter or the
    # newline that ends it -- so n + 2 bounds both counts.
    cap = n + 2
    data = np.zeros(cap, dtype=np.uint8)
    cell_off = np.zeros(cap, dtype=np.int64)
    cell_len = np.zeros(cap, dtype=np.int32)
    row_cell = np.zeros(cap, dtype=np.int64)
    row_line = np.zeros(cap, dtype=np.int64)
    scratch = np.zeros(_SCRATCH_WORDS, dtype=np.int64)
    lt = np.frombuffer(lineterminator.encode("utf-8"), dtype=np.uint8)

    err = lib().csv_reader(
        src.ctypes.data, n,
        scratch.ctypes.data,
        data.ctypes.data, cell_off.ctypes.data, cell_len.ctypes.data,
        row_cell.ctypes.data, row_line.ctypes.data,
        _char(delimiter), _char(quotechar), _char(escapechar),
        1 if doublequote else 0, 1 if skipinitialspace else 0,
        quoting, 1 if strict else 0,
        lt.ctypes.data, len(lt),
    )
    if err == 3:
        # scratch[1]/scratch[2] still point at the field upstream's
        # PyNumber_Float choked on.
        off, ln = int(scratch[1]), int(scratch[2])
        text = data[off:off + ln].tobytes().decode("utf-8", "replace")
        raise ValueError(f"could not convert string to float: {text!r}")
    if err:
        raise _csv.Error(
            "new-line character seen in unquoted field - do you need to open "
            "the file with newline=''?"
            if err == 2
            else "illegal character after a quoted field"
        )

    nrows = int(scratch[5])
    ncells = int(scratch[4])
    table = Table(
        data[: int(scratch[6])], cell_off[:ncells], cell_len[:ncells],
        row_cell[: nrows + 1], row_line[:nrows], nrows,
    )
    if line_numbers:
        return _add_line_numbers(table, header=header)
    return table


def _add_line_numbers(table: Table, header: bool) -> Table:
    """`agate.csv_py3.Reader.__next__`'s `line_numbers` branch."""
    cells = []
    for r in range(table.nrows):
        row = table.row(r)
        row.insert(0, "line_numbers" if (header and r == 0) else str(r if header else r + 1))
        cells.append(row)
    return from_rows(cells)


def write(
    table: Table,
    delimiter: str = ",",
    quotechar: str | None = '"',
    escapechar: str | None = None,
    doublequote: bool = True,
    skipinitialspace: bool = False,
    quoting: int = QUOTE_MINIMAL,
    lineterminator: str = "\r\n",
    **kwargs,
) -> bytes:
    """Every row of `table` rendered as one CSV byte string."""
    lt = np.frombuffer(lineterminator.encode("utf-8"), dtype=np.uint8)
    state = np.zeros(_WRITER_WORDS, dtype=np.int64)
    nrows = table.nrows
    # Every cell contributes its bytes plus a quote on each side, plus one
    # delimiter and the terminator per row.
    cap = int(table.data.shape[0]) + 8 * table.ncells + 4 * nrows + 16
    rec = np.zeros(max(cap, 16), dtype=np.uint8)
    pieces: list[bytes] = []
    _keep = (lt, state, rec)
    for r in range(nrows):
        start = int(table.row_cell[r])
        end = int(table.row_cell[r + 1])
        err = lib().csv_writerow(
            _char(delimiter), _char(quotechar), _char(escapechar),
            1 if doublequote else 0, 1 if skipinitialspace else 0,
            quoting,
            lt.ctypes.data, len(lt),
            state.ctypes.data, rec.ctypes.data,
            table.data.ctypes.data,
            table.cell_off[start:end].ctypes.data,
            table.cell_len[start:end].ctypes.data,
            end - start,
        )
        if err == -1:
            raise _csv.Error("need to escape, but no escapechar set")
        if err == -2:
            raise _csv.Error("single empty field record must be quoted")
        pieces.append(rec[: int(state[0])].tobytes())
    _ = _keep
    return b"".join(pieces)


def writerow(row, **kwargs) -> bytes:
    """`csv.writer.writerow` for one row, in Mojo."""
    return write(from_rows([row]), **kwargs)