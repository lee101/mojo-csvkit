"""`csvkit.cleanup`: `Error`, `join_rows` and `RowChecker`.

`RowChecker.checked_rows` is upstream's control flow, unchanged. The two pieces
of it that scale with the size of the input are kernel calls: the empty-cell
tally (`row_empty_counts`) and the short-row merge (`join_rows`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._lib import lib
from .csv import Table, from_rows


@dataclass
class Error:
    line_number: int
    row: int
    msg: str


def join_rows(rows, separator: str) -> list:
    """`csvkit.cleanup.join_rows`.

    Given a series of rows, return them as a single row where the inner edge
    cells are merged.
    """
    rows = list(rows)
    fixed_row = rows[0][:]

    for row in rows[1:]:
        if len(row) == 0:
            row = [""]
        fixed_row[-1] += f"{separator}{row[0]}"
        fixed_row.extend(row[1:])

    return fixed_row


class RowChecker:
    """`csvkit.cleanup.RowChecker`: iterate rows, produce cleaned rows and
    errors."""

    def __init__(
        self,
        table: Table,
        length_mismatch: bool = False,
        empty_columns: bool = False,
        header_normalize_space: bool = False,
        join_short_rows: bool = False,
        separator: str = "\n",
        fill_short_rows: bool = False,
        fillvalue=None,
        remove_empty_columns: bool = False,
        zero_based: bool = False,
        omit_error_rows: bool = False,
    ):
        self.table = table
        self.length_mismatch = length_mismatch
        self.empty_columns = empty_columns
        self.join_short_rows = join_short_rows
        self.separator = separator
        self.fill_short_rows = fill_short_rows
        self.fillvalue = fillvalue
        self.remove_empty_columns = remove_empty_columns
        self.zero_based = zero_based
        self.omit_error_rows = omit_error_rows

        if table.nrows:
            self.column_names = table.header
            if header_normalize_space:
                self.column_names = [" ".join(c.split()) for c in self.column_names]
        else:
            self.column_names = []

        self.output_column_names = self.column_names
        self.errors: list[Error] = []

    def _empty_counts(self, len_column_names: int) -> np.ndarray:
        """`for i in range(len_column_names): if i >= len(row) or row[i] == ''`.

        The Mojo `row_empty_counts` kernel: the per-cell scan the empty-column
        checks rest on.
        """
        counts = np.zeros(max(len_column_names, 1), dtype=np.int64)
        if not (self.empty_columns or self.remove_empty_columns):
            return counts
        lib().row_empty_counts(
            self.table.cell_len.ctypes.data, self.table.row_cell.ctypes.data,
            self.table.nrows, len_column_names, 1, counts.ctypes.data,
        )
        return counts

    def checked_rows(self):
        """A generator which yields rows which are ready to write to output."""
        len_column_names = len(self.column_names)
        joinable_row_errors = []

        row_count = 0
        empty_counts = self._empty_counts(len_column_names)

        # Empty columns can only be removed once every data row has been read,
        # so buffer rows instead of yielding them.
        output_rows = [] if self.remove_empty_columns else None

        for r in range(1, self.table.nrows):
            row = self.table.row(r)
            # `self.reader.line_num - 1`
            line_number = int(self.table.row_line[r]) - 1
            row_count += 1

            length_error = Error(
                line_number, row,
                f"Expected {len_column_names} columns, found {len(row)} columns")

            if self.fill_short_rows:
                if len(row) < len_column_names:
                    row += [self.fillvalue] * (len_column_names - len(row))
            elif self.join_short_rows:
                # Don't join short rows across valid rows or with long rows.
                if len(row) >= len_column_names:
                    joinable_row_errors = []
                else:
                    joinable_row_errors.append(length_error)

                    if len(joinable_row_errors) > 1:
                        while joinable_row_errors:
                            fixed_row = _join_rows(
                                self.table, joinable_row_errors, self.separator)

                            if len(fixed_row) < len_column_names:
                                # Stop trying, if we are too short.
                                break

                            if len(fixed_row) == len_column_names:
                                row = fixed_row

                                # Remove the errors that are now fixed.
                                if self.length_mismatch:
                                    for fixed in joinable_row_errors:
                                        self.errors.remove(fixed)

                                joinable_row_errors = []
                                break

                            # Keep trying, if we are too long.
                            joinable_row_errors = joinable_row_errors[1:]

            if self.length_mismatch:
                if len(row) != len_column_names:
                    self.errors.append(length_error)

            if not self.omit_error_rows or len(row) == len_column_names:
                if self.remove_empty_columns:
                    output_rows.append(row)
                else:
                    yield row

        # File-level checks and fixes.
        empty_columns = []
        if row_count:  # Don't report all columns as empty if there are none.
            empty_columns = [i for i, count in enumerate(empty_counts) if count == row_count]

            if empty_columns and self.empty_columns:
                offset = 0 if self.zero_based else 1
                self.errors.append(
                    Error(
                        1,
                        ["" for _ in range(len_column_names)],
                        f"Empty columns named "
                        f"{', '.join(repr(self.column_names[i]) for i in empty_columns)}! "
                        f"Try: csvcut -C {','.join(str(i + offset) for i in empty_columns)}",
                    )
                )

        if self.remove_empty_columns:
            empty = set(empty_columns)
            keep = [i for i in range(len_column_names) if i not in empty]
            self.output_column_names = [self.column_names[i] for i in keep]
            for row in output_rows:
                yield [row[i] for i in keep if i < len(row)]


def _join_rows(table: Table, errors, separator: str) -> list:
    """`join_rows` over the error rows, through the Mojo `join_rows` kernel."""
    rows = [e.row for e in errors]
    flat, offsets, lengths = [], [0], []
    for row in rows:
        flat.extend(row)
        lengths.append(len(row))
        offsets.append(len(flat))
    sep = separator.encode("utf-8")
    sep_arr = np.frombuffer(sep if sep else b" ", dtype=np.uint8)
    cell_arr = np.arange(len(flat), dtype=np.int64)
    off_arr = np.array(offsets, dtype=np.int64)
    len_arr = np.array(lengths, dtype=np.int64)
    out_off = np.zeros(max(len(flat) + len(rows) * 2, 1), dtype=np.int64)
    out_len = np.zeros(out_off.shape[0], dtype=np.int32)
    out_data = np.zeros(
        int(table.data.shape[0]) + 8 * len(rows) + 16, dtype=np.uint8)
    res = np.zeros(2, dtype=np.int64)
    n, used = lib().join_rows(
        table.data.ctypes.data, table.cell_off.ctypes.data,
        table.cell_len.ctypes.data, sep_arr.ctypes.data, len(sep_arr),
        cell_arr.ctypes.data, off_arr.ctypes.data, len_arr.ctypes.data,
        len(rows), out_data.ctypes.data, out_off.ctypes.data,
        out_len.ctypes.data, res.ctypes.data,
    )
    data = out_data[:used]
    out = []
    for i in range(n):
        o = int(out_off[i])
        ln = int(out_len[i])
        if ln < 0:
            out.append(None)
        else:
            out.append(data[o:o + ln].tobytes().decode("utf-8", "replace"))
    return out