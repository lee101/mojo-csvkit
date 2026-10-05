#!/usr/bin/env python

"""csvcut: filter and truncate CSV files, like Unix `cut` but for tabular data.

`CSVCut.main` is upstream's, with the row loop replaced by one `csvcut` kernel
call over the arena.
"""

import sys

import numpy as np

from .._lib import lib
from .._tableutil import flat_gather
from ..cli import CSVKitUtility
from ..csv import Table, from_rows


class CSVCut(CSVKitUtility):
    description = (
        'Filter and truncate CSV files. Like the Unix "cut" command, but for '
        "tabular data."
    )
    override_flags = ["L", "I"]

    def add_arguments(self):
        self.argparser.add_argument(
            "-n", "--names", dest="names_only", action="store_true",
            help="Display column names and indices from the input CSV and exit.")
        self.argparser.add_argument(
            "-c", "--columns", dest="columns",
            help="A comma-separated list of column indices, names or ranges to "
                 'be extracted, e.g. "1,id,3-5". Defaults to all columns.')
        self.argparser.add_argument(
            "-C", "--not-columns", dest="not_columns",
            help="A comma-separated list of column indices, names or ranges to "
                 'be excluded, e.g. "1,id,3-5". Defaults to no columns.')
        self.argparser.add_argument(
            "-x", "--delete-empty-rows", dest="delete_empty", action="store_true",
            help="After cutting, delete rows which are completely empty.")
        self.argparser.add_argument(
            "--ignore-unknown-columns", dest="ignore_unknown_columns",
            action="store_true",
            help="Ignore identifiers in --columns (-c) that do not match a "
                 "column in the input, instead of erroring.")

    def main(self):
        if self.args.names_only:
            self.print_column_names()
            return

        if self.additional_input_expected():
            sys.stderr.write(
                "No input file or piped data provided. Waiting for standard input:\n"
            )

        rows, column_names, column_ids = \
            self.get_rows_and_column_names_and_column_ids(**self.reader_kwargs)

        out = _cut(rows, column_ids, self.args.delete_empty)
        rendered = _render(out, [column_names[i] for i in column_ids])
        self.output_file.write(rendered.decode("utf-8", "replace"))


def _cut(rows: Table, column_ids, delete_empty: bool) -> Table:
    """`out_row = [row[column_id] if column_id < len(row) else None ...]`."""
    ncols = len(column_ids)
    cols = np.array(column_ids, dtype=np.int64)
    cap = max(rows.nrows * max(ncols, 1), 1)
    cells = np.zeros(cap, dtype=np.int64)
    out_rows = np.zeros(cap + 1, dtype=np.int64)
    ncells = np.zeros(2, dtype=np.int64)
    n = lib().csvcut(
        rows.data.ctypes.data, rows.cell_off.ctypes.data,
        rows.cell_len.ctypes.data, rows.row_cell.ctypes.data,
        rows.nrows, 1, cols.ctypes.data, ncols,
        1 if delete_empty else 0,
        cells.ctypes.data, out_rows.ctypes.data, ncells.ctypes.data,
    )
    total = int(ncells[0])
    return flat_gather(rows, cells[:total], ncols) if n else _empty(ncols)


def _empty(ncols: int) -> Table:
    return from_rows([])


def _render(table: Table, header) -> bytes:
    from .. import csv as mcsv

    if table.nrows == 0 and header:
        return mcsv.write(from_rows([header]), lineterminator="\n")
    rows = [header] + [table.row(r) for r in range(table.nrows)]
    return mcsv.write(from_rows(rows), lineterminator="\n")


def launch_new_instance():
    utility = CSVCut()
    utility.run()


if __name__ == "__main__":
    launch_new_instance()