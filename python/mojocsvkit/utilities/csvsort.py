#!/usr/bin/env python

"""csvsort: sort CSV files, like Unix `sort` but for tabular data.

`table.order_by(key, reverse)` is agate's, and csvkit's `ignore_case_sort` key
function is on top of it; the sort itself is the Mojo `order_by` kernel.
"""

import numpy as np

from .._lib import lib
from .._tableutil import gather
from ..cli import CSVKitUtility, parse_column_identifiers
from ..csv import Table, from_rows
from ..table import T_NUMBER, ValueSets, cast_column, infer_types


class CSVSort(CSVKitUtility):
    description = 'Sort CSV files. Like the Unix "sort" command, but for tabular data.'

    def add_arguments(self):
        self.argparser.add_argument(
            "-n", "--names", dest="names_only", action="store_true",
            help="Display column names and indices from the input CSV and exit.")
        self.argparser.add_argument(
            "-c", "--columns", dest="columns",
            help="A comma-separated list of column indices, names or ranges to "
                 'sort by, e.g. "1,id,3-5". Defaults to all columns.')
        self.argparser.add_argument(
            "-r", "--reverse", dest="reverse", action="store_true",
            help="Sort in descending order.")
        self.argparser.add_argument(
            "-i", "--ignore-case", dest="ignore_case", action="store_true",
            help="Perform case-independent sorting.")
        self.argparser.add_argument(
            "-y", "--snifflimit", dest="sniff_limit", type=int, default=1024,
            help="Limit CSV dialect sniffing to the specified number of bytes.")
        self.argparser.add_argument(
            "-I", "--no-inference", dest="no_inference", action="store_true",
            help="Disable type inference when parsing the input.")

    def main(self):
        if self.args.names_only:
            self.print_column_names()
            return

        table = self.read_table()
        order, vs = self.get_column_types()
        column_names = table.header
        key = parse_column_identifiers(
            self.args.columns, column_names, self.get_column_offset())

        sorted_table = order_by(table, key, reverse=self.args.reverse,
                                upper=self.args.ignore_case, vs=vs, order=order)
        from .. import csv as mcsv

        self.output_file.write(
            mcsv.write(sorted_table, lineterminator="\n").decode("utf-8", "replace"))


def order_by(table: Table, key, reverse: bool = False, upper: bool = False,
             vs=None, order=None) -> Table:
    """`agate.Table.order_by(key, reverse)` over the arena.

    A key column inferred as Number is compared numerically and the rest as
    text, which is what agate's sort over the typed column does.
    """
    from .. import csv as mcsv

    vs = vs or ValueSets()
    order = order or [T_NUMBER]
    types, _keep = infer_types(table, vs, order)
    cols = np.array(key, dtype=np.int64)
    nrows = table.nrows - 1
    numeric = all(types[c] == T_NUMBER for c in key) and len(key) > 0

    # `order_by` takes one contiguous run per numeric key column, in key
    # order, so the comparison can index column i at `i * nrows + row`.
    vals = np.zeros(max(nrows * max(len(key), 1), 1), dtype=np.float64)
    if numeric and key:
        for i, c in enumerate(key):
            per = cast_column(table, c, vs, row_start=1)
            vals[i * nrows:(i + 1) * nrows] = per.vals[1:nrows + 1]

    out = np.zeros(max(nrows, 1), dtype=np.int64)
    isnull = np.zeros(max(nrows, 1), dtype=np.uint8)
    # One upper-casing slot per key column, sized to its widest cell; the
    # kernel uses a second slot at `ncols + i`.
    uppercap = np.zeros(max(len(key), 1), dtype=np.int64)
    for i, c in enumerate(key):
        widest = 0
        for r in range(1, table.nrows):
            base = int(table.row_cell[r])
            end = int(table.row_cell[r + 1])
            if c < end - base:
                widest = max(widest, int(table.cell_len[base + c]))
        uppercap[i] = max(widest, 1)
    upperbuf = np.zeros(
        2 * int(uppercap.sum()) + 8, dtype=np.uint8)

    lib().order_by(
        table.data.ctypes.data, table.cell_off.ctypes.data,
        table.cell_len.ctypes.data, table.row_cell.ctypes.data, table.nrows,
        1, cols.ctypes.data, len(cols), vals.ctypes.data,
        isnull.ctypes.data,
        1 if numeric else 0, 1 if upper else 0, 1 if reverse else 0,
        out.ctypes.data, upperbuf.ctypes.data, uppercap.ctypes.data,
    )

    # Apply the permutation to whole rows, header first.
    rows = [0] + [int(v) + 1 for v in out[:nrows]]
    cells = []
    bounds = [0]
    for r in rows:
        cells.extend(range(int(table.row_cell[r]), int(table.row_cell[r + 1])))
        bounds.append(len(cells))
    return gather(table, np.array(cells, dtype=np.int64),
                  np.array(bounds, dtype=np.int64))


def ignore_case_sort(key):
    """`csvkit.utilities.csvsort.ignore_case_sort`."""
    def inner(row):
        return tuple(
            None if row[n] is None else (row[n].upper() if isinstance(row[n], str) else row[n])
            for n in key
        )

    return inner


def launch_new_instance():
    utility = CSVSort()
    utility.run()


if __name__ == "__main__":
    launch_new_instance()