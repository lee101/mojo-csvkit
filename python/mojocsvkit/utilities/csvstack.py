#!/usr/bin/env python

"""csvstack: stack the rows from multiple CSV files, optionally with a grouping
value.

The header union and the per-row reordering into that union run in the Mojo
`csvstack_headers` / `csvstack_project` kernels.
"""

import os.path

import numpy as np

from .._lib import lib
from .._tableutil import concat
from ..cli import CSVKitUtility, isatty, make_default_headers
from ..csv import Table, from_rows


class CSVStack(CSVKitUtility):
    description = ("Stack up the rows from multiple CSV files, optionally "
                   "adding a grouping value.")
    override_flags = ["f", "L", "I"]

    def add_arguments(self):
        self.argparser.add_argument(
            metavar="FILE", nargs="*", dest="input_paths", default=["-"],
            help="The CSV file(s) to operate on. If omitted, will accept input "
                 "as piped data via STDIN.")
        self.argparser.add_argument(
            "-g", "--groups", dest="groups",
            help="A comma-separated list of values to add as \"grouping "
                 "factors\", one per CSV being stacked. These are added to the "
                 "output as a new column.")
        self.argparser.add_argument(
            "-n", "--group-name", dest="group_name",
            help="A name for the grouping column, e.g. \"year\". Only used when "
                 "also specifying -g.")
        self.argparser.add_argument(
            "--filenames", dest="group_by_filenames", action="store_true",
            help="Use the filename of each input file as its grouping value.")

    def main(self):
        if isatty(__import__("sys").stdin) and self.args.input_paths == ["-"]:
            __import__("sys").stderr.write(
                "No input file or piped data provided. Waiting for standard input:\n")

        from .. import csv as mcsv

        texts = []
        for path in self.args.input_paths:
            f = self._open_input_file(path)
            texts.append(f.read())
            f.close()
        tables = [mcsv.reader(t.encode("utf-8"), **self.reader_kwargs) for t in texts]

        has_groups = self.args.groups is not None or self.args.group_by_filenames
        if self.args.groups is not None and not self.args.group_by_filenames:
            groups = self.args.groups.split(",")
            if len(groups) != len(self.args.input_paths):
                self.argparser.error(
                    "The number of grouping values must be equal to the number "
                    "of CSV files being stacked.")
        else:
            groups = None

        group_name = self.args.group_name if self.args.group_name else "group"
        use_fieldnames = not self.args.no_header_row

        if use_fieldnames:
            headers, colmaps = stack_headers(tables)
            if has_groups:
                # `headers.insert(0, group_name)`: the grouping column is not
                # part of the union, so it is prepended on the way out.
                headers = [group_name] + headers
        else:
            ncols = len(tables[0].row(0)) if tables[0].nrows else 0
            headers = make_default_headers(ncols)
            if has_groups:
                headers.insert(0, group_name)
            colmaps = [None] * len(tables)

        out_rows = [headers]
        for i, table in enumerate(tables):
            group = -1
            if use_fieldnames:
                for row in project(table, colmaps[i], headers):
                    if has_groups:
                        # `row[group_name] = group`
                        row = [groups[i] if groups
                               else os.path.basename(self.args.input_paths[i])] + row
                    out_rows.append(row)

        self.output_file.write(
            mcsv.write(from_rows(out_rows), lineterminator="\n").decode("utf-8", "replace"))


def stack_headers(tables, group_name=None):
    """The ordered union of the inputs' fieldnames, and each input's map into it.

    The union is a kernel call because it compares every header against every
    header already seen, which is quadratic in the number of columns; the
    tables share one arena so a cell index means the same thing throughout.
    """
    combined, starts = concat(tables)
    head_off = np.zeros(len(tables), dtype=np.int64)
    head_len = np.zeros(len(tables), dtype=np.int64)
    for i, t in enumerate(tables):
        head_off[i] = int(combined.row_cell[starts[i]])
        head_len[i] = int(combined.row_cell[starts[i] + 1]) - head_off[i]
    out_cell = np.zeros(sum(int(x) for x in head_len) + 1, dtype=np.int64)
    scratch = np.zeros(1, dtype=np.int64)
    n = lib().csvstack_headers(
        combined.data.ctypes.data, combined.cell_off.ctypes.data,
        combined.cell_len.ctypes.data, head_off.ctypes.data,
        head_len.ctypes.data, len(tables), out_cell.ctypes.data,
        scratch.ctypes.data,
    )
    union = [combined.cell(int(c)) for c in out_cell[:n]]
    if group_name:
        union = [group_name] + union
    colmaps = []
    for t in tables:
        names = t.header
        colmaps.append([names.index(u) if u in names else -1 for u in union])
    return union, colmaps


def project(table: Table, colmap, headers) -> list:
    """`csvstack_project`: each row's cells reordered into the union order.

    A name this input does not have becomes `None`, which is what csvkit's
    DictWriter writes for a missing key.
    """
    nstack = len(colmap)
    cap = max(table.nrows * nstack, 1)
    cells = np.zeros(cap, dtype=np.int64)
    rows = np.zeros(cap + 1, dtype=np.int64)
    ncells = np.zeros(2, dtype=np.int64)
    colmap_arr = np.array(colmap, dtype=np.int64)
    n = lib().csvstack_project(
        table.cell_len.ctypes.data, table.row_cell.ctypes.data, table.nrows, 1,
        colmap_arr.ctypes.data, nstack, -1, cells.ctypes.data, rows.ctypes.data,
        ncells.ctypes.data,
    )
    bounds = [int(v) for v in rows[:n]]
    bounds.append(int(ncells[0]))
    return [
        [table.cell(int(c)) for c in cells[bounds[r]:bounds[r + 1]]]
        for r in range(n)
    ]


def launch_new_instance():
    utility = CSVStack()
    utility.run()


if __name__ == "__main__":
    launch_new_instance()