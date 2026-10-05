#!/usr/bin/env python

"""csvgrep: search CSV files, like Unix `grep` but for tabular data.

`FilteringCSVReader.test_row` is upstream's; the per-row scan for `-m/--match`
patterns runs in the Mojo `test_row` kernel. `-r/--regex` and `-f/--file` are
Python-side, as upstream's `pattern_as_function` builds a Python callable for
them; see the README's coverage section.
"""

import re
import sys

import numpy as np

from .._lib import lib
from .._tableutil import gather
from ..cli import CSVKitUtility
from ..csv import Table, from_rows


class FilteringCSVReader:
    """`csvkit.grep.FilteringCSVReader`, for Python-side patterns."""

    returned_header = False
    column_names = None

    def __init__(self, reader, patterns, header=True, any_match=False, inverse=False):
        self.reader = reader
        self.header = header
        if self.header:
            self.column_names = next(reader)
        self.any_match = any_match
        self.inverse = inverse
        self.patterns = standardize_patterns(self.column_names, patterns)

    def __iter__(self):
        return self

    def __next__(self):
        if self.column_names and not self.returned_header:
            self.returned_header = True
            return self.column_names
        while True:
            row = next(self.reader)
            if self.test_row(row):
                return row
        raise StopIteration()

    def test_row(self, row):
        for idx, test in self.patterns.items():
            try:
                value = row[idx]
            except IndexError:
                value = ""
            result = test(value)
            if self.any_match:
                if result:
                    return not self.inverse
            else:
                if not result:
                    return self.inverse
        if self.any_match:
            return self.inverse
        return not self.inverse


def standardize_patterns(column_names, patterns):
    """`csvkit.grep.standardize_patterns`."""
    from ..exceptions import ColumnIdentifierError

    try:
        patterns = {k: pattern_as_function(v) for k, v in patterns.items() if v}
        if not column_names:
            return patterns
        p2 = {}
        for k in patterns:
            if k in column_names:
                idx = column_names.index(k)
                if idx in patterns:
                    raise ColumnIdentifierError(
                        "Column %s has index %i which already has a pattern."
                        % (k, idx))
                p2[idx] = patterns[k]
            else:
                p2[k] = patterns[k]
        return p2
    except AttributeError:
        return {i: pattern_as_function(x) for i, x in enumerate(patterns)}


def pattern_as_function(obj):
    """`csvkit.grep.pattern_as_function`."""
    if callable(obj):
        return obj
    if hasattr(obj, "match"):
        return regex_callable(obj)
    return lambda x: obj in x


class regex_callable:
    def __init__(self, pattern):
        self.pattern = pattern

    def __call__(self, arg):
        return self.pattern.search(arg)


class CSVGrep(CSVKitUtility):
    description = 'Search CSV files. Like the Unix "grep" command, but for tabular data.'
    override_flags = ["L", "I"]

    def add_arguments(self):
        self.argparser.add_argument(
            "-n", "--names", dest="names_only", action="store_true",
            help="Display column names and indices from the input CSV and exit.")
        self.argparser.add_argument(
            "-c", "--columns", dest="columns",
            help="A comma-separated list of column indices, names or ranges to "
                 'be searched, e.g. "1,id,3-5".')
        self.argparser.add_argument(
            "-m", "--match", dest="pattern", action="store",
            help="A string to search for.")
        self.argparser.add_argument(
            "-r", "--regex", dest="regex", action="store",
            help="A regular expression to match.")
        self.argparser.add_argument(
            "-f", "--file", dest="matchfile", action="store",
            help="A path to a file. For each row, if any line in the file "
                 "(stripped of line separators) is an exact match of the cell "
                 "value, the row matches.")
        self.argparser.add_argument(
            "-i", "--invert-match", dest="inverse", action="store_true",
            help="Select non-matching rows, instead of matching rows.")
        self.argparser.add_argument(
            "-a", "--any-match", dest="any_match", action="store_true",
            help="Select rows in which any column matches, instead of all columns.")

    def main(self):
        if self.args.names_only:
            self.print_column_names()
            return

        if not self.args.columns:
            self.argparser.error("You must specify at least one column to search using the -c option.")
        if self.args.regex is None and self.args.pattern is None and self.args.matchfile is None:
            self.argparser.error("One of -r, -m or -f must be specified, unless using the -n option.")

        reader_kwargs = dict(self.reader_kwargs)
        if self.writer_kwargs.pop("line_numbers", False):
            reader_kwargs["line_numbers"] = True
        # Move the line_numbers option from the writer to the reader.
        self.reader_kwargs = reader_kwargs

        rows, column_names, column_ids = \
            self.get_rows_and_column_names_and_column_ids(**reader_kwargs)

        if self.args.regex:
            pattern = re.compile(self.args.regex)
        elif self.args.matchfile:
            with open(self.args.matchfile) as f:
                lines = {line.rstrip() for line in f}
            pattern = lambda x: x in lines  # noqa: E731
        else:
            pattern = self.args.pattern

        if callable(pattern) or hasattr(pattern, "search"):
            patterns = {column_id: pattern for column_id in column_ids}
            filter_reader = FilteringCSVReader(
                iter(rows.rows()), header=False, patterns=patterns,
                inverse=self.args.inverse, any_match=self.args.any_match)
            # `output.writerow(column_names)` runs before the filter, so the
            # header is emitted whatever the patterns matched.
            kept = [rows.row(0)] + list(filter_reader)
        else:
            kept_rows = _test_row(rows, column_ids, pattern, self.args.any_match,
                                  self.args.inverse)
            kept = [rows.row(r) for r in range(rows.nrows) if kept_rows[r]]

        from .. import csv as mcsv

        self.output_file.write(
            mcsv.write(from_rows(kept), lineterminator="\n").decode("utf-8", "replace"))


def _test_row(table: Table, column_ids, pattern, any_match, inverse) -> np.ndarray:
    """The `test_row` kernel, for plain `-m` patterns."""
    needles = [pattern.encode("utf-8")] * len(column_ids)
    buf = np.frombuffer(b"".join(needles), dtype=np.uint8) if needles \
        else np.zeros(1, dtype=np.uint8)
    off = np.zeros(len(needles), dtype=np.int64)
    ln = np.zeros(len(needles), dtype=np.int64)
    pos = 0
    for i, n in enumerate(needles):
        off[i] = pos
        ln[i] = len(n)
        pos += len(n)
    cols = np.array(column_ids, dtype=np.int64)
    keep = np.zeros(table.nrows, dtype=np.uint8)
    empty = np.zeros(1, dtype=np.uint8)
    lib().test_row(
        table.data.ctypes.data, table.cell_off.ctypes.data,
        table.cell_len.ctypes.data, table.row_cell.ctypes.data, table.nrows,
        cols.ctypes.data, len(cols), buf.ctypes.data, off.ctypes.data,
        ln.ctypes.data, 1 if any_match else 0, 1 if inverse else 0,
        empty.ctypes.data, keep.ctypes.data,
    )
    return keep


def launch_new_instance():
    utility = CSVGrep()
    utility.run()


if __name__ == "__main__":
    launch_new_instance()