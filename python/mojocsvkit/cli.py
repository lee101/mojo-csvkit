"""`csvkit.cli`: the shared argument handling, and the column-identifier parser.

Upstream `csvkit.cli` is an `argparse` class, `CSVKitUtility`, that every
utility inherits. This module keeps that shape -- same flags, same defaults,
same `ColumnIdentifierError` messages -- and swaps the underlying CSV reader for
the Mojo one.
"""

from __future__ import annotations

import csv as _csv
import os
import sys
from argparse import ArgumentParser, FileType

import numpy as np

from . import csv as mcsv
from ._lib import lib
from .csv import Table
from .exceptions import ColumnIdentifierError, RequiredHeaderError
from .table import get_column_types, infer_types

QUOTING_CHOICES = sorted(
    getattr(_csv, name) for name in dir(_csv) if name.startswith("QUOTE_")
)


def isatty(f) -> bool:
    try:
        return f.isatty()
    except ValueError:  # I/O operation on closed file
        return False


def default_str_decimal(obj):
    import datetime
    import decimal

    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    if isinstance(obj, decimal.Decimal):
        return str(obj)
    raise TypeError(f"{repr(obj)} is not JSON serializable")


def default_float_decimal(obj):
    import datetime
    import decimal

    if isinstance(obj, datetime.timedelta):
        return obj.total_seconds()
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    return default_str_decimal(obj)


def make_default_headers(n: int) -> list[str]:
    """`csvkit.cli.make_default_headers`: the `letter_name` run for 0..n-1."""
    buf = np.zeros(64 * max(n, 1) + 8, dtype=np.uint8)
    total = lib().make_default_headers(n, buf.ctypes.data)
    return _split_letters(buf[:total].tobytes())


def _split_letters(raw: bytes) -> list[str]:
    out = []
    i = 0
    while i < len(raw):
        j = i
        while j < len(raw) and raw[j] == raw[i]:
            j += 1
        out.append(raw[i:j].decode())
        i = j
    return out


def match_column_identifier(
    column_names: list[str], c, column_offset: int = 1
) -> int:
    """`csvkit.cli.match_column_identifier(column_names, c, column_offset)`."""
    if isinstance(c, str) and not c.isdigit() and c in column_names:
        return column_names.index(c)

    try:
        c = int(c) - column_offset
    except ValueError:
        raise ColumnIdentifierError(
            "Column '%s' is invalid. It is neither an integer nor a column "
            "name. Column names are: %s" % (c, repr(column_names)[1:-1])
        )

    if c < 0:
        raise ColumnIdentifierError(
            "Column %i is invalid. Columns are 1-based." % (c + column_offset)
        )

    if c >= len(column_names):
        raise ColumnIdentifierError(
            "Column %i is invalid. The last column is '%s' at index %i."
            % (c + column_offset, column_names[-1], len(column_names) - 1 + column_offset)
        )

    return c


def _names_arena(column_names: list[str]) -> Table:
    from .csv import from_rows

    return from_rows([column_names])


def parse_column_identifiers(
    ids: str | None,
    column_names: list[str],
    column_offset: int = 1,
    excluded_columns: str | None = None,
    ignore_unknown_columns: bool = False,
) -> list[int]:
    """`csvkit.cli.parse_column_identifiers(...)`.

    The split on ',' and the range handling run in the Mojo kernel; this wrapper
    only translates the status codes into upstream's exceptions.
    """
    if not column_names:
        return []

    arena = _names_arena(column_names)
    ids_b = (ids or "").encode("utf-8")
    exc_b = (excluded_columns or "").encode("utf-8")
    cap = max(len(column_names), 1) * 4 + 8
    out = np.zeros(cap, dtype=np.int64)
    scratch = np.zeros(cap, dtype=np.int64)
    status = np.zeros(2, dtype=np.int64)

    count = lib().parse_column_identifiers(
        arena.data.ctypes.data, arena.cell_off.ctypes.data,
        arena.cell_len.ctypes.data,
        int(arena.row_cell[0]), len(column_names),
        _addr_of(ids_b), len(ids_b),
        _addr_of(exc_b), len(exc_b),
        column_offset, 1 if ignore_unknown_columns else 0,
        out.ctypes.data, scratch.ctypes.data, status.ctypes.data,
    )
    _raise_for_status(int(status[0]), ids or "", column_names)
    return [int(v) for v in out[:count]]


def _addr_of(blob: bytes) -> int:
    if not blob:
        # A zero-length ids/excluded string still needs a valid address.
        return _EMPTY.ctypes.data
    return _np_bytes(blob).ctypes.data


_EMPTY = np.zeros(1, dtype=np.uint8)


def _np_bytes(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.uint8)


def _raise_for_status(status: int, identifier: str, column_names: list[str]):
    if status == 0:
        return
    if status == 2:
        raise ColumnIdentifierError(
            "Column '%s' is invalid. It is neither an integer nor a column "
            "name. Column names are: %s" % (identifier, repr(column_names)[1:-1])
        )
    if status == 3:
        raise ColumnIdentifierError(
            "Invalid range %s. Ranges must be two integers separated by a - "
            "or : character." % identifier
        )
    if status == 4:
        raise ColumnIdentifierError(
            "Column %s is invalid. Columns are 1-based." % identifier
        )
    raise ColumnIdentifierError("Column '%s' is invalid." % identifier)


def parse_list(pairs):
    import ast

    options = {}
    for key, value in pairs:
        try:
            value = ast.literal_eval(value)
        except ValueError:
            pass
        options[key] = value
    return options


class CSVKitUtility:
    """`csvkit.cli.CSVKitUtility`, with the Mojo CSV reader underneath."""

    description = ""
    epilog = ""
    override_flags = ""

    def __init__(self, args=None, output_file=None, error_file=None):
        if args is None:
            args = sys.argv[1:]
        self._init_common_parser()
        self.add_arguments()
        self.args = self.argparser.parse_args(args)

        self.output_file = sys.stdout if output_file is None else output_file
        self.error_file = sys.stderr if error_file is None else error_file

        self.reader_kwargs = self._extract_csv_reader_kwargs()
        self.writer_kwargs = self._extract_csv_writer_kwargs()

    def add_arguments(self):
        raise NotImplementedError(
            "add_arguments must be provided by each subclass of CSVKitUtility."
        )

    def run(self):
        if "f" not in self.override_flags:
            self.input_file = self._open_input_file(self.args.input_path)
        try:
            self.main()
        finally:
            if "f" not in self.override_flags:
                self.input_file.close()

    def main(self):
        raise NotImplementedError(
            "main must be provided by each subclass of CSVKitUtility."
        )

    def _init_common_parser(self):
        self.argparser = ArgumentParser(
            prog=type(self).__name__.lower(),
            description=self.description,
            epilog=self.epilog,
        )
        omit = self.override_flags

        if "f" not in omit:
            self.argparser.add_argument(
                metavar="FILE", nargs="?", dest="input_path",
                help="The CSV file to operate on. If omitted, will accept input "
                     "as piped data via STDIN.")
        if "d" not in omit:
            self.argparser.add_argument(
                "-d", "--delimiter", dest="delimiter",
                help="Delimiting character of the input CSV file.")
        if "t" not in omit:
            self.argparser.add_argument(
                "-t", "--tabs", dest="tabs", action="store_true",
                help="Specify that the input CSV file is delimited with tabs. "
                     'Overrides "-d".')
        if "q" not in omit:
            self.argparser.add_argument(
                "-q", "--quotechar", dest="quotechar",
                help="Character used to quote strings in the input CSV file.")
        if "u" not in omit:
            self.argparser.add_argument(
                "-u", "--quoting", dest="quoting", type=int,
                choices=QUOTING_CHOICES,
                help="Quoting style used in the input CSV file: 0 quote minimal, "
                     "1 quote all, 2 quote non-numeric, 3 quote none.")
        if "b" not in omit:
            self.argparser.add_argument(
                "-b", "--no-doublequote", dest="doublequote",
                action="store_false",
                help="Whether or not double quotes are doubled in the input CSV file.")
        if "p" not in omit:
            self.argparser.add_argument(
                "-p", "--escapechar", dest="escapechar",
                help="Character used to escape the delimiter if --quoting 3 "
                     '("quote none") is specified and to escape the QUOTECHAR if '
                     "--no-doublequote is specified.")
        if "z" not in omit:
            self.argparser.add_argument(
                "-z", "--maxfieldsize", dest="field_size_limit", type=int,
                help="Maximum length of a single field in the input CSV file.")
        if "e" not in omit:
            self.argparser.add_argument(
                "-e", "--encoding", dest="encoding",
                default=os.getenv("PYTHONIOENCODING", "utf-8-sig"),
                help="Specify the encoding of the input CSV file.")
        if "L" not in omit:
            self.argparser.add_argument(
                "-L", "--locale", dest="locale", default="en_US",
                help="Specify the locale (en_US) of any formatted numbers.")
        if "S" not in omit:
            self.argparser.add_argument(
                "-S", "--skipinitialspace", dest="skipinitialspace",
                action="store_true",
                help="Ignore whitespace immediately following the delimiter.")
        if "I" not in omit:
            self.argparser.add_argument(
                "--blanks", dest="blanks", action="store_true",
                help='Do not convert "", "na", "n/a", "none", "null", "." to NULL.')
            self.argparser.add_argument(
                "--null-value", dest="null_values", nargs="+", default=[],
                help="Convert this value to NULL. --null-value can be specified "
                     "multiple times.")
            self.argparser.add_argument(
                "--date-format", dest="date_format",
                help='Specify a strptime date format string like "%%m/%%d/%%Y".')
            self.argparser.add_argument(
                "--datetime-format", dest="datetime_format",
                help="Specify a strptime datetime format string like "
                     '"%%m/%%d/%%Y %%I:%%M %%p".')
            self.argparser.add_argument(
                "--no-leading-zeroes", dest="no_leading_zeroes",
                action="store_true",
                help="Do not convert a numeric value with leading zeroes to a number.")
        if "H" not in omit:
            self.argparser.add_argument(
                "-H", "--no-header-row", dest="no_header_row",
                action="store_true",
                help="Specify that the input CSV file has no header row. Will "
                     "create default headers (a,b,c,...).")
        if "K" not in omit:
            self.argparser.add_argument(
                "-K", "--skip-lines", dest="skip_lines", type=int, default=0,
                help="Specify the number of initial lines to skip before the "
                     "header row (e.g. comments, copyright notices, empty rows).")
        if "v" not in omit:
            self.argparser.add_argument(
                "-v", "--verbose", dest="verbose", action="store_true",
                help="Print detailed tracebacks when errors occur.")
        if "l" not in omit:
            self.argparser.add_argument(
                "-l", "--linenumbers", dest="line_numbers", action="store_true",
                help="Insert a column of line numbers at the front of the output.")
        if "add-bom" not in omit:
            self.argparser.add_argument(
                "--add-bom", dest="add_bom", action="store_true",
                help="Add the UTF-8 byte-order mark (BOM) to the output, for "
                     "Excel compatibility")
        if "zero" not in omit:
            self.argparser.add_argument(
                "--zero", dest="zero_based", action="store_true",
                help="When interpreting or displaying column numbers, use "
                     "zero-based numbering instead of the default 1-based numbering.")
        self.argparser.add_argument(
            "-V", "--version", action="version", version="%(prog)s 2.2.0",
            help="Display version information and exit.")

    def _open_input_file(self, path, opened=False):
        if not path or path == "-":
            return sys.stdin
        return open(path, "rt", encoding=self.args.encoding, newline=None)

    def _extract_csv_reader_kwargs(self):
        kwargs = {}
        if self.args.tabs:
            kwargs["delimiter"] = "\t"
        elif self.args.delimiter:
            kwargs["delimiter"] = self.args.delimiter
        for arg in ("quotechar", "quoting", "doublequote", "escapechar",
                    "skipinitialspace"):
            value = getattr(self.args, arg)
            if value is not None:
                kwargs[arg] = value
        if getattr(self.args, "no_header_row", None):
            kwargs["header"] = not self.args.no_header_row
        return kwargs

    def _extract_csv_writer_kwargs(self):
        kwargs = {}
        if getattr(self.args, "line_numbers", None):
            kwargs["line_numbers"] = True
        return kwargs

    def get_column_types(self):
        """`CSVKitUtility.get_column_types`."""
        order, vs = get_column_types(
            blanks=getattr(self.args, "blanks", None),
            null_values=getattr(self.args, "null_values", []),
            date_format=self.args.date_format,
            datetime_format=self.args.datetime_format,
            no_inference=getattr(self.args, "no_inference", None),
            locale=self.args.locale,
            out_quoting=getattr(self.args, "out_quoting", None),
            no_leading_zeroes=getattr(self.args, "no_leading_zeroes", None),
        )
        return order, vs

    def get_column_offset(self):
        return 0 if self.args.zero_based else 1

    def skip_lines(self):
        """`CSVKitUtility.skip_lines`: drop the first `-K` lines of the input."""
        if isinstance(self.args.skip_lines, int):
            out = []
            n = self.args.skip_lines
            while n > 0:
                line = self.input_file.readline()
                if not line:
                    break
                n -= 1
        else:
            raise ValueError("skip_lines argument must be an int")
        return self.input_file

    def read_table(self) -> Table:
        """The whole input, parsed by the Mojo reader with this utility's
        dialect."""
        text = self.input_file.read()
        return mcsv.reader(text.encode(self.args.encoding.replace("utf-8-sig", "utf-8")
                                       if self.args.encoding.startswith("utf-8")
                                       else self.args.encoding), **self.reader_kwargs)

    def write_table(self, table: Table) -> None:
        """`agate.csv.writer`'s counterpart for a whole table."""
        kw = dict(self.writer_kwargs)
        line_numbers = kw.pop("line_numbers", False)
        if line_numbers:
            self.output_file.write(
                mcsv.write(table, **kw).decode(self.args.encoding, "replace")
            )
            return
        self.output_file.write(
            mcsv.write(table, **kw).decode(self.args.encoding, "replace")
        )

    def get_rows_and_column_names_and_column_ids(self, **kwargs):
        rows = self.read_table()
        if rows.nrows == 0:
            return rows, [], []
        if self.args.no_header_row:
            column_names = make_default_headers(len(rows.row(0)))
            rows = _with_default_header(rows, column_names)
        else:
            column_names = rows.header
        column_offset = self.get_column_offset()
        if kwargs.get("line_numbers"):
            column_offset -= 1
        column_ids = parse_column_identifiers(
            self.args.columns, column_names, column_offset,
            getattr(self.args, "not_columns", None),
            ignore_unknown_columns=getattr(self.args, "ignore_unknown_columns", False),
        )
        return rows, column_names, column_ids

    def print_column_names(self):
        if getattr(self.args, "no_header_row", None):
            raise RequiredHeaderError(
                "You cannot use --no-header-row with the -n or --names options."
            )
        start = 0 if getattr(self.args, "zero_based", None) else 1
        column_names = self.read_table().header
        for i, c in enumerate(column_names, start):
            self.output_file.write("%3i: %s\n" % (i, c))

    def additional_input_expected(self):
        return isatty(sys.stdin) and not self.args.input_path


def _with_default_header(table: Table, names: list[str]) -> Table:
    """`make_default_headers` prepends a synthetic header row."""
    from .csv import from_rows

    rows = [names] + [table.row(r) for r in range(table.nrows)]
    return from_rows(rows)