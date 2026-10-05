#!/usr/bin/env python

"""csvstat: descriptive statistics for each column of a CSV file.

`OPERATIONS` and `print_one` / `print_stats` / `print_csv` / `print_json` are
upstream's; the arithmetic behind each operation runs in Mojo. `get_type`,
`get_unique` and `get_freq` are upstream's module-level helpers, transcribed.
"""

import json
import locale
import warnings
from collections import Counter, OrderedDict
from decimal import Decimal

import numpy as np

from .._lib import lib
from ..cli import CSVKitUtility, default_float_decimal, parse_column_identifiers
from ..csv import Table
from ..table import (T_BOOLEAN, T_NUMBER, T_TEXT, cast_column, infer_types)

locale.setlocale(locale.LC_ALL, "")

# csvstat's OPERATIONS, in upstream's order. `aggregation` names the agate
# aggregation the Mojo kernel implements; `None` means a `get_*` helper.
OPERATIONS = OrderedDict([
    ("type", {"aggregation": None, "label": "Type of data: "}),
    ("nulls", {"aggregation": "has_nulls", "label": "Contains null values: "}),
    ("nonnulls", {"aggregation": "count", "label": "Non-null values: "}),
    ("unique", {"aggregation": None, "label": "Unique values: "}),
    ("min", {"aggregation": "min", "label": "Smallest value: "}),
    ("max", {"aggregation": "max", "label": "Largest value: "}),
    ("sum", {"aggregation": "sum", "label": "Sum: "}),
    ("mean", {"aggregation": "mean", "label": "Mean: "}),
    ("median", {"aggregation": "median", "label": "Median: "}),
    ("stdev", {"aggregation": "stdev", "label": "StDev: "}),
    ("len", {"aggregation": "max_length", "label": "Longest value: "}),
    ("maxprecision", {"aggregation": "max_precision", "label": "Most decimal places: "}),
    ("freq", {"aggregation": None, "label": "Most common values: "}),
])

_TYPE_NAMES = {
    T_BOOLEAN: "Boolean",
    T_NUMBER: "Number",
    2: "TimeDelta",
    3: "Date",
    4: "DateTime",
    T_TEXT: "Text",
}


class CSVStat(CSVKitUtility):
    description = "Print descriptive statistics for each column in a CSV file."

    def add_arguments(self):
        self.argparser.add_argument(
            "--csv", dest="csv_output", action="store_true",
            help="Output results as a CSV table, rather than plain text.")
        self.argparser.add_argument(
            "--json", dest="json_output", action="store_true",
            help="Output results as JSON text, rather than plain text.")
        self.argparser.add_argument(
            "-i", "--indent", dest="indent", type=int,
            help="Indent the output JSON this many spaces. Disabled by default.")
        self.argparser.add_argument(
            "-n", "--names", dest="names_only", action="store_true",
            help="Display column names and indices from the input CSV and exit.")
        self.argparser.add_argument(
            "-c", "--columns", dest="columns",
            help="A comma-separated list of column indices, names or ranges to "
                 'be examined, e.g. "1,id,3-5". Defaults to all columns.')
        for flag, dest in (
            ("--type", "type_only"), ("--nulls", "nulls_only"),
            ("--non-nulls", "nonnulls_only"), ("--unique", "unique_only"),
            ("--min", "min_only"), ("--max", "max_only"), ("--sum", "sum_only"),
            ("--mean", "mean_only"), ("--median", "median_only"),
            ("--stdev", "stdev_only"), ("--len", "len_only"),
            ("--max-precision", "maxprecision_only"), ("--freq", "freq_only"),
        ):
            self.argparser.add_argument(
                flag, dest=dest, action="store_true",
                help="Only output %s." % dest[:-5].replace("_", "-"))
        self.argparser.add_argument(
            "--freq-count", dest="freq_count", type=int,
            help="The maximum number of frequent values to display.")
        self.argparser.add_argument(
            "--count", dest="count_only", action="store_true",
            help="Only output total row count.")
        self.argparser.add_argument(
            "--decimal-format", dest="decimal_format", type=str, default="%.3f",
            help='%%-format specification for printing decimal numbers. '
                 'Defaults to locale-specific formatting with "%%.3f".')
        self.argparser.add_argument(
            "-G", "--no-grouping-separator", dest="no_grouping_separator",
            action="store_true",
            help="Do not use grouping separators in decimal numbers.")
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

        operations = [op for op in OPERATIONS if getattr(self.args, op + "_only")]
        if len(operations) > 1:
            self.argparser.error(
                "Only one operation argument may be specified (--mean, --median, etc).")
        if operations and self.args.csv_output:
            self.argparser.error(
                "You may not specify --csv and an operation (--mean, --median, etc) "
                "at the same time.")
        if operations and self.args.json_output:
            self.argparser.error(
                "You may not specify --json and an operation (--mean, --median, etc) "
                "at the same time.")
        if operations and self.args.count_only:
            self.argparser.error(
                "You may not specify --count and an operation (--mean, --median, "
                "etc) at the same time.")

        table = self.read_table()
        order, vs = self.get_column_types()
        table._stats = _Stats(table, vs, order)

        if self.args.count_only:
            count = table.nrows
            if not self.args.no_header_row:
                count -= 1
            self.output_file.write("%i\n" % count)
            return

        column_ids = parse_column_identifiers(
            self.args.columns, table.header, self.get_column_offset())

        kwargs = {}
        if self.args.freq_count:
            kwargs["freq_count"] = self.args.freq_count

        if operations:
            if len(column_ids) == 1:
                self.print_one(table, column_ids[0], operations[0], label=False, **kwargs)
            else:
                for column_id in column_ids:
                    self.print_one(table, column_id, operations[0], **kwargs)
        else:
            stats = {cid: self.calculate_stats(table, cid, **kwargs) for cid in column_ids}
            if self.args.csv_output:
                self.print_csv(table, column_ids, stats)
            elif self.args.json_output:
                self.print_json(table, column_ids, stats)
            else:
                self.print_stats(table, column_ids, stats)

    def is_finite_decimal(self, value):
        return isinstance(value, Decimal) and value.is_finite()

    def _calculate_stat(self, table, column_id, op_name, op_data, **kwargs):
        getter = globals().get(f"get_{op_name}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                if getter:
                    return getter(table, column_id, **kwargs)
                op = op_data["aggregation"]
                v = _aggregate(table._stats, column_id, op)
                if self.is_finite_decimal(v) and not self.args.json_output:
                    return format_decimal(v, self.args.decimal_format,
                                         self.args.no_grouping_separator)
                return v
            except Exception:
                return None

    def print_one(self, table, column_id, op_name, label=True, **kwargs):
        column_name = table.header[column_id]
        stat = self._calculate_stat(table, column_id, op_name, OPERATIONS[op_name], **kwargs)
        if op_name == "freq":
            stat = ", ".join([f"\"{str(row['value'])}\": {row['count']}" for row in stat])
            stat = "{ %s }" % stat
        if label:
            self.output_file.write("%3i. %s: %s\n" % (column_id + 1, column_name, stat))
        else:
            self.output_file.write(f"{stat}\n")

    def calculate_stats(self, table, column_id, **kwargs):
        return {
            op_name: self._calculate_stat(table, column_id, op_name, op_data, **kwargs)
            for op_name, op_data in OPERATIONS.items()
        }

    def print_stats(self, table, column_ids, stats):
        label_column_width = max(len(op_data["label"]) for op_data in OPERATIONS.values())
        for column_id in column_ids:
            column_name = table.header[column_id]
            column_stats = stats[column_id]
            self.output_file.write('%3i. "%s"\n\n' % (column_id + 1, column_name))
            for op_name, op_data in OPERATIONS.items():
                if column_stats[op_name] is None:
                    continue
                label = op_data["label"].ljust(label_column_width)
                if op_name == "freq":
                    for i, row in enumerate(column_stats["freq"]):
                        self.output_file.write(
                            ("\t%s " % label) if i == 0 else
                            ("\t%s " % "".ljust(label_column_width)))
                        v = row["value"]
                        if _is_number(v) and self.is_finite_decimal(v):
                            v = format_decimal(v, self.args.decimal_format,
                                                self.args.no_grouping_separator)
                        self.output_file.write(f"{v} ({row['count']}x)\n")
                    continue
                v = column_stats[op_name]
                if op_name == "nulls" and v:
                    v = f"{v} (excluded from calculations)"
                elif op_name == "len":
                    v = f"{v} characters"
                self.output_file.write(f"\t{label} {v}\n")
            self.output_file.write("\n")
        self.output_file.write(f"Row count: {table.nrows - 1}\n")

    def print_csv(self, table, column_ids, stats):
        from .. import csv as mcsv
        from ..csv import from_rows

        header = ["column_id", "column_name"] + list(OPERATIONS)
        rows = [header]
        for row in self._rows(table, column_ids, stats):
            if "freq" in row:
                row["freq"] = ", ".join([str(r["value"]) for r in row["freq"]])
            rows.append([row.get(k, "") for k in header])
        self.output_file.write(
            mcsv.write(from_rows(rows), lineterminator="\n").decode("utf-8", "replace"))

    def print_json(self, table, column_ids, stats):
        data = list(self._rows(table, column_ids, stats))
        json.dump(data, self.output_file, default=default_float_decimal,
                  ensure_ascii=False, indent=self.args.indent)

    def _rows(self, table, column_ids, stats):
        for column_id in column_ids:
            column_stats = stats[column_id]
            output_row = {"column_id": column_id + 1, "column_name": table.header[column_id]}
            for op_name in OPERATIONS:
                if column_stats[op_name] is not None:
                    output_row[op_name] = column_stats[op_name]
            yield output_row


def _is_number(v):
    return isinstance(v, Decimal)


def format_decimal(d, f="%.3f", no_grouping_separator=False):
    """`csvkit.utilities.csvstat.format_decimal`."""
    return locale.format_string(f, d, grouping=not no_grouping_separator) \
        .rstrip("0").rstrip(".")


class _DataColumn:
    """One typed column, viewed from the first data row."""

    __slots__ = ("vals", "isnull", "ndig", "expo", "special", "nrows")

    def __init__(self, vals, isnull, ndig, expo, special, nrows):
        self.vals = vals
        self.isnull = isnull
        self.ndig = ndig
        self.expo = expo
        self.special = special
        self.nrows = nrows


def _dec(v):
    """A float64 result back on agate's Decimal footing."""
    return None if v is None else Decimal(repr(float(v)))


def _base_of(table, cell, column_id):
    """The row a cell index belongs to, so its value can be read back."""
    for r in range(table.nrows - 1, 0, -1):
        if int(table.row_cell[r]) <= cell < int(table.row_cell[r + 1]):
            return r
    return 1


def _number_value(text):
    """`Number.cast`, applied on the host to keep the source's scale.

    The Mojo cast produces the value and the normalized digit count; this is
    the same transform over the raw text, and it is what a grouped or
    currency-prefixed cell needs when the value has to be printed the way agate
    would print it.
    """
    from ..table import DEFAULT_CURRENCY_SYMBOLS

    if text is None:
        return None
    d = text.strip().strip("%")
    sign = ""
    if d.startswith("-"):
        sign = "-"
        d = d[1:]
    for symbol in DEFAULT_CURRENCY_SYMBOLS:
        d = d.strip(symbol)
    d = d.replace(",", "")
    try:
        return Decimal(d) * (Decimal(-1) if sign else Decimal(1))
    except Exception:
        return text


def _value_at(stats, cell, column_id):
    """The aggregate's winning value as agate would have printed it.

    The source text is used when it reads as a Decimal, which keeps agate's
    exact formatting ("1" not "1.0"); a grouped or currency-prefixed cell falls
    back to the parsed value, which is what agate's cast would have held.
    """
    if cell < 0:
        return None
    text = stats.table.cell(cell)
    try:
        return Decimal(text)
    except Exception:
        # A grouped or currency-prefixed cell: `Number.cast` strips the symbols
        # and keeps the scale, which the parsed float would lose.
        return _number_value(text)




def _aggregate(stats, column_id, op):
    """Run one agate aggregation over a column, as `table.aggregate` does."""
    table = stats.table
    col = stats.column(column_id)
    col_type = stats.types[column_id]
    n = stats.nrows
    L = lib()

    if op == "has_nulls":
        return bool(L.agg_has_nulls(col.isnull.ctypes.data, n))
    if op == "count":
        return L.agg_count(col.isnull.ctypes.data, n)
    if op in ("sum", "mean", "stdev", "variance"):
        if col_type != T_NUMBER:
            return None
        fn = {"sum": L.agg_sum, "mean": L.agg_mean,
              "stdev": L.agg_stdev, "variance": L.agg_variance}[op]
        return _dec(fn(col.vals.ctypes.data, col.isnull.ctypes.data, n))
    if op in ("min", "max"):
        if col_type != T_NUMBER:
            return None
        # The value arrays are the data-row view, so the count is n, not the
        # table's row count; `row_start` maps back into `row_cell`.
        cell = (L.agg_min if op == "min" else L.agg_max)(
            table.cell_off.ctypes.data, table.cell_len.ctypes.data,
            table.row_cell.ctypes.data, n, 1, column_id,
            col.vals.ctypes.data, col.isnull.ctypes.data)
        return _value_at(stats, cell, column_id)
    if op == "max_length":
        if col_type != T_TEXT:
            return None
        # `MaxLength.run` returns a Decimal, so the JSON output is a float.
        return _dec(Decimal(L.agg_max_length(
            table.data.ctypes.data, table.cell_off.ctypes.data,
            table.cell_len.ctypes.data, table.row_cell.ctypes.data,
            table.nrows, 1, column_id)))
    if op == "max_precision":
        if col_type != T_NUMBER:
            return None
        return L.agg_max_precision(
            col.ndig.ctypes.data, col.expo.ctypes.data, col.isnull.ctypes.data,
            col.special.ctypes.data, n)
    if op == "median":
        if col_type != T_NUMBER:
            return None
        keys = np.zeros(max(n, 1), dtype=np.float64)
        rows = np.zeros(max(n, 1), dtype=np.int64)
        out = np.zeros(4, dtype=np.int64)
        L.agg_median(col.vals.ctypes.data, col.isnull.ctypes.data, n,
                     table.row_cell.ctypes.data, column_id,
                     keys.ctypes.data, rows.ctypes.data, 50, 1, out.ctypes.data)
        if int(out[0]) != 0:
            return None
        lo = _value_at(stats, int(out[2]), column_id)
        hi = _value_at(stats, int(out[3]), column_id)
        if lo is None:
            return None
        return lo if lo == hi else (lo + hi) / 2
    return None


# ------------------------------------------------------------ the aggregations
class _Stats:
    """The per-run state `csvstat` builds: the type tester, the cast columns and
    the distinct-value tables, each computed once and reused across the
    operations the way upstream reuses `table.aggregate`."""

    def __init__(self, table: Table, vs, order):
        self.table = table
        self.vs = vs
        self.order = order
        self.types, _keep = infer_types(table, vs, order)
        self._cols: dict[int, object] = {}
        self._distinct: dict[int, tuple] = {}

    @property
    def nrows(self) -> int:
        """Data rows: the header is not a column value."""
        return self.table.nrows - 1

    def column(self, column_id: int):
        """The typed column, indexed from the first data row.

        The header is not a column value, so the cast arrays are viewed from
        row 1; every aggregation below then walks exactly the data rows.
        """
        if column_id not in self._cols:
            col = cast_column(self.table, column_id, self.vs)
            # A Boolean column's distinct set is over the cast booleans.
            keys = col.vals[1:].copy()
            if self.types[column_id] == T_BOOLEAN:
                keys = _boolean_keys(self.table, column_id, self.vs, self.nrows)
            self._cols[column_id] = _DataColumn(
                keys, col.isnull[1:], col.ndig[1:], col.expo[1:],
                col.special[1:], self.nrows)
        return self._cols[column_id]

    def distinct(self, column_id: int):
        """`values_distinct` and the frequency table behind `get_freq`.

        The values are the column's *cast* ones, so the keys are built from the
        typed column: a Number column's "1.0" and "1" are one value, and a
        Boolean column's five spellings of True are one.
        """
        if column_id in self._distinct:
            return self._distinct[column_id]
        rendered = _rendered_values(self, column_id)
        ranked, counts = _counter(rendered)
        first = {}
        seen = set()
        for r in range(self.nrows):
            v = rendered[r]
            if v not in seen:
                seen.add(v)
                cell = int(self.table.row_cell[r + 1]) + column_id
                first[v] = cell if cell < self.table.ncells else -1
        entries = [[first[v], counts[v]] for v in ranked]
        result = (len(ranked), entries, [e[1] for e in entries], None)
        self._distinct[column_id] = result
        return result


def _rendered_values(stats, column_id):
    """A column's cast values, in the form the counter and the set see them."""
    col = stats.column(column_id)
    col_type = stats.types[column_id]
    out = []
    for r in range(stats.nrows):
        if col_type == T_NUMBER:
            cell = int(stats.table.row_cell[r + 1]) + column_id
            out.append(_value_at(stats, cell if cell < stats.table.ncells else -1,
                                 column_id))
        elif col_type == T_BOOLEAN:
            key = float(col.vals[r])
            out.append(None if key < 0 else (key > 0))
        else:
            cell = int(stats.table.row_cell[r + 1]) + column_id
            out.append(stats.table.cell(cell) if cell < stats.table.ncells else "")
    return out



def _counter(values):
    """`Counter(values).most_common()`, ties broken by first appearance.

    `heapq.nlargest` over an insertion-ordered Counter resolves equal counts in
    favour of the one that appeared first, which is what this reproduces.
    """
    counts = {}
    order = []
    for v in values:
        if v not in counts:
            counts[v] = 0
            order.append(v)
        counts[v] += 1
    return sorted(order, key=lambda v: -counts[v]), counts


def _boolean_keys(table, column_id, vs, nrows):
    """A Boolean column's values as 1 / 0 / -1 for True / False / None.

    `Column.values()` yields the cast values, so a set over them collapses every
    spelling of True into one entry. `Number.cast` does not produce booleans,
    so this applies `Boolean.cast` -- the same rules, on the host.
    """
    from ..table import DEFAULT_FALSE_VALUES, DEFAULT_TRUE_VALUES

    nulls = {v.lower() for v in vs.nulls_value_list()}
    trues = set(DEFAULT_TRUE_VALUES)
    falses = set(DEFAULT_FALSE_VALUES)
    keys = np.zeros(max(nrows, 1), dtype=np.float64)
    for r in range(1, table.nrows):
        i = int(table.row_cell[r]) + column_id
        text = table.cell(i) if i < table.ncells else None
        d = (text or "").replace(",", "").strip().lower()
        keys[r - 1] = -1.0 if d in nulls else (1.0 if d in trues else 0.0)
    return keys


def get_type(table, column_id, **kwargs):
    """`csvkit.utilities.csvstat.get_type`."""
    return _TYPE_NAMES[table._stats.types[column_id]]


def get_unique(table, column_id, **kwargs):
    """`csvkit.utilities.csvstat.get_unique`: `len(values_distinct())`."""
    return table._stats.distinct(column_id)[0]


def get_freq(table, column_id, freq_count=5, **kwargs):
    """`csvkit.utilities.csvstat.get_freq`: `Counter(values).most_common(n)`.

    The counter runs over the *cast* values, so the reported value is the one
    the column's type holds, not the text it was spelled with. A Text value is
    reported from the cell it came from so its exact spelling survives; a
    Number or Boolean value is reported as the cast object, which is what
    `format_decimal` and `jsonify` then take.
    """
    stats = table._stats
    col_type = stats.types[column_id]
    _ndistinct, entries, _counts, _keep = stats.distinct(column_id)
    first = [e[0] for e in entries]
    ranked, counts = _counter(_rendered_values(stats, column_id))
    out = []
    for v in ranked[:freq_count]:
        if col_type == T_TEXT:
            i = ranked.index(v)
            cell = first[i]
            value = table.cell(cell) if cell >= 0 else None
        else:
            value = v
        out.append({"value": value, "count": counts[v]})
    return out


def launch_new_instance():
    utility = CSVStat()
    utility.run()


if __name__ == "__main__":
    launch_new_instance()
