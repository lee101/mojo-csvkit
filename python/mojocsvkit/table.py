"""The arena layout, and the agate type inference and aggregations over it.

`Table` is the parsed CSV: one `UInt8` arena holding every cell's decoded bytes,
an offset and a length per cell, and a per-row cell index. Row r, column c is
cell `row_cell[r] + c`, which is what lets the Mojo kernels address a column
without building an array of Python strings.
"""

from __future__ import annotations

import numpy as np

from ._lib import lib

# agate.data_types.base.DEFAULT_NULL_VALUES, lower-cased the way DataType's
# constructor lower-cases them.
DEFAULT_NULL_VALUES = ("", "na", "n/a", "none", "null", ".")

# agate.data_types.boolean.DEFAULT_TRUE_VALUES / DEFAULT_FALSE_VALUES
DEFAULT_TRUE_VALUES = ("yes", "y", "true", "t", "1")
DEFAULT_FALSE_VALUES = ("no", "n", "false", "f", "0")

# agate.data_types.number.DEFAULT_CURRENCY_SYMBOLS
DEFAULT_CURRENCY_SYMBOLS = (
    "؋", "$", "ƒ", "៛", "¥", "₡", "₱", "£", "€", "¢", "﷼", "₪", "₩",
    "₭", "₮", "₦", "฿", "₤", "₫",
)

# agate.type_tester's preference order. csvkit's get_column_types inserts Number
# into this list; see `get_column_types`.
T_BOOLEAN, T_NUMBER, T_TIMEDELTA, T_DATE, T_DATETIME, T_TEXT = range(6)


class StringList:
    """A list of byte strings, laid out for `list_contains`.

    `list_contains` reads an offset table and a length table beside one buffer,
    so a list of Python strings crosses the ABI as three pointers plus a count.
    """

    __slots__ = ("buf", "off", "ln")

    def __init__(self, values):
        blobs = [v.encode("utf-8") if isinstance(v, str) else bytes(v) for v in values]
        self.buf = np.frombuffer(b"".join(blobs) if blobs else b"", dtype=np.uint8)
        if not self.buf.size:
            self.buf = np.zeros(1, dtype=np.uint8)
        self.off = np.zeros(len(blobs), dtype=np.int64)
        self.ln = np.zeros(len(blobs), dtype=np.int64)
        pos = 0
        for i, b in enumerate(blobs):
            self.off[i] = pos
            self.ln[i] = len(b)
            pos += len(b)

    def __len__(self) -> int:
        return int(self.off.shape[0])


class ValueSets:
    """`ValueSets` in Mojo: the null/true/false/currency lists and the locale's
    number symbols, as one Int64 block."""

    __slots__ = ("nulls", "true_vals", "false_vals", "currency", "group_symbol",
                 "decimal_symbol", "no_leading_zeroes", "_keep", "_block",
                 "_null_values")

    def __init__(self, null_values=DEFAULT_NULL_VALUES, locale="en_US",
                 no_leading_zeroes=None, true_values=DEFAULT_TRUE_VALUES,
                 false_values=DEFAULT_FALSE_VALUES):
        self._null_values = list(null_values)
        self.nulls = StringList([v.lower() for v in null_values])
        self.true_vals = StringList([v.lower() for v in true_values])
        self.false_vals = StringList([v.lower() for v in false_values])
        self.currency = StringList(DEFAULT_CURRENCY_SYMBOLS)
        # babel's number_symbols for the requested locale; only the two csvkit
        # passes through are needed here.
        group, decimal = _number_symbols(locale)
        self.group_symbol = ord(group)
        self.decimal_symbol = ord(decimal)
        self.no_leading_zeroes = bool(no_leading_zeroes)
        self._keep = (self.nulls, self.true_vals, self.false_vals, self.currency)
        self._block = np.zeros(19, dtype=np.int64)
        self._block[0] = self.nulls.buf.ctypes.data
        self._block[1] = self.nulls.off.ctypes.data
        self._block[2] = self.nulls.ln.ctypes.data
        self._block[3] = len(self.nulls)
        self._block[4] = self.true_vals.buf.ctypes.data
        self._block[5] = self.true_vals.off.ctypes.data
        self._block[6] = self.true_vals.ln.ctypes.data
        self._block[7] = len(self.true_vals)
        self._block[8] = self.false_vals.buf.ctypes.data
        self._block[9] = self.false_vals.off.ctypes.data
        self._block[10] = self.false_vals.ln.ctypes.data
        self._block[11] = len(self.false_vals)
        self._block[12] = self.currency.buf.ctypes.data
        self._block[13] = self.currency.off.ctypes.data
        self._block[14] = self.currency.ln.ctypes.data
        self._block[15] = len(self.currency)
        self._block[16] = self.group_symbol
        self._block[17] = self.decimal_symbol
        self._block[18] = 1 if self.no_leading_zeroes else 0

    @property
    def addr(self) -> int:
        return self._block.ctypes.data

    def nulls_value_list(self):
        """The `null_values` this set was built with, lower-cased."""
        return self._null_values


def _number_symbols(locale: str) -> tuple[str, str]:
    """`Number`'s `group_symbol` and `decimal_symbol` for a locale."""
    if locale in ("en_US", "en_GB", "en_CA", "en_AU", "en_NZ", "en_IE", ""):
        return ",", "."
    if locale.startswith("de") or locale.startswith("fr") or locale.startswith("es"):
        return ".", ","
    if locale.startswith("pt") or locale.startswith("it"):
        return ".", ","
    return ",", "."


def get_column_types(
    blanks: bool = False,
    null_values=(),
    date_format: str | None = None,
    datetime_format: str | None = None,
    no_inference: bool = False,
    locale: str = "en_US",
    out_quoting: int | None = None,
    no_leading_zeroes: bool | None = None,
):
    """`CSVKitUtility.get_column_types`.

    Returns `(order, valuesets)`, where `order` is the TypeTester preference
    list csvkit builds. Date, DateTime and TimeDelta are dropped from the Mojo
    candidate set -- see the README's coverage section -- so the list below is
    what this port tests, in upstream's order otherwise.
    """
    if blanks:
        nv = []
    else:
        nv = list(DEFAULT_NULL_VALUES)
    nv.extend(null_values)
    vs = ValueSets(null_values=nv, locale=locale, no_leading_zeroes=no_leading_zeroes)

    if no_inference:
        return [T_TEXT], vs

    if out_quoting == 2:  # QUOTE_NONNUMERIC
        return [T_NUMBER, T_TEXT], vs

    # "See the order in the `agate.TypeTester` class."
    types = [T_BOOLEAN, T_NUMBER, T_TIMEDELTA, T_DATE, T_DATETIME, T_TEXT]
    # "In order to parse dates like "20010101"."
    if datetime_format:
        types.insert(-1, T_NUMBER)
    elif date_format:
        types.insert(-2, T_NUMBER)
    else:
        types.insert(1, T_NUMBER)

    # The three unported types always fail their test, so testing them is a
    # no-op; drop them from the candidate set while keeping the order above.
    order = [t for t in types if t not in (T_TIMEDELTA, T_DATE, T_DATETIME)]
    return order, vs


class Column:
    """One column's typed values, as `column_cast_number` produced them."""

    __slots__ = ("vals", "isnull", "ndig", "expo", "special", "nrows")

    def __init__(self, nrows: int):
        self.nrows = nrows
        self.vals = np.zeros(nrows, dtype=np.float64)
        self.isnull = np.zeros(nrows, dtype=np.uint8)
        self.ndig = np.zeros(nrows, dtype=np.int64)
        self.expo = np.zeros(nrows, dtype=np.int64)
        self.special = np.zeros(nrows, dtype=np.uint8)

    def __len__(self) -> int:
        return self.nrows


def cast_column(table, col: int, vs: ValueSets, row_start: int = 1) -> Column:
    """`column_cast_number`: one column through `Number.cast`."""
    out = Column(table.nrows)
    buf = np.zeros(max(int(table.data.shape[0]), 1) + 8, dtype=np.uint8)
    # number_cast reads its Float64 result and its meta block through scratch.
    value = np.zeros(8, dtype=np.float64)
    meta = np.zeros(8, dtype=np.int64)
    scratch = np.zeros(8, dtype=np.int64)
    scratch[0] = value.ctypes.data
    scratch[1] = meta.ctypes.data
    keep = (value, meta, scratch, buf)
    lib().column_cast_number(
        table.data.ctypes.data, table.cell_off.ctypes.data,
        table.cell_len.ctypes.data, table.row_cell.ctypes.data,
        table.nrows, row_start, col,
        out.vals.ctypes.data, out.isnull.ctypes.data, out.ndig.ctypes.data,
        out.expo.ctypes.data, out.special.ctypes.data,
        buf.ctypes.data, scratch.ctypes.data, vs.addr,
    )
    _ = keep
    return out


def infer_types(table, vs: ValueSets, order, row_start: int = 1) -> list[int]:
    """`agate.type_tester.TypeTester.run` over the whole table."""
    ncols = table.ncols
    order_arr = np.array(order, dtype=np.int64)
    # hypotheses, a Float64 and a meta block for the casts, and a byte buffer
    scratch = np.zeros(64, dtype=np.int64)
    fbuf = np.zeros(8, dtype=np.float64)
    buf = np.zeros(max(int(table.data.shape[0]), 1) + 8, dtype=np.uint8)
    # The kernel wants the value and meta pointers at scratch[0] and scratch[1]
    # and the hypotheses at scratch[16]; keep those addresses alive.
    scratch[0] = fbuf.ctypes.data
    meta = np.zeros(8, dtype=np.int64)
    scratch[1] = meta.ctypes.data
    scratch[2] = buf.ctypes.data
    hyp = np.zeros(max(ncols, 1), dtype=np.int64)
    scratch[16] = hyp.ctypes.data
    types = np.zeros(max(ncols, 1), dtype=np.int64)
    lib().type_tester_run(
        table.data.ctypes.data, table.cell_off.ctypes.data,
        table.cell_len.ctypes.data, table.row_cell.ctypes.data,
        table.nrows, ncols, row_start,
        order_arr.ctypes.data, len(order),
        scratch.ctypes.data, types.ctypes.data, vs.addr,
    )
    keep = (fbuf, buf, hyp, order_arr, scratch, meta)
    res = [int(t) for t in types[:ncols]]
    _ = keep
    return res, keep