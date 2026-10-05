"""Mojo port of csvkit's CSV parse and transform core.

Every function below is a transcription of the upstream function of the same
name, in upstream's source order and with upstream's branch structure. Two
dialect facts shape the code and are the only departures from upstream:

  * buffers cross the C ABI as `Int` addresses, so the wrappers in capi.mojo
    rebuild pointers and the kernels below take plain pointers;
  * a `struct` field may not have an `AnyOrigin` type in this dialect, so the
    parser/writer state that upstream keeps in `ReaderObj` / `WriterObj` lives
    in a caller-owned `Int64` scratch block and `Dialect` carries the
    lineterminator as an address rather than a `String`.

Memory layout. A parsed table is a flat arena of cells:

    data      UInt8[nbytes]   the decoded bytes of every cell, back to back
    cell_off  Int64[ncells]   start of cell i in `data`
    cell_len  Int32[ncells]   length of cell i; -1 marks an absent cell, which
                             is upstream's `None` / dst-of-range row index
    row_cell  Int64[nrows]    index of the first cell of row r; a row's cells
                             are `row_cell[r] .. row_cell[r+1]-1`, so ragged
                             rows need no padding and `row_cell[nrows-1]` is
                             the total cell count

Addressing cells by (offset, length) rather than by value is what lets the
string-shaped kernels -- `match_column_identifier`, `contains`,
`column_values_distinct`, `order_by` -- run without a Python object per cell.

Source order below follows csvkit's own: the `csv` layer csvkit delegates to,
then `csvkit/cli.py`, `csvkit/cleanup.py`, `csvkit/grep.py`, then
`csvkit/utilities/` in upstream's alphabetical order.
"""

from std.math import ceil, floor, sqrt

comptime CPtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int, AnyOrigin[mut=True]]
comptime I32Ptr = Pointer[Int32, AnyOrigin[mut=True]]
comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime U8Ptr = Pointer[UInt8, AnyOrigin[mut=True]]

# csv.QUOTE_* constants (Python's csv module).
comptime QUOTE_MINIMAL: Int = 0
comptime QUOTE_ALL: Int = 1
comptime QUOTE_NONNUMERIC: Int = 2
comptime QUOTE_NONE: Int = 3
comptime QUOTE_STRINGS: Int = 4
comptime QUOTE_NOTNULL: Int = 5

# `NOT_SET` in _csv.c. The dialect holds the delimiter, quotechar and
# escapechar as `Int` rather than `UInt8` precisely so that this -1 cannot
# collide with a NUL byte in the input, which upstream's Py_UCS4 comparison
# also cannot.
comptime NOT_SET: Int = -1

# agate's DEFAULT_NULL_VALUES, matched by `Text.cast` after strip+lower.
comptime NULL_VALUES: Int = 6

# agate's DEFAULT_TRUE_VALUES / DEFAULT_FALSE_VALUES.
comptime TRUE_VALUES: Int = 5
comptime FALSE_VALUES: Int = 5

# agate TypeTester type order, and csvkit's get_column_types insertion of
# Number into it.
comptime T_BOOLEAN: Int = 0
comptime T_NUMBER: Int = 1
comptime T_TIMEDELTA: Int = 2
comptime T_DATE: Int = 3
comptime T_DATETIME: Int = 4
comptime T_TEXT: Int = 5

# decimal.getcontext().prec, which agate's max_precision clamps against.
comptime DECIMAL_PREC: Int = 28


struct Dialect:
    var delimiter: Int
    var quotechar: Int
    var escapechar: Int
    var doublequote: Bool
    var skipinitialspace: Bool
    var quoting: Int
    var strict: Bool
    var lt_addr: Int
    var lt_len: Int

    def __init__(
        out self,
        delimiter: Int,
        quotechar: Int,
        escapechar: Int,
        doublequote: Bool,
        skipinitialspace: Bool,
        quoting: Int,
        strict: Bool,
        lt_addr: Int,
        lt_len: Int,
    ):
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.escapechar = escapechar
        self.doublequote = doublequote
        self.skipinitialspace = skipinitialspace
        self.quoting = quoting
        self.strict = strict
        self.lt_addr = lt_addr
        self.lt_len = lt_len


# `PyUnicode_FindChar(dialect->lineterminator, c, ...) >= 0`
def in_lineterminator(lt: CPtr, lt_len: Int, c: UInt8) -> Bool:
    var i = 0
    while i < lt_len:
        if lt.unsafe_load(i) == c:
            return True
        i += 1
    return False


# ============================================================ the csv module
# csvkit reads and writes through agate.csv, which is CPython's csv module,
# i.e. Modules/_csv.c. The state machine below is upstream's, case for case.

# `EOL` is upstream's dst-of-band end-of-line sentinel. Input is processed one
# byte at a time and the sentinel is passed as `c == EOL`, so it cannot collide
# with a data byte.
comptime EOL: Int = 256

# ParserState, in _csv.c order.
comptime START_RECORD: Int = 0
comptime START_FIELD: Int = 1
comptime ESCAPED_CHAR: Int = 2
comptime IN_FIELD: Int = 3
comptime IN_QUOTED_FIELD: Int = 4
comptime ESCAPE_IN_QUOTED_FIELD: Int = 5
comptime QUOTE_IN_QUOTED_FIELD: Int = 6
comptime EAT_CRNL: Int = 7
comptime AFTER_ESCAPED_CRNL: Int = 8

# Reader scratch layout.
comptime ST_STATE: Int = 0
comptime ST_FIELD_START: Int = 1
comptime ST_FIELD_LEN: Int = 2
comptime ST_UNQUOTED: Int = 3
comptime ST_NCELLS: Int = 4
comptime ST_NROWS: Int = 5
comptime ST_DATALEN: Int = 6
comptime ST_ERROR: Int = 7

# Writer scratch layout.
comptime WT_REC_LEN: Int = 0
comptime WT_QUOTED: Int = 1
comptime WT_NUM_FIELDS: Int = 2
comptime WT_NULL_FIELD: Int = 3


def parse_reset(scratch: IPtr):
    """`parse_reset`: begin a new record."""
    scratch.unsafe_store(ST_STATE, START_RECORD)
    scratch.unsafe_store(ST_FIELD_START, scratch.unsafe_load(ST_DATALEN))
    scratch.unsafe_store(ST_FIELD_LEN, 0)
    scratch.unsafe_store(ST_UNQUOTED, 0)


def parse_save_field(
    scratch: IPtr, data: CPtr, cell_off: IPtr, cell_len: I32Ptr, quoting: Int
):
    """`parse_save_field`, committing the accumulated field to the arena.

    Upstream appends a Python object to `self->fields`; here the field's
    (offset, length) pair is written instead. The `unquoted_field` /
    `field_len` / quoting conditions are upstream's unchanged.
    """
    var unquoted = scratch.unsafe_load(ST_UNQUOTED) != 0
    var field_len = scratch.unsafe_load(ST_FIELD_LEN)
    var field_start = scratch.unsafe_load(ST_FIELD_START)
    var ncells = scratch.unsafe_load(ST_NCELLS)

    if (
        unquoted
        and field_len == 0
        and (quoting == QUOTE_NOTNULL or quoting == QUOTE_STRINGS)
    ):
        # field = Py_None: an absent cell, marked by length -1.
        cell_off.unsafe_store(ncells, field_start)
        cell_len.unsafe_store(ncells, Int32(-1))
    else:
        if (
            unquoted
            and field_len != 0
            and (quoting == QUOTE_NONNUMERIC or quoting == QUOTE_STRINGS)
        ):
            # field = PyNumber_Float(field): upstream replaces the text with
            # its float value, rendered back into the field's own bytes.
            var text = String(Span(unsafe_ptr=data + field_start, length=field_len))
            try:
                var fb = String(Float64(text)).as_bytes()
                field_len = fb.__len__()
                var k = 0
                while k < fb.__len__():
                    data.unsafe_store(field_start + k, fb[k])
                    k += 1
            except:
                # Upstream propagates PyNumber_Float's ValueError dst of
                # Reader.__next__; ST_ERROR == 3 carries it to the caller.
                scratch.unsafe_store(ST_ERROR, 3)
                return
            scratch.unsafe_store(ST_DATALEN, field_start + field_len)

        cell_off.unsafe_store(ncells, field_start)
        cell_len.unsafe_store(ncells, Int32(field_len))
        scratch.unsafe_store(ST_DATALEN, field_start + field_len)

    scratch.unsafe_store(ST_NCELLS, ncells + 1)
    scratch.unsafe_store(ST_FIELD_LEN, 0)
    scratch.unsafe_store(ST_FIELD_START, scratch.unsafe_load(ST_DATALEN))


def parse_add_char(scratch: IPtr, data: CPtr, c: UInt8):
    """`parse_add_char`."""
    var field_len = scratch.unsafe_load(ST_FIELD_LEN)
    data.unsafe_store(scratch.unsafe_load(ST_DATALEN) + field_len, c)
    scratch.unsafe_store(ST_FIELD_LEN, field_len + 1)


def parse_process_char(
    scratch: IPtr,
    data: CPtr,
    cell_off: IPtr,
    cell_len: I32Ptr,
    d: Dialect,
    c: Int,
):
    """`parse_process_char`: one character through upstream's state machine."""
    var state = scratch.unsafe_load(ST_STATE)
    var byte = UInt8(0) if c == EOL else UInt8(c)
    var quoting = d.quoting

    if state == START_RECORD:
        # start of record
        if c == EOL:
            # empty line - return []
            pass
        elif c == 10 or c == 13:
            scratch.unsafe_store(ST_STATE, EAT_CRNL)
        else:
            # normal character - handle as START_FIELD
            scratch.unsafe_store(ST_STATE, START_FIELD)
            parse_process_char(scratch, data, cell_off, cell_len, d, c)
        return

    if state == START_FIELD:
        # expecting field
        scratch.unsafe_store(ST_UNQUOTED, 1)
        if c == 10 or c == 13 or c == EOL:
            # save empty field - return [fields]
            parse_save_field(scratch, data, cell_off, cell_len, quoting)
            scratch.unsafe_store(ST_STATE, START_RECORD if c == EOL else EAT_CRNL)
        elif Int(byte) == d.quotechar and quoting != QUOTE_NONE:
            # start quoted field
            scratch.unsafe_store(ST_UNQUOTED, 0)
            scratch.unsafe_store(ST_STATE, IN_QUOTED_FIELD)
        elif Int(byte) == d.escapechar:
            # possible escaped character
            scratch.unsafe_store(ST_STATE, ESCAPED_CHAR)
        elif Int(byte) == 32 and d.skipinitialspace:
            # ignore spaces at start of field
            pass
        elif Int(byte) == d.delimiter:
            # save empty field
            parse_save_field(scratch, data, cell_off, cell_len, quoting)
        else:
            # begin new unquoted field
            parse_add_char(scratch, data, byte)
            scratch.unsafe_store(ST_STATE, IN_FIELD)
        return

    if state == ESCAPED_CHAR:
        if c == 10 or c == 13:
            parse_add_char(scratch, data, byte)
            scratch.unsafe_store(ST_STATE, AFTER_ESCAPED_CRNL)
            return
        # `if (c == EOL) c = '\n';` without reassigning the parameter.
        if c == EOL:
            parse_add_char(scratch, data, UInt8(10))
        else:
            parse_add_char(scratch, data, byte)
        scratch.unsafe_store(ST_STATE, IN_FIELD)
        return

    if state == AFTER_ESCAPED_CRNL:
        if c == EOL:
            return
        # fallthru

    if state == IN_FIELD:
        # in unquoted field
        if c == 10 or c == 13 or c == EOL:
            # end of line - return [fields]
            parse_save_field(scratch, data, cell_off, cell_len, quoting)
            scratch.unsafe_store(ST_STATE, START_RECORD if c == EOL else EAT_CRNL)
        elif Int(byte) == d.escapechar:
            # possible escaped character
            scratch.unsafe_store(ST_STATE, ESCAPED_CHAR)
        elif Int(byte) == d.delimiter:
            # save field - wait for new field
            parse_save_field(scratch, data, cell_off, cell_len, quoting)
            scratch.unsafe_store(ST_STATE, START_FIELD)
        else:
            # normal character - save in field
            parse_add_char(scratch, data, byte)
        return

    if state == IN_QUOTED_FIELD:
        # in quoted field
        if c == EOL:
            pass
        elif Int(byte) == d.escapechar:
            # Possible escape character
            scratch.unsafe_store(ST_STATE, ESCAPE_IN_QUOTED_FIELD)
        elif Int(byte) == d.quotechar and quoting != QUOTE_NONE:
            if d.doublequote:
                # doublequote; " represented by ""
                scratch.unsafe_store(ST_STATE, QUOTE_IN_QUOTED_FIELD)
            else:
                # end of quote part of field
                scratch.unsafe_store(ST_STATE, IN_FIELD)
        else:
            # normal character - save in field
            parse_add_char(scratch, data, byte)
        return

    if state == ESCAPE_IN_QUOTED_FIELD:
        parse_add_char(scratch, data, UInt8(10) if c == EOL else byte)
        scratch.unsafe_store(ST_STATE, IN_QUOTED_FIELD)
        return

    if state == QUOTE_IN_QUOTED_FIELD:
        # doublequote - seen a quote in a quoted field
        if quoting != QUOTE_NONE and Int(byte) == d.quotechar:
            # save "" as "
            parse_add_char(scratch, data, byte)
            scratch.unsafe_store(ST_STATE, IN_QUOTED_FIELD)
        elif Int(byte) == d.delimiter:
            # save field - wait for new field
            parse_save_field(scratch, data, cell_off, cell_len, quoting)
            scratch.unsafe_store(ST_STATE, START_FIELD)
        elif c == 10 or c == 13 or c == EOL:
            # end of line - return [fields]
            parse_save_field(scratch, data, cell_off, cell_len, quoting)
            scratch.unsafe_store(ST_STATE, START_RECORD if c == EOL else EAT_CRNL)
        elif not d.strict:
            parse_add_char(scratch, data, byte)
            scratch.unsafe_store(ST_STATE, IN_FIELD)
        else:
            # illegal: "'%c' expected after '%c'"
            scratch.unsafe_store(ST_ERROR, 1)
        return

    if state == EAT_CRNL:
        if c == 10 or c == 13:
            pass
        elif c == EOL:
            scratch.unsafe_store(ST_STATE, START_RECORD)
        else:
            # new-line character seen in unquoted field
            scratch.unsafe_store(ST_ERROR, 2)


def csv_reader(
    src: CPtr,
    n: Int,
    scratch: IPtr,
    data: CPtr,
    cell_off: IPtr,
    cell_len: I32Ptr,
    row_cell: IPtr,
    row_line: IPtr,
    d: Dialect,
) -> Int:
    """`Reader.__next__` over the whole buffer: `csv_reader`.

    Upstream pulls one line at a time from an iterator, runs
    `parse_process_char` over its characters and then over `EOL`, and loops
    until the parser is back at START_RECORD -- one record per trip. Here the
    lines are the newlines in the buffer and the same loop runs over byte
    offsets. Returns 0, or the error code `parse_process_char` stored.
    """
    scratch.unsafe_store(ST_NCELLS, 0)
    scratch.unsafe_store(ST_NROWS, 0)
    scratch.unsafe_store(ST_DATALEN, 0)
    scratch.unsafe_store(ST_ERROR, 0)
    scratch.unsafe_store(ST_STATE, START_RECORD)

    var pos = 0
    var line_num = 0
    while pos < n:
        # Upstream resets once per record, then keeps pulling lines until the
        # parser is back at START_RECORD -- that is what lets a quoted field
        # span several physical lines.
        parse_reset(scratch)
        var row_start = scratch.unsafe_load(ST_NCELLS)
        while True:
            # One line, up to and including its terminator, which is what
            # PyIter_Next hands to Reader_iternext.
            while pos < n:
                var c = Int(src.unsafe_load(pos))
                pos += 1
                if c == 10 or c == 13:
                    line_num += 1
                parse_process_char(scratch, data, cell_off, cell_len, d, c)
                if scratch.unsafe_load(ST_ERROR) != 0:
                    return Int(scratch.unsafe_load(ST_ERROR))
                if c == 10 or c == 13:
                    break
            parse_process_char(scratch, data, cell_off, cell_len, d, EOL)
            if scratch.unsafe_load(ST_ERROR) != 0:
                return Int(scratch.unsafe_load(ST_ERROR))
            if scratch.unsafe_load(ST_STATE) == START_RECORD or pos >= n:
                break

        if pos >= n:
            # End of input: upstream saves a pending field when the record was
            # left unterminated, before returning it.
            if scratch.unsafe_load(ST_FIELD_LEN) != 0 or scratch.unsafe_load(
                ST_STATE
            ) == IN_QUOTED_FIELD:
                parse_save_field(scratch, data, cell_off, cell_len, d.quoting)
                if scratch.unsafe_load(ST_ERROR) != 0:
                    return Int(scratch.unsafe_load(ST_ERROR))

        row_cell.unsafe_store(scratch.unsafe_load(ST_NROWS), row_start)
        # `Reader.line_num` is what csvcleen's `self.reader.line_num - 1`
        # reports, so each record records the physical line it ended on.
        row_line.unsafe_store(scratch.unsafe_load(ST_NROWS), line_num)
        scratch.unsafe_store(ST_NROWS, scratch.unsafe_load(ST_NROWS) + 1)

        if pos >= n:
            break

    # Trailing bound, so the end of the last row is explicit.
    row_cell.unsafe_store(
        scratch.unsafe_load(ST_NROWS), scratch.unsafe_load(ST_NCELLS)
    )
    row_line.unsafe_store(scratch.unsafe_load(ST_NROWS), line_num)

    return Int(scratch.unsafe_load(ST_ERROR))


def join_append_data(
    d: Dialect, state: IPtr, rec: CPtr, field: CPtr, field_len: Int, copy_phase: Bool
) -> Int:
    """`join_append_data`: emit one field into the record buffer.

    `copy_phase` is upstream's flag -- the call runs once to measure the record
    and once to write it. The character classification and the quoting rules
    are upstream's, including the lineterminator membership test.
    """
    var lt = CPtr(unsafe_from_address=d.lt_addr)
    var rec_len = state.unsafe_load(WT_REC_LEN)
    var quoted = state.unsafe_load(WT_QUOTED) != 0

    if state.unsafe_load(WT_NUM_FIELDS) > 0:
        if copy_phase:
            rec.unsafe_store(rec_len, UInt8(d.delimiter))
        rec_len += 1

    # Handle preceding quote
    if copy_phase and quoted:
        rec.unsafe_store(rec_len, UInt8(d.quotechar))
        rec_len += 1

    # Copy/count field data. If field is null just pass over.
    var i = 0
    while i < field_len:
        var c = field.unsafe_load(i)
        var want_escape = False

        if (
            Int(c) == d.delimiter
            or Int(c) == d.escapechar
            or Int(c) == d.quotechar
            or Int(c) == 10
            or Int(c) == 13
            or in_lineterminator(lt, d.lt_len, c)
        ):
            if d.quoting == QUOTE_NONE:
                want_escape = True
            else:
                if Int(c) == d.quotechar:
                    if d.doublequote:
                        if copy_phase:
                            rec.unsafe_store(rec_len, UInt8(d.quotechar))
                        rec_len += 1
                    else:
                        want_escape = True
                elif Int(c) == d.escapechar:
                    want_escape = True
                if not want_escape:
                    quoted = True
            if want_escape:
                if d.escapechar == NOT_SET:
                    return -1
                if copy_phase:
                    rec.unsafe_store(rec_len, UInt8(d.escapechar))
                rec_len += 1
        # Copy field character into record buffer.
        if copy_phase:
            rec.unsafe_store(rec_len, c)
        rec_len += 1
        i += 1

    if quoted:
        if copy_phase:
            rec.unsafe_store(rec_len, UInt8(d.quotechar))
            rec_len += 1
        else:
            rec_len += 1  # starting quote
            rec_len += 1  # ending quote

    state.unsafe_store(WT_QUOTED, 1 if quoted else 0)
    return rec_len


def csv_writerow(
    d: Dialect,
    state: IPtr,
    rec: CPtr,
    data: CPtr,
    cells: IPtr,
    lens: I32Ptr,
    ncells_in_row: Int,
) -> Int:
    """`csv_writerow` for one row: `csv_writerow`.

    `cells[i]` is an offset into `data` and `lens[i]` a length; `lens[i] < 0`
    is upstream's `field == Py_None`. Returns 0, or -1 for the error cases
    upstream raises.
    """
    var lt = CPtr(unsafe_from_address=d.lt_addr)
    # join_reset
    state.unsafe_store(WT_REC_LEN, 0)
    state.unsafe_store(WT_NUM_FIELDS, 0)
    state.unsafe_store(WT_QUOTED, 0)

    var i = 0
    while i < ncells_in_row:
        var is_null = lens.unsafe_load(i) < 0
        var quoted: Bool

        if d.quoting == QUOTE_NONNUMERIC:
            # `quoted = !PyNumber_Check(field)`: every cell here is text, and
            # None is not a number either, so this mode always quotes.
            quoted = True
        elif d.quoting == QUOTE_ALL:
            quoted = True
        elif d.quoting == QUOTE_STRINGS:
            quoted = not is_null
        elif d.quoting == QUOTE_NOTNULL:
            quoted = not is_null
        else:
            quoted = False

        state.unsafe_store(WT_NULL_FIELD, 1 if is_null else 0)

        # `!field_len` is true for a null field as well as an empty one.
        if (
            lens.unsafe_load(i) <= 0
            and d.delimiter == 32
            and d.skipinitialspace
        ):
            if d.quoting == QUOTE_NONE or (
                is_null
                and (d.quoting == QUOTE_STRINGS or d.quoting == QUOTE_NOTNULL)
            ):
                return -2
            quoted = True

        var field_len = 0 if is_null else Int(lens.unsafe_load(i))
        var field = data.unsafe_offset(cells.unsafe_load(i))

        # Upstream calls join_append_data twice: measure, then copy. `rec_len`
        # stays at the record length from before this field, so both passes
        # start from the same place; only `quoted` carries over.
        state.unsafe_store(WT_QUOTED, 1 if quoted else 0)
        var r = join_append_data(d, state, rec, field, field_len, False)
        if r < 0:
            return -1
        r = join_append_data(d, state, rec, field, field_len, True)
        if r < 0:
            return -1
        state.unsafe_store(WT_REC_LEN, r)
        state.unsafe_store(WT_NUM_FIELDS, state.unsafe_load(WT_NUM_FIELDS) + 1)
        i += 1

    if state.unsafe_load(WT_NUM_FIELDS) > 0 and state.unsafe_load(WT_REC_LEN) == 0:
        if d.quoting == QUOTE_NONE or (
            state.unsafe_load(WT_NULL_FIELD) != 0
            and (d.quoting == QUOTE_STRINGS or d.quoting == QUOTE_NOTNULL)
        ):
            return -2
        state.unsafe_store(WT_NUM_FIELDS, state.unsafe_load(WT_NUM_FIELDS) - 1)
        state.unsafe_store(WT_QUOTED, 1)
        var r = join_append_data(d, state, rec, data, 0, False)
        r = join_append_data(d, state, rec, data, 0, True)
        state.unsafe_store(WT_REC_LEN, r)
        state.unsafe_store(WT_NUM_FIELDS, state.unsafe_load(WT_NUM_FIELDS) + 1)

    # Add line terminator.
    var rec_len = state.unsafe_load(WT_REC_LEN)
    var j = 0
    while j < d.lt_len:
        rec.unsafe_store(rec_len + j, lt.unsafe_load(j))
        j += 1
    state.unsafe_store(WT_REC_LEN, rec_len + d.lt_len)

    return 0




def rowcell(row_cell: IPtr, r: Int) -> Int:
    """`row_cell[r]`: the index of row r's first cell."""
    return row_cell.unsafe_load(r)


def ncells_of(row_cell: IPtr, nrows: Int) -> Int:
    """The total cell count, which `row_cell[nrows-1]` already holds."""
    if nrows == 0:
        return 0
    return row_cell.unsafe_load(nrows - 1)


def cell_is_null(cell_len: I32Ptr, i: Int) -> Bool:
    """`row[i] is None` / `i >= len(row)` collapsed to one test."""
    return cell_len.unsafe_load(i) < 0


def cell_text(data: CPtr, cell_off: IPtr, cell_len: I32Ptr, i: Int) -> String:
    """The decoded text of cell i."""
    var n = Int(cell_len.unsafe_load(i))
    if n < 0:
        return ""
    return String(
        Span(unsafe_ptr=data + cell_off.unsafe_load(i), length=n)
    )


comptime CAST_FAIL: Int = 0
comptime CAST_OK: Int = 1
comptime CAST_NULL: Int = 2
comptime CAST_FALSE: Int = 3
comptime CAST_TRUE: Int = 4

# `meta` layout shared by number_cast.
comptime META_STATUS: Int = 0
comptime META_NDIGITS: Int = 1
comptime META_EXPONENT: Int = 2
comptime META_SPECIAL: Int = 3


struct ValueSets:
    var nulls: Int
    var null_off: Int
    var null_len: Int
    var nnulls: Int
    var true_vals: Int
    var true_off: Int
    var true_len: Int
    var ntrue: Int
    var false_vals: Int
    var false_off: Int
    var false_len: Int
    var nfalse: Int
    var currency: Int
    var cur_off: Int
    var cur_len: Int
    var ncur: Int
    var group_symbol: Int
    var decimal_symbol: Int
    var no_leading_zeroes: Bool

    def __init__(
        out self,
        nulls: Int,
        null_off: Int,
        null_len: Int,
        nnulls: Int,
        true_vals: Int,
        true_off: Int,
        true_len: Int,
        ntrue: Int,
        false_vals: Int,
        false_off: Int,
        false_len: Int,
        nfalse: Int,
        currency: Int,
        cur_off: Int,
        cur_len: Int,
        ncur: Int,
        group_symbol: Int,
        decimal_symbol: Int,
        no_leading_zeroes: Bool,
    ):
        self.nulls = nulls
        self.null_off = null_off
        self.null_len = null_len
        self.nnulls = nnulls
        self.true_vals = true_vals
        self.true_off = true_off
        self.true_len = true_len
        self.ntrue = ntrue
        self.false_vals = false_vals
        self.false_off = false_off
        self.false_len = false_len
        self.nfalse = nfalse
        self.currency = currency
        self.cur_off = cur_off
        self.cur_len = cur_len
        self.ncur = ncur
        self.group_symbol = group_symbol
        self.decimal_symbol = decimal_symbol
        self.no_leading_zeroes = no_leading_zeroes


def ascii_space(c: UInt8) -> Bool:
    """Python's `str.strip()` whitespace, restricted to ASCII."""
    return (
        c == UInt8(32)
        or c == UInt8(9)
        or c == UInt8(10)
        or c == UInt8(11)
        or c == UInt8(12)
        or c == UInt8(13)
    )


def list_contains(
    vals: CPtr,
    off: IPtr,
    lens: IPtr,
    n: Int,
    s: CPtr,
    m: Int,
    lower_s: Bool,
) -> Bool:
    """`value in null_values`, over a list addressed by an offset table."""
    var i = 0
    while i < n:
        if lens.unsafe_load(i) == m:
            var a = vals + off.unsafe_load(i)
            var j = 0
            var same = True
            while j < m:
                var ca = a.unsafe_load(j)
                var cb = s.unsafe_load(j)
                if lower_s and cb >= UInt8(65) and cb <= UInt8(90):
                    cb = cb + UInt8(32)
                if ca != cb:
                    same = False
                    break
                j += 1
            if same:
                return True
        i += 1
    return False


def strip_ends(src: CPtr, n: Int) -> Tuple[Int, Int]:
    """`s.strip()`, as an inclusive-exclusive byte range."""
    var lo = 0
    var hi = n
    while lo < hi and ascii_space(src.unsafe_load(lo)):
        lo += 1
    while hi > lo and ascii_space(src.unsafe_load(hi - 1)):
        hi -= 1
    return lo, hi


def pow10i(k: Int) -> Float64:
    """10**k by repeated multiplication, which keeps this free of any table
    indexing the compiler has to constant-fold."""
    var neg = k < 0
    var e = -k if neg else k
    var acc = 1.0
    var i = 0
    while i < e:
        acc *= 10.0
        i += 1
    return 1.0 / acc if neg else acc


def is_special_number(src: CPtr, n: Int) -> Bool:
    """The `Infinity` / `NaN` spellings Python's Decimal constructor accepts."""
    var s = String(Span(unsafe_ptr=src, length=n)).lower()
    return (
        s == "infinity"
        or s == "inf"
        or s == "nan"
        or s == "snan"
        or s == "+infinity"
        or s == "+inf"
        or s == "+nan"
        or s == "+snan"
        or s == "-infinity"
        or s == "-inf"
        or s == "-nan"
        or s == "-snan"
    )


# `Decimal(s)`: an optional sign, digits with an optional point, an optional
# exponent, and the special spellings. False is agate's InvalidOperation.
def decimal_parse(src: CPtr, n: Int, value: FPtr, meta: IPtr) -> Bool:
    var lo = 0
    var hi = n
    var neg = False
    if lo < hi and (src.unsafe_load(lo) == UInt8(45) or src.unsafe_load(lo) == UInt8(43)):
        neg = src.unsafe_load(lo) == UInt8(45)
        lo += 1

    if is_special_number(src + lo, hi - lo):
        value.unsafe_store(0, 0.0)
        meta.unsafe_store(META_NDIGITS, 1)
        meta.unsafe_store(META_EXPONENT, 0)
        meta.unsafe_store(META_SPECIAL, 1)
        return True

    var dot = -1
    var exp_lo = -1
    var i = lo
    var saw_digit = False
    var prev_digit = False
    while i < hi:
        var c = src.unsafe_load(i)
        if c >= UInt8(48) and c <= UInt8(57):
            saw_digit = True
            prev_digit = True
            i += 1
        elif c == UInt8(95) and prev_digit and i + 1 < hi:
            # PEP 515: an underscore is allowed between digits.
            prev_digit = False
            i += 1
        elif c == UInt8(46) and dot < 0:
            dot = i
            prev_digit = False
            i += 1
        elif (c == UInt8(101) or c == UInt8(69)) and saw_digit:
            exp_lo = i
            i = hi
        else:
            return False
    if not saw_digit:
        return False

    var digits_end = exp_lo if exp_lo >= 0 else hi
    var exponent = 0
    if exp_lo >= 0:
        var e_neg = False
        var k = exp_lo + 1
        if k < hi and (
            src.unsafe_load(k) == UInt8(45) or src.unsafe_load(k) == UInt8(43)
        ):
            e_neg = src.unsafe_load(k) == UInt8(45)
            k += 1
        if k >= hi:
            return False
        while k < hi:
            var c = src.unsafe_load(k)
            if c < UInt8(48) or c > UInt8(57):
                return False
            exponent = exponent * 10 + Int(c - UInt8(48))
            if exponent > 1000000:
                return False
            k += 1
        if e_neg:
            exponent = -exponent

    # Digits after the point scale the exponent down.
    if dot >= 0:
        var after = 0
        var k = dot + 1
        while k < digits_end:
            if src.unsafe_load(k) != UInt8(95):
                after += 1
            k += 1
        exponent -= after

    # Normalize the way `Decimal.normalize()` does: drop leading zeros, then
    # trailing zeros, which raise the exponent back.
    var d_lo = lo
    var d_hi = digits_end
    var all_zero = True
    var k = d_lo
    while k < d_hi:
        var c = src.unsafe_load(k)
        if c != UInt8(48) and c != UInt8(95) and c != UInt8(46):
            all_zero = False
            break
        k += 1
    if all_zero:
        value.unsafe_store(0, 0.0)
        meta.unsafe_store(META_NDIGITS, 1)
        meta.unsafe_store(META_EXPONENT, 0)
        meta.unsafe_store(META_SPECIAL, 0)
        return True

    while d_lo < d_hi and (src.unsafe_load(d_lo) == UInt8(48) or src.unsafe_load(d_lo) == UInt8(46)):
        d_lo += 1
    while (
        d_hi - 1 > d_lo
        and (src.unsafe_load(d_hi - 1) == UInt8(48) or src.unsafe_load(d_hi - 1) == UInt8(46))
    ):
        d_hi -= 1
        exponent += 1

    # Accumulate up to 19 significant digits exactly; past that the remaining
    # digits only move the exponent.
    var mant = 0.0
    var scale = exponent
    var ndig = 0
    var nsignif = 0
    var m_lo = d_lo
    while m_lo < d_hi:
        var c = src.unsafe_load(m_lo)
        # Only digits feed the mantissa; the point and PEP 515 underscores do not.
        if c != UInt8(95) and c != UInt8(46):
            nsignif += 1
            if ndig < 19:
                mant = mant * 10.0 + Float64(c - UInt8(48))
                ndig += 1
            else:
                # Past 19 digits, each further digit is another factor of ten.
                scale += 1
        m_lo += 1

    var v = mant * pow10i(scale)
    value.unsafe_store(0, -v if neg else v)
    meta.unsafe_store(META_NDIGITS, nsignif)
    meta.unsafe_store(META_EXPONENT, exponent)
    meta.unsafe_store(META_SPECIAL, 0)
    return True


def number_cast(
    src: CPtr, n: Int, vs: ValueSets, buf: CPtr, value: FPtr, meta: IPtr
) -> Int:
    """`agate.data_types.number.Number.cast`, in upstream's order.

    `meta` also carries the significand digit count and exponent of
    `value.normalize()`, which is what `agate.utils.max_precision` reads, so
    both come dst of the one parse.
    """
    var r = strip_ends(src, n)
    var lo = r[0]
    var hi = r[1]

    # `if d.lower() in self.null_values: return None`
    if list_contains(
        CPtr(unsafe_from_address=vs.nulls),
        IPtr(unsafe_from_address=vs.null_off),
        IPtr(unsafe_from_address=vs.null_len),
        vs.nnulls,
        src + lo,
        hi - lo,
        True,
    ):
        value.unsafe_store(0, 0.0)
        meta.unsafe_store(META_NDIGITS, 1)
        meta.unsafe_store(META_EXPONENT, 0)
        meta.unsafe_store(META_SPECIAL, 0)
        meta.unsafe_store(META_STATUS, CAST_NULL)
        return CAST_NULL

    # `d = d.strip('%')`
    var lo2 = lo
    while lo2 < hi and src.unsafe_load(lo2) == UInt8(37):
        lo2 += 1
    while hi > lo2 and src.unsafe_load(hi - 1) == UInt8(37):
        hi -= 1
    lo = lo2

    # Peel the sign so what follows can be scanned as a magnitude.
    var neg = False
    if hi > lo and src.unsafe_load(lo) == UInt8(45):
        neg = True
        lo += 1

    # `for symbol in self.currency_symbols: d = d.strip(symbol)`
    var cur = CPtr(unsafe_from_address=vs.currency)
    var cur_off = IPtr(unsafe_from_address=vs.cur_off)
    var cur_len = IPtr(unsafe_from_address=vs.cur_len)
    var ci = 0
    while ci < vs.ncur:
        var sym = cur + cur_off.unsafe_load(ci)
        var symn = cur_len.unsafe_load(ci)
        var peeled = True
        while peeled:
            peeled = False
            if hi - lo >= symn:
                var head_ok = True
                var tail_ok = True
                var j = 0
                while j < symn:
                    if src.unsafe_load(lo + j) != sym.unsafe_load(j):
                        head_ok = False
                    if src.unsafe_load(hi - symn + j) != sym.unsafe_load(j):
                        tail_ok = False
                    j += 1
                if head_ok:
                    lo += symn
                    peeled = True
                if tail_ok:
                    hi -= symn
                    peeled = True
        ci += 1

    # `d = d.replace(group_symbol, '')` and `d.replace(decimal_symbol, '.')`
    var w = 0
    var i = lo
    while i < hi:
        var c = src.unsafe_load(i)
        if Int(c) != vs.group_symbol:
            var out_c = c
            if vs.decimal_symbol != 46 and Int(c) == vs.decimal_symbol:
                out_c = UInt8(46)
            buf.unsafe_store(w, out_c)
            w += 1
        i += 1

    if (
        vs.no_leading_zeroes
        and w > 1
        and buf.unsafe_load(0) == UInt8(48)
        and buf.unsafe_load(1) != UInt8(46)
    ):
        meta.unsafe_store(META_STATUS, CAST_FAIL)
        return CAST_FAIL

    if not decimal_parse(buf, w, value, meta):
        meta.unsafe_store(META_STATUS, CAST_FAIL)
        return CAST_FAIL

    if neg:
        value.unsafe_store(0, -value.unsafe_load(0))
    meta.unsafe_store(META_STATUS, CAST_OK)
    return CAST_OK


def boolean_cast(src: CPtr, n: Int, vs: ValueSets, buf: CPtr) -> Int:
    """`agate.data_types.boolean.Boolean.cast`: fail, null, false or true."""
    # `d = d.replace(',', '').strip()`, then matched case-insensitively.
    var w = 0
    var i = 0
    while i < n:
        var c = src.unsafe_load(i)
        if c != UInt8(44):
            var folded = c
            if c >= UInt8(65) and c <= UInt8(90):
                folded = c + UInt8(32)
            buf.unsafe_store(w, folded)
            w += 1
        i += 1
    var r = strip_ends(buf, w)
    var b = buf + r[0]
    var m = r[1] - r[0]

    if list_contains(
        CPtr(unsafe_from_address=vs.nulls),
        IPtr(unsafe_from_address=vs.null_off),
        IPtr(unsafe_from_address=vs.null_len),
        vs.nnulls,
        b,
        m,
        False,
    ):
        return CAST_NULL
    if list_contains(
        CPtr(unsafe_from_address=vs.true_vals),
        IPtr(unsafe_from_address=vs.true_off),
        IPtr(unsafe_from_address=vs.true_len),
        vs.ntrue,
        b,
        m,
        False,
    ):
        return CAST_TRUE
    if list_contains(
        CPtr(unsafe_from_address=vs.false_vals),
        IPtr(unsafe_from_address=vs.false_off),
        IPtr(unsafe_from_address=vs.false_len),
        vs.nfalse,
        b,
        m,
        False,
    ):
        return CAST_FALSE
    return CAST_FAIL


def text_cast(src: CPtr, n: Int, vs: ValueSets) -> Int:
    """`agate.data_types.text.Text.cast`, reduced to its null test."""
    var r = strip_ends(src, n)
    return (
        CAST_NULL
        if list_contains(
            CPtr(unsafe_from_address=vs.nulls),
            IPtr(unsafe_from_address=vs.null_off),
            IPtr(unsafe_from_address=vs.null_len),
            vs.nnulls,
            src + r[0],
            r[1] - r[0],
            True,
        )
        else CAST_OK
    )


def test_value(
    t: Int, p: CPtr, clen: Int, vs: ValueSets, buf: CPtr, value: FPtr, meta: IPtr
) -> Bool:
    """`DataType.test`, which is `cast` with its `CastError` swallowed.

    TimeDelta, Date and DateTime have no port, so they always fail their test
    and a column they would have claimed falls through to Text. See the README.
    """
    if t == T_BOOLEAN:
        return boolean_cast(p, clen, vs, buf) != CAST_FAIL
    if t == T_NUMBER:
        return (
            number_cast(p, clen, vs, buf, value, meta)
            != CAST_FAIL
        )
    if t == T_TEXT:
        return text_cast(p, clen, vs) != CAST_FAIL
    return False


def type_tester_run(
    data: CPtr,
    cell_off: IPtr,
    cell_len: I32Ptr,
    row_cell: IPtr,
    nrows: Int,
    ncols: Int,
    row_start: Int,
    order: IPtr,
    ntypes: Int,
    vs: ValueSets,
    scratch: IPtr,
    out_types: IPtr,
) -> Int:
    """`agate.type_tester.TypeTester.run`.

    A column keeps a type while every one of its values tests true against it,
    so the surviving hypothesis is the intersection over the column and the
    chosen type is the first still standing in preference order. The pass stays
    serial: a type dies as soon as any one value fails it, so rows cannot be
    evaluated independently without a merge.
    """
    var hypotheses = IPtr(unsafe_from_address=scratch.unsafe_load(16))
    var value = FPtr(unsafe_from_address=scratch.unsafe_load(0))
    var meta = IPtr(unsafe_from_address=scratch.unsafe_load(1))
    var buf = CPtr(unsafe_from_address=scratch.unsafe_load(2))

    var c = 0
    while c < ncols:
        hypotheses.unsafe_store(c, (1 << ntypes) - 1)
        c += 1

    var r = row_start
    while r < nrows:
        tester_row(
            data, cell_off, cell_len, row_cell, r, ncols, order, ntypes, vs,
            hypotheses, value, meta, buf,
        )
        r += 1

    # "Select in prefer order": the first type in `order` still hypothesised.
    c = 0
    while c < ncols:
        var h = hypotheses.unsafe_load(c)
        var chosen = T_TEXT
        var k = 0
        while k < ntypes:
            if (h >> k) & 1 == 1:
                chosen = order.unsafe_load(k)
                break
            k += 1
        out_types.unsafe_store(c, chosen)
        c += 1

    return 0


def tester_row(
    data: CPtr,
    cell_off: IPtr,
    cell_len: I32Ptr,
    row_cell: IPtr,
    r: Int,
    ncols: Int,
    order: IPtr,
    ntypes: Int,
    vs: ValueSets,
    hypotheses: IPtr,
    value: FPtr,
    meta: IPtr,
    buf: CPtr,
):
    """One row's contribution to each column's hypothesis."""
    var base = row_cell.unsafe_load(r)
    var rend = row_cell.unsafe_load(r + 1)
    var c = 0
    while c < ncols and c < rend - base:
        var h = hypotheses.unsafe_load(c)
        if h == 1:
            c += 1
            continue
        var i = base + c
        var clen = Int(cell_len.unsafe_load(i))
        if clen < 0:
            # `if len(row) > i` is false for a missing cell, so a short row
            # leaves this column's hypothesis untouched.
            c += 1
            continue
        var p = data + cell_off.unsafe_load(i)
        var k = 0
        while k < ntypes and h != 0:
            if (h >> k) & 1 == 1:
                if not test_value(order.unsafe_load(k), p, clen, vs, buf, value, meta):
                    h = h & ~(1 << k)
            k += 1
        hypotheses.unsafe_store(c, h)
        c += 1


comptime AGG_STATUS: Int = 0
comptime AGG_A: Int = 1
comptime AGG_B: Int = 2
comptime AGG_C: Int = 3


# The column's typed values, once: Float64 in `vals`, a null flag in `isnull`,
# and the normalized (digit count, exponent) pair `MaxPrecision` needs.
def column_cast_number(
    data: CPtr,
    cell_off: IPtr,
    cell_len: I32Ptr,
    row_cell: IPtr,
    nrows: Int,
    row_start: Int,
    col: Int,
    vs: ValueSets,
    vals: FPtr,
    isnull: U8Ptr,
    ndig: IPtr,
    expo: IPtr,
    special: U8Ptr,
    buf: CPtr,
    scratch: IPtr,
) -> Int:
    """Cast one column with `Number.cast`, as `Table.from_csv` would."""
    var value = FPtr(unsafe_from_address=scratch.unsafe_load(0))
    var meta = IPtr(unsafe_from_address=scratch.unsafe_load(1))
    # Rows before `row_start` -- the header -- are not column values.
    var pre = 0
    while pre < row_start:
        isnull.unsafe_store(pre, UInt8(1))
        vals.unsafe_store(pre, 0.0)
        ndig.unsafe_store(pre, 1)
        expo.unsafe_store(pre, 0)
        special.unsafe_store(pre, UInt8(0))
        pre += 1

    var r = row_start
    while r < nrows:
        var base = row_cell.unsafe_load(r)
        var rend = row_cell.unsafe_load(r + 1)
        var i = base + col
        if col < rend - base and cell_len.unsafe_load(i) >= 0:
            var clen = Int(cell_len.unsafe_load(i))
            var status = number_cast(
                data + cell_off.unsafe_load(i), clen, vs, buf, value, meta
            )
            isnull.unsafe_store(r, UInt8(1 if status == CAST_NULL else 0))
            vals.unsafe_store(r, value.unsafe_load(0))
            ndig.unsafe_store(r, meta.unsafe_load(META_NDIGITS))
            expo.unsafe_store(r, meta.unsafe_load(META_EXPONENT))
            special.unsafe_store(r, UInt8(meta.unsafe_load(META_SPECIAL)))
        else:
            isnull.unsafe_store(r, UInt8(1))
            vals.unsafe_store(r, 0.0)
            ndig.unsafe_store(r, 1)
            expo.unsafe_store(r, 0)
            special.unsafe_store(r, UInt8(0))
        r += 1
    return 0


def count_run(isnull: U8Ptr, nrows: Int) -> Int:
    """`agate.aggregations.count.Count.run`: `len(values_without_nulls())`."""
    var n = 0
    var r = 0
    while r < nrows:
        if isnull.unsafe_load(r) == 0:
            n += 1
        r += 1
    return n


def has_nulls_run(isnull: U8Ptr, nrows: Int) -> Int:
    """`agate.aggregations.has_nulls.HasNulls.run`: `None in values()`."""
    var r = 0
    while r < nrows:
        if isnull.unsafe_load(r) != 0:
            return 1
        r += 1
    return 0


def min_run(
    cell_off: IPtr, cell_len: I32Ptr, row_cell: IPtr, nrows: Int, row_start: Int,
    col: Int, vals: FPtr, isnull: U8Ptr
) -> Int:
    """`agate.aggregations.min.Min.run`: `min(values_without_nulls())`.

    Returns the winning cell's index so the caller can render the original
    text, which is what upstream's Decimal would have been printed from.
    """
    var best = -1
    var r = 0
    while r < nrows:
        if isnull.unsafe_load(r) == 0:
            if best < 0 or vals.unsafe_load(r) < vals.unsafe_load(best):
                best = r
        r += 1
    if best < 0:
        return -1
    var base = row_cell.unsafe_load(best + row_start)
    return base + col


def max_run(
    cell_off: IPtr, cell_len: I32Ptr, row_cell: IPtr, nrows: Int, row_start: Int,
    col: Int, vals: FPtr, isnull: U8Ptr
) -> Int:
    """`agate.aggregations.max.Max.run`: `max(values_without_nulls())`."""
    var best = -1
    var r = 0
    while r < nrows:
        if isnull.unsafe_load(r) == 0:
            if best < 0 or vals.unsafe_load(r) > vals.unsafe_load(best):
                best = r
        r += 1
    if best < 0:
        return -1
    var base = row_cell.unsafe_load(best + row_start)
    return base + col


def sum_run(vals: FPtr, isnull: U8Ptr, nrows: Int) -> Float64:
    """`agate.aggregations.sum.Sum.run`: `sum(values_without_nulls(), 0)`.

    agate sums Decimals; this sums in binary floating point, so the parity
    tolerance in the tests is float64's, not exact. The compensation keeps the
    running error near one ulp instead of n ulps for a long column.
    """
    var total = 0.0
    var comp = 0.0
    var r = 0
    while r < nrows:
        if isnull.unsafe_load(r) == 0:
            var v = vals.unsafe_load(r)
            var t = total + v
            if abs_(total) >= abs_(v):
                comp += (total - t) + v
            else:
                comp += (v - t) + total
            total = t
        r += 1
    return total + comp


def abs_(x: Float64) -> Float64:
    return -x if x < 0.0 else x


def mean_run(vals: FPtr, isnull: U8Ptr, nrows: Int) -> Float64:
    """`agate.aggregations.mean.Mean.run`: `Sum.run(table) / len(data)`."""
    var n = count_run(isnull, nrows)
    if n == 0:
        return 0.0
    return sum_run(vals, isnull, nrows) / Float64(n)


def variance_run(vals: FPtr, isnull: U8Ptr, nrows: Int) -> Float64:
    """`agate.aggregations.variance.Variance.run`.

    `sum((n - mean) ** 2 for n in data) / (len(data) - 1)` -- the sample
    variance, in upstream's two passes: mean first, then the squared
    deviations.
    """
    var n = count_run(isnull, nrows)
    if n == 0:
        return 0.0
    var mean = mean_run(vals, isnull, nrows)
    var total = 0.0
    var r = 0
    while r < nrows:
        if isnull.unsafe_load(r) == 0:
            var d = vals.unsafe_load(r) - mean
            total += d * d
        r += 1
    return total / Float64(n - 1)


def stdev_run(vals: FPtr, isnull: U8Ptr, nrows: Int) -> Float64:
    """`agate.aggregations.stdev.StDev.run`: `Variance.run(table).sqrt()`."""
    if count_run(isnull, nrows) == 0:
        return 0.0
    return sqrt(variance_run(vals, isnull, nrows))


# `agate.aggregations.percentiles.Percentiles.run` sorts the column's
# non-null values and then fills 101 quantiles with the CDF rule. csvstat only
# ever reads quantiles[50] (through `Median`), so the kernel returns the low
# and high order statistics the rule picks for the requested percentile and
# the caller averages them on Decimals when they differ.
def percentiles_run(
    vals: FPtr,
    isnull: U8Ptr,
    nrows: Int,
    row_cell: IPtr,
    col: Int,
    keys: FPtr,
    rows: IPtr,
    percentile: Int,
    row_start: Int,
    dst: IPtr,
) -> Int:
    var n = count_run(isnull, nrows)
    if n == 0:
        dst.unsafe_store(AGG_STATUS, -1)
        return -1

    # `values_without_nulls_sorted()`: copy the non-null values with their row
    # numbers, then sort.
    var m = 0
    var r = 0
    while r < nrows:
        if isnull.unsafe_load(r) == 0:
            keys.unsafe_store(m, vals.unsafe_load(r))
            rows.unsafe_store(m, r)
            m += 1
        r += 1
    sort_pairs(keys, rows, m)

    dst.unsafe_store(AGG_STATUS, 0)
    dst.unsafe_store(AGG_A, m)
    if percentile == 0:
        dst.unsafe_store(AGG_B, row_cell.unsafe_load(rows.unsafe_load(0) + row_start) + col)
        return 0
    if percentile == 100:
        dst.unsafe_store(AGG_B, row_cell.unsafe_load(rows.unsafe_load(m - 1) + row_start) + col)
        return 0

    # `k = len(data) * (float(percentile) / 100)`
    var k = Float64(m) * (Float64(percentile) / 100.0)
    # `low = max(1, int(math.ceil(k)))`, `high = min(len(data), int(math.floor(k + 1)))`
    var low = Int(ceil(k))
    if low < 1:
        low = 1
    var high = Int(floor(k + 1.0))
    if high > m:
        high = m
    dst.unsafe_store(AGG_B, row_cell.unsafe_load(rows.unsafe_load(low - 1) + row_start) + col)
    if low == high:
        dst.unsafe_store(AGG_C, dst.unsafe_load(AGG_B))
    else:
        dst.unsafe_store(AGG_C, row_cell.unsafe_load(rows.unsafe_load(high - 1) + row_start) + col)
    return 0


def sort_pairs(keys: FPtr, rows: IPtr, n: Int):
    """An introsort over (value, row) pairs, so csvstat can keep the row that
    produced each sorted value. Upstream sorts Python objects; the ordering it
    produces is the same."""
    if n < 2:
        return
    quicksort_pairs(keys, rows, 0, n - 1)
    _ = insertion_sort_pairs


def quicksort_pairs(keys: FPtr, rows: IPtr, lo_in: Int, hi_in: Int):
    var lo = lo_in
    var hi = hi_in
    while lo < hi:
        if hi - lo < 12:
            insertion_sort_pairs(keys, rows, lo, hi)
            return
        var p = keys.unsafe_load(lo + (hi - lo) // 2)
        var i = lo
        var j = hi
        while i <= j:
            while keys.unsafe_load(i) < p:
                i += 1
            while keys.unsafe_load(j) > p:
                j -= 1
            if i <= j:
                swap_pairs(keys, rows, i, j)
                i += 1
                j -= 1
        # Recurse into the smaller half, loop on the larger one.
        if j - lo < hi - i:
            quicksort_pairs(keys, rows, lo, j)
            lo = i
        else:
            quicksort_pairs(keys, rows, i, hi)
            hi = j


def swap_pairs(keys: FPtr, rows: IPtr, a: Int, b: Int):
    var k = keys.unsafe_load(a)
    keys.unsafe_store(a, keys.unsafe_load(b))
    keys.unsafe_store(b, k)
    var r = rows.unsafe_load(a)
    rows.unsafe_store(a, rows.unsafe_load(b))
    rows.unsafe_store(b, r)


def insertion_sort_pairs(keys: FPtr, rows: IPtr, lo: Int, hi: Int):
    var i = lo + 1
    while i <= hi:
        var k = keys.unsafe_load(i)
        var r = rows.unsafe_load(i)
        var j = i - 1
        while j >= lo and keys.unsafe_load(j) > k:
            keys.unsafe_store(j + 1, keys.unsafe_load(j))
            rows.unsafe_store(j + 1, rows.unsafe_load(j))
            j -= 1
        keys.unsafe_store(j + 1, k)
        rows.unsafe_store(j + 1, r)
        i += 1


def max_length_run(
    data: CPtr, cell_off: IPtr, cell_len: I32Ptr, row_cell: IPtr, nrows: Int,
    row_start: Int, col: Int
) -> Int:
    """`agate.aggregations.max_length.MaxLength.run`: `max(len(d) for d in ...)`.

    Length is counted in characters, not bytes, so UTF-8 continuation bytes are
    skipped -- Python's `len` on a `str`.
    """
    var longest = 0
    var r = row_start
    while r < nrows:
        var base = row_cell.unsafe_load(r)
        if col < row_cell.unsafe_load(r + 1) - base:
            var i = base + col
            var n = Int(cell_len.unsafe_load(i))
            if n >= 0:
                var p = data + cell_off.unsafe_load(i)
                var chars = 0
                var k = 0
                while k < n:
                    if (p.unsafe_load(k) & UInt8(192)) != UInt8(128):
                        chars += 1
                    k += 1
                if chars > longest:
                    longest = chars
        r += 1
    return longest


def max_precision_run(
    ndig: IPtr, expo: IPtr, isnull: U8Ptr, special: U8Ptr, nrows: Int
) -> Int:
    """`agate.aggregations.max_precision.MaxPrecision.run`, which delegates to
    `agate.utils.max_precision`."""
    var max_whole_places = 1
    var max_decimal_places = 0

    var r = 0
    while r < nrows:
        if isnull.unsafe_load(r) == 0 and special.unsafe_load(r) == 0:
            # `sign, digits, exponent = value.normalize().as_tuple()`
            var digits = ndig.unsafe_load(r)
            var exponent = expo.unsafe_load(r)
            var exponent_places = -exponent
            var whole_places = digits - exponent_places

            if whole_places > max_whole_places:
                max_whole_places = whole_places
            if exponent_places > max_decimal_places:
                max_decimal_places = exponent_places
        r += 1

    if max_whole_places + max_decimal_places > DECIMAL_PREC:
        max_decimal_places = DECIMAL_PREC - max_whole_places

    return max_decimal_places


# `get_freq` is `Counter(values).most_common(freq_count)`. Counting is done in
# a caller-sized open-addressed table keyed by a 64-bit hash of the cell's
# bytes; ties are broken by first appearance, which is what
# `heapq.nlargest` over an insertion-ordered Counter does.
# ============================================================== csvkit/cli.py
def letter_name(index: Int, dst: CPtr) -> Int:
    """`agate.utils.letter_name`: `letters[i % 26] * (i // 26 + 1)`.

    The result is written into `dst`; the return value is its length.
    """
    var count = 26
    var reps = index // count + 1
    var letter = UInt8(97 + index % count)
    var w = 0
    var r = 0
    while r < reps:
        dst.unsafe_store(w, letter)
        w += 1
        r += 1
    return w


def make_default_headers(n: Int, dst: CPtr) -> Int:
    """`csvkit.cli.make_default_headers`: the `letter_name` run for 0..n-1."""
    var total = 0
    var i = 0
    while i < n:
        total += letter_name(i, dst + total)
        i += 1
    return total


def str_is_digit(p: CPtr, n: Int) -> Bool:
    """Python's `str.isdigit`, for the ASCII range a column id can be."""
    if n == 0:
        return False
    var i = 0
    while i < n:
        var c = p.unsafe_load(i)
        if c < UInt8(48) or c > UInt8(57):
            return False
        i += 1
    return True


def parse_int(p: CPtr, n: Int) -> Tuple[Int, Bool]:
    """`int(s)`, returning (value, ok). Upstream raises ValueError on failure."""
    var i = 0
    while i < n and p.unsafe_load(i) == UInt8(32):
        i += 1
    var neg = False
    if i < n and (p.unsafe_load(i) == UInt8(45) or p.unsafe_load(i) == UInt8(43)):
        neg = p.unsafe_load(i) == UInt8(45)
        i += 1
    var start = i
    var acc = 0
    while i < n:
        var c = p.unsafe_load(i)
        if c < UInt8(48) or c > UInt8(57):
            break
        acc = acc * 10 + Int(c - UInt8(48))
        i += 1
    while i < n and p.unsafe_load(i) == UInt8(32):
        i += 1
    if i != n or i == start:
        return 0, False
    return (-acc if neg else acc), True


comptime ID_NOT_A_NAME: Int = -1
comptime ID_NEGATIVE: Int = -2
comptime ID_OUT_OF_RANGE: Int = -3

comptime RESOLVE_UNKNOWN: Int = 2
comptime RESOLVE_BAD_RANGE: Int = 3
comptime RESOLVE_UNKNOWN_INDEX: Int = 4

def cell_text_eq(
    data: CPtr, cell_off: IPtr, cell_len: I32Ptr, i: Int, p: CPtr, n: Int
) -> Bool:
    if Int(cell_len.unsafe_load(i)) != n:
        return False
    var a = data + cell_off.unsafe_load(i)
    var j = 0
    while j < n:
        if a.unsafe_load(j) != p.unsafe_load(j):
            return False
        j += 1
    return True


def match_column_identifier(
    data: CPtr,
    cell_off: IPtr,
    cell_len: I32Ptr,
    header_base: Int,
    header_ncells: Int,
    id_addr: Int,
    id_len: Int,
    column_offset: Int,
) -> Int:
    """`csvkit.cli.match_column_identifier(column_names, c, column_offset)`."""
    var p = CPtr(unsafe_from_address=id_addr)

    # `if isinstance(c, str) and not c.isdigit() and c in column_names`
    if not str_is_digit(p, id_len):
        var i = 0
        while i < header_ncells:
            if cell_text_eq(data, cell_off, cell_len, header_base + i, p, id_len):
                return i
            i += 1

    # `try: c = int(c) - column_offset` / `except ValueError: raise`
    var r = parse_int(p, id_len)
    if not r[1]:
        return ID_NOT_A_NAME
    var c = r[0] - column_offset

    # Fail dst if index is 0-based
    if c < 0:
        return ID_NEGATIVE
    # Fail dst if index is dst of range
    if c >= header_ncells:
        return ID_OUT_OF_RANGE
    return c


def match_column_index(header_ncells: Int, index: Int, column_offset: Int) -> Int:
    """`match_column_identifier` for an identifier that is already an int, which
    is how the range branch calls it."""
    var c = index - column_offset
    if c < 0:
        return ID_NEGATIVE
    if c >= header_ncells:
        return ID_OUT_OF_RANGE
    return c


def resolve_column_identifier(
    data: CPtr,
    cell_off: IPtr,
    cell_len: I32Ptr,
    header_base: Int,
    header_ncells: Int,
    id_addr: Int,
    id_len: Int,
    column_offset: Int,
    ignore_unknown_columns: Bool,
    ignore_invalid_range: Bool,
    dst: IPtr,
) -> Tuple[Int, Int]:
    """`csvkit.cli._resolve_column_identifier(...)`. Returns (n, status)."""
    var p = CPtr(unsafe_from_address=id_addr)
    var m = match_column_identifier(
        data, cell_off, cell_len, header_base, header_ncells, id_addr, id_len,
        column_offset,
    )
    if m >= 0:
        dst.unsafe_store(0, m)
        return 1, 0

    # `if ':' in identifier` else `elif '-' in identifier`
    var sep = -1
    var i = 0
    while i < id_len:
        var c = p.unsafe_load(i)
        if (c == UInt8(58) or c == UInt8(45)) and sep < 0:
            sep = i
        i += 1

    if sep < 0:
        return 0, RESOLVE_UNKNOWN if not ignore_unknown_columns else 0

    var a_len = sep
    var b_len = id_len - sep - 1

    # `a = int(a) if a else 1`
    var a = 1
    if a_len > 0:
        var ra = parse_int(p, a_len)
        if not ra[1]:
            return 0, 0 if ignore_invalid_range else RESOLVE_BAD_RANGE
        a = ra[0]
    # `b = int(b) + 1 if b else len(column_names) + 1`
    var b = header_ncells + 1
    if b_len > 0:
        var rb = parse_int(p + sep + 1, b_len)
        if not rb[1]:
            return 0, 0 if ignore_invalid_range else RESOLVE_BAD_RANGE
        b = rb[0] + 1

    var w = 0
    var index = a
    while index < b:
        var mm = match_column_index(header_ncells, index, column_offset)
        if mm >= 0:
            dst.unsafe_store(w, mm)
            w += 1
        elif not ignore_invalid_range:
            return w, RESOLVE_UNKNOWN_INDEX
        index += 1
    return w, 0


def parse_column_identifiers(
    data: CPtr,
    cell_off: IPtr,
    cell_len: I32Ptr,
    header_base: Int,
    header_ncells: Int,
    ids_addr: Int,
    ids_len: Int,
    excluded_addr: Int,
    excluded_len: Int,
    column_offset: Int,
    ignore_unknown_columns: Bool,
    dst: IPtr,
    scratch: IPtr,
) -> Tuple[Int, Int]:
    """`csvkit.cli.parse_column_identifiers(...)`. Returns (count, status)."""
    if header_ncells == 0:
        return 0, 0

    var ids = CPtr(unsafe_from_address=ids_addr)
    var excluded = CPtr(unsafe_from_address=excluded_addr)
    var columns = dst
    var excludes = scratch

    # `if not ids and not excluded_columns: return range(len(column_names))`
    if ids_len == 0 and excluded_len == 0:
        var i = 0
        while i < header_ncells:
            columns.unsafe_store(i, i)
            i += 1
        return header_ncells, 0

    var ncols = 0
    if ids_len > 0:
        # `for identifier in ids.split(','): columns.extend(_resolve(...))`
        var pos = 0
        while True:
            var end = pos
            while end < ids_len and ids.unsafe_load(end) != UInt8(44):
                end += 1
            var r = resolve_column_identifier(
                data, cell_off, cell_len, header_base, header_ncells,
                ids_addr + pos, end - pos, column_offset,
                ignore_unknown_columns, ignore_unknown_columns, columns + ncols,
            )
            if r[1] != 0:
                return 0, r[1]
            ncols += r[0]
            if end >= ids_len:
                break
            pos = end + 1
    else:
        # `columns = range(len(column_names))`
        var i = 0
        while i < header_ncells:
            columns.unsafe_store(i, i)
            i += 1
        ncols = header_ncells

    var nex = 0
    if excluded_len > 0:
        # `-C` is always tolerant of unknown names; a malformed range is not.
        var pos = 0
        while True:
            var end = pos
            while end < excluded_len and excluded.unsafe_load(end) != UInt8(44):
                end += 1
            var r = resolve_column_identifier(
                data, cell_off, cell_len, header_base, header_ncells,
                excluded_addr + pos, end - pos, column_offset, True, False,
                excludes + nex,
            )
            if r[1] != 0:
                return 0, r[1]
            nex += r[0]
            if end >= excluded_len:
                break
            pos = end + 1

    # `[index for index in columns if index not in excludes]`
    var w = 0
    var i = 0
    while i < ncols:
        var keep = True
        var j = 0
        while j < nex:
            if excludes.unsafe_load(j) == columns.unsafe_load(i):
                keep = False
                break
            j += 1
        if keep:
            columns.unsafe_store(w, columns.unsafe_load(i))
            w += 1
        i += 1
    return w, 0


# ============================================================== csvkit/cleanup.py
# `csvkit.cleanup.join_rows(rows, separator)`: every row's first cell is
# appended to the running row's last cell, separated by `separator`, and the
# remaining cells are appended as they are. Offsets and lengths are written
# rather than cells, so an untouched cell can be copied verbatim and only the
# joined one needs new bytes.
def join_rows(
    data: CPtr,
    cell_off: IPtr,
    cell_len: I32Ptr,
    sep: CPtr,
    sep_len: Int,
    in_cell: IPtr,
    in_off: IPtr,
    in_len: IPtr,
    nrows: Int,
    out_data: CPtr,
    out_off: IPtr,
    out_len: I32Ptr,
) -> Tuple[Int, Int]:
    """Returns (ncells, bytes written to `out_data`)."""
    var w = 0
    var used = 0

    # `fixed_row = rows[0][:]`
    var i = 0
    var n0 = in_len.unsafe_load(0)
    while i < n0:
        var cell = in_cell.unsafe_load(in_off.unsafe_load(0) + i)
        out_off.unsafe_store(w, cell_off.unsafe_load(cell))
        out_len.unsafe_store(w, cell_len.unsafe_load(cell))
        w += 1
        i += 1

    var r = 1
    while r < nrows:
        var base = in_off.unsafe_load(r)
        var n = in_len.unsafe_load(r)
        if n == 0:
            # `if len(row) == 0: row = ['']`
            out_off.unsafe_store(w, used)
            out_len.unsafe_store(w, Int32(0))
            w += 1
        else:
            # `fixed_row[-1] += f"{separator}{row[0]}"`
            var last = out_off.unsafe_load(w - 1)
            var last_len = Int(out_len.unsafe_load(w - 1))
            var src = in_cell.unsafe_load(base)
            var src_len = Int(cell_len.unsafe_load(src))
            if src_len < 0:
                src_len = 0
            var dst = out_data + used
            var k = 0
            while k < last_len:
                dst.unsafe_store(k, data.unsafe_load(last + k))
                k += 1
            var j = 0
            while j < sep_len:
                dst.unsafe_store(last_len + j, sep.unsafe_load(j))
                j += 1
            var q = 0
            while q < src_len:
                dst.unsafe_store(last_len + sep_len + q, data.unsafe_load(cell_off.unsafe_load(src) + q))
                q += 1
            out_off.unsafe_store(w - 1, used)
            out_len.unsafe_store(w - 1, Int32(last_len + sep_len + src_len))
            used += last_len + sep_len + src_len
        # `fixed_row.extend(row[1:])`
        var q = 1
        while q < n:
            var cell = in_cell.unsafe_load(base + q)
            out_off.unsafe_store(w, cell_off.unsafe_load(cell))
            out_len.unsafe_store(w, cell_len.unsafe_load(cell))
            w += 1
            q += 1
        r += 1

    return w, used


# ============================================================== csvkit/grep.py
# `FilteringCSVReader.test_row` with every pattern a plain string, which is what
# `-m/--match` compiles to upstream (`lambda x: obj in x`). Regex and
# match-file patterns stay on the Python side; see the README.
def test_row(
    data: CPtr,
    cell_off: IPtr,
    cell_len: I32Ptr,
    row_cell: IPtr,
    nrows: Int,
    cols: IPtr,
    ncols: Int,
    needles: CPtr,
    needle_off: IPtr,
    needle_len: IPtr,
    any_match: Bool,
    inverse: Bool,
    empty: CPtr,
    keep: U8Ptr,
) -> Int:
    """One pass over the table writing a keep/drop flag per data row.

    Row 0 is the header and is always kept, as upstream returns it untested.
    """
    var r = 1
    while r < nrows:
        var base = row_cell.unsafe_load(r)
        var rend = row_cell.unsafe_load(r + 1)
        var decided = False
        var i = 0
        while i < ncols and not decided:
            # `try: value = row[idx] except IndexError: value = ''`
            var col = cols.unsafe_load(i)
            var value = empty
            var vlen = 0
            if col < rend - base:
                var cell = base + col
                vlen = Int(cell_len.unsafe_load(cell))
                if vlen < 0:
                    vlen = 0
                value = data + cell_off.unsafe_load(cell)

            var hit = contains(
                value, vlen, needles + needle_off.unsafe_load(i),
                needle_len.unsafe_load(i),
            )
            if any_match:
                if hit:
                    keep.unsafe_store(r, UInt8(0 if inverse else 1))
                    decided = True
            else:
                if not hit:
                    keep.unsafe_store(r, UInt8(1 if inverse else 0))
                    decided = True
            i += 1

        if not decided:
            # `if self.any_match: return self.inverse` / `return not inverse`
            if any_match:
                keep.unsafe_store(r, UInt8(1 if inverse else 0))
            else:
                keep.unsafe_store(r, UInt8(0 if inverse else 1))
        r += 1

    if nrows > 0:
        keep.unsafe_store(0, 1)
    return 0


def contains(hay: CPtr, hn: Int, needle: CPtr, nn: Int) -> Bool:
    """`obj in x`, the substring test `-m` compiles to."""
    if nn > hn:
        return False
    if nn == 0:
        return True
    var i = 0
    while i + nn <= hn:
        var j = 0
        while j < nn:
            if hay.unsafe_load(i + j) != needle.unsafe_load(j):
                break
            j += 1
        if j == nn:
            return True
        i += 1
    return False


def csvcut(
    data: CPtr,
    cell_off: IPtr,
    cell_len: I32Ptr,
    row_cell: IPtr,
    nrows: Int,
    row_start: Int,
    cols: IPtr,
    ncols: Int,
    delete_empty: Bool,
    out_cell: IPtr,
    out_row: IPtr,
    out_ncells: IPtr,
) -> Int:
    """Gather the selected columns of each row. Returns the row count."""
    var w = 0
    var rows = 0
    var r = row_start
    while r < nrows:
        var base = row_cell.unsafe_load(r)
        var rend = row_cell.unsafe_load(r + 1)
        out_row.unsafe_store(rows, w)
        var any = False
        var i = 0
        while i < ncols:
            var col = cols.unsafe_load(i)
            if col < rend - base:
                var cell = base + col
                out_cell.unsafe_store(w, cell)
                var n = Int(cell_len.unsafe_load(cell))
                # `any(out_row)` is truthy for "" but False for None.
                if n != 0:
                    any = True
            else:
                # upstream's `None`
                out_cell.unsafe_store(w, -1)
            w += 1
            i += 1
        # `if not self.args.delete_empty or any(out_row)`
        if not delete_empty or any:
            rows += 1
        else:
            w -= ncols
        r += 1
    out_ncells.unsafe_store(0, w)
    return rows


# ======================================================== csvkit/utilities/csvstack.py
# The stacked header is the ordered union of every input's fieldnames, and each
# row is written through a DictWriter keyed on that union, so a row's cells come
# dst in the stacked header's order and a name this input does not have is
# written as None. `csvstack_headers` builds the union; `csvstack_project` is the
# per-row gather, driven by a name-to-column map computed once on the host.
def csvstack_headers(
    data: CPtr,
    cell_off: IPtr,
    cell_len: I32Ptr,
    head_off: IPtr,
    head_len: IPtr,
    ninputs: Int,
    out_cell: IPtr,
    scratch: IPtr,
) -> Tuple[Int, Int]:
    """`if field not in headers: headers.append(field)`, over every input."""
    var w = 0
    var nseen = 0
    var f = 0
    while f < ninputs:
        var n = head_len.unsafe_load(f)
        var i = 0
        while i < n:
            var cell = head_off.unsafe_load(f) + i
            var found = False
            var k = 0
            while k < nseen:
                if cell_text_eq(
                    data, cell_off, cell_len, out_cell.unsafe_load(k),
                    data + cell_off.unsafe_load(cell),
                    Int(cell_len.unsafe_load(cell)),
                ):
                    found = True
                    break
                k += 1
            if not found:
                out_cell.unsafe_store(w, cell)
                w += 1
                nseen += 1
            i += 1
        f += 1
    _ = scratch
    return w, 0


def csvstack_project(
    cell_len: I32Ptr,
    row_cell: IPtr,
    nrows: Int,
    row_start: Int,
    colmap: IPtr,
    nstack: Int,
    group: Int,
    out_cell: IPtr,
    out_row: IPtr,
    out_ncells: IPtr,
) -> Int:
    """Reorder each row's cells into the stacked header's order."""
    var w = 0
    var rows = 0
    var r = row_start
    while r < nrows:
        var base = row_cell.unsafe_load(r)
        var rend = row_cell.unsafe_load(r + 1)
        out_row.unsafe_store(rows, w)
        if group >= 0:
            out_cell.unsafe_store(w, group)
            w += 1
        var s = 0
        while s < nstack:
            var col = colmap.unsafe_load(s)
            if col < 0 or col >= rend - base:
                out_cell.unsafe_store(w, -1)
            else:
                out_cell.unsafe_store(w, base + col)
            w += 1
            s += 1
        rows += 1
        r += 1
    out_ncells.unsafe_store(0, w)
    return rows


# ======================================================== csvkit/utilities/csvsort.py
# `agate.Table.order_by`: `sorted(enumerate(rows), key=sort_key, reverse=...)`.
# The key is the tuple of the named columns with `NullOrder()` standing in for a
# null; NullOrder sorts greater than everything but another null.
#
# `upper` is csvsort's `-i/--ignore-case`, which upper-cases string values
# before comparing. `numeric` selects numeric rather than lexicographic
# comparison, which is what a Number-typed key column needs.
def order_by(
    data: CPtr,
    cell_off: IPtr,
    cell_len: I32Ptr,
    row_cell: IPtr,
    nrows: Int,
    row_start: Int,
    cols: IPtr,
    ncols: Int,
    vals: FPtr,
    isnull: U8Ptr,
    numeric: Bool,
    upper: Bool,
    reverse: Bool,
    dst: IPtr,
    upperbuf: CPtr,
    uppercap: IPtr,
) -> Int:
    """Fill `dst` with row offsets 0..nrows-row_start-1 in sorted order."""
    var m = nrows - row_start
    var i = 0
    while i < m:
        dst.unsafe_store(i, i)
        i += 1
    if m < 2:
        return 0

    # A gap sort over the index array. It is a comparison sort, like Python's
    # `sorted`, and it needs no key materialisation.
    var gap = m // 2
    while gap > 0:
        var k = gap
        while k < m:
            var a = dst.unsafe_load(k)
            var j = k
            while j >= gap:
                var b = dst.unsafe_load(j - gap)
                # `a[j - gap] > temp` is upstream's swap condition, with `b`
                # the earlier element and `a` the one being inserted.
                if (
                    compare_rows(
                        data, cell_off, cell_len, row_cell, row_start, b, cols,
                        ncols, vals, isnull, numeric, upper, reverse, upperbuf,
                        uppercap, a, m,
                    ) > 0
                ):
                    dst.unsafe_store(j, b)
                    j -= gap
                else:
                    break
            dst.unsafe_store(j, a)
            k += 1
        gap //= 2
    return 0


# Positive when row `a` sorts after row `b`.
def compare_rows(
    data: CPtr,
    cell_off: IPtr,
    cell_len: I32Ptr,
    row_cell: IPtr,
    row_start: Int,
    a: Int,
    cols: IPtr,
    ncols: Int,
    vals: FPtr,
    isnull: U8Ptr,
    numeric: Bool,
    upper: Bool,
    reverse: Bool,
    buf: CPtr,
    bufcap: IPtr,
    b: Int,
    nrows: Int,
) -> Int:
    var ra = row_start + a
    var rb = row_start + b
    var base_a = row_cell.unsafe_load(ra)
    var end_a = row_cell.unsafe_load(ra + 1)
    var base_b = row_cell.unsafe_load(rb)
    var end_b = row_cell.unsafe_load(rb + 1)

    var i = 0
    while i < ncols:
        var col = cols.unsafe_load(i)
        var na = col < end_a - base_a and cell_len.unsafe_load(base_a + col) >= 0
        var nb = col < end_b - base_b and cell_len.unsafe_load(base_b + col) >= 0
        if not na and not nb:
            i += 1
            continue
        if not na:
            return 1 if not reverse else -1
        if not nb:
            return -1 if not reverse else 1

        var c = 0
        if numeric:
            # `vals` holds one contiguous run per key column, in key order.
            c = compare_f64(
                vals.unsafe_load(i * nrows + a), vals.unsafe_load(i * nrows + b)
            )
        else:
            var la = Int(cell_len.unsafe_load(base_a + col))
            var lb = Int(cell_len.unsafe_load(base_b + col))
            var pa = data + cell_off.unsafe_load(base_a + col)
            var pb = data + cell_off.unsafe_load(base_b + col)
            if upper:
                # Two slots per key column: the left and the right value, both
                # upper-cased, as csvsort's `ignore_case_sort` does.
                var cap = bufcap.unsafe_load(i)
                var slot_a = buf + i * cap
                la = to_upper(pa, la, slot_a)
                pa = slot_a
                var slot_b = buf + (ncols + i) * cap
                lb = to_upper(pb, lb, slot_b)
                pb = slot_b
            c = compare_bytes(pa, la, pb, lb)
        if c != 0:
            return c if not reverse else -c
        i += 1
    return 0


def compare_bytes(a: CPtr, an: Int, b: CPtr, bn: Int) -> Int:
    """Python's `str` comparison: byte order over code points, then length."""
    var n = an if an < bn else bn
    var i = 0
    while i < n:
        var ca = a.unsafe_load(i)
        var cb = b.unsafe_load(i)
        if ca != cb:
            return 1 if ca > cb else -1
        i += 1
    if an == bn:
        return 0
    return 1 if an > bn else -1


def compare_f64(a: Float64, b: Float64) -> Int:
    if a < b:
        return -1
    if a > b:
        return 1
    return 0


def to_upper(p: CPtr, n: Int, dst: CPtr) -> Int:
    var i = 0
    while i < n:
        var c = p.unsafe_load(i)
        if c >= UInt8(97) and c <= UInt8(122):
            c = c - UInt8(32)
        dst.unsafe_store(i, c)
        i += 1
    return n


comptime J_LEFT: Int = 0
comptime J_INNER: Int = 1
comptime J_FULL_OUTER: Int = 2

comptime FNV_OFFSET: UInt64 = 14695981039346656037
comptime FNV_PRIME: UInt64 = 1099511628211


def hash_bytes(p: CPtr, n: Int, seed: UInt64) -> UInt64:
    """FNV-1a, the bucket key for a run of cell bytes."""
    var h = seed
    var i = 0
    while i < n:
        h = (h ^ UInt64(p.unsafe_load(i))) * FNV_PRIME
        i += 1
    return h


def hash_row(
    data: CPtr,
    cell_off: IPtr,
    cell_len: I32Ptr,
    row_cell: IPtr,
    r: Int,
    keys: IPtr,
    nkeys: Int,
    sequential: Bool,
) -> UInt64:
    """FNV-1a over the join columns; a sequential join keys on the row number."""
    var h = FNV_OFFSET
    if sequential:
        return (h ^ UInt64(r + 1)) * FNV_PRIME
    var base = row_cell.unsafe_load(r)
    var end = row_cell.unsafe_load(r + 1)
    var k = 0
    while k < nkeys:
        var col = keys.unsafe_load(k)
        if col >= end - base:
            # A missing key cell is None; it still has to hash to itself.
            h = (h ^ UInt64(0xFFFFFFFFFFFFFFFF)) * FNV_PRIME
        else:
            var cell = base + col
            var n = Int(cell_len.unsafe_load(cell))
            if n < 0:
                h = (h ^ UInt64(0xFFFFFFFFFFFFFFFF)) * FNV_PRIME
            else:
                h = hash_bytes(data + cell_off.unsafe_load(cell), n, h)
        h = (h ^ UInt64(255)) * FNV_PRIME
        k += 1
    return h


def cell_text_eq_bytes(a: CPtr, b: CPtr, n: Int) -> Bool:
    var j = 0
    while j < n:
        if a.unsafe_load(j) != b.unsafe_load(j):
            return False
        j += 1
    return True


def keys_equal(
    left_data: CPtr,
    left_off: IPtr,
    left_len: I32Ptr,
    left_row: IPtr,
    l: Int,
    left_keys: IPtr,
    nleft_keys: Int,
    right_data: CPtr,
    right_off: IPtr,
    right_len: I32Ptr,
    right_row: IPtr,
    rr: Int,
    right_keys: IPtr,
    nright_keys: Int,
    sequential: Bool,
    left_row_start: Int,
) -> Bool:
    """The equality behind `right_hash.get(left_value)`, byte for byte."""
    if sequential:
        # `left_data = tuple(range(len(self._rows)))`: ordinals, not table rows.
        return l - left_row_start == rr
    var base_l = left_row.unsafe_load(l)
    var end_l = left_row.unsafe_load(l + 1)
    var base_r = right_row.unsafe_load(rr)
    var end_r = right_row.unsafe_load(rr + 1)
    var k = 0
    while k < nleft_keys and k < nright_keys:
        var col = left_keys.unsafe_load(k)
        var rcol = right_keys.unsafe_load(k)
        var a = -1
        var b = -1
        if col < end_l - base_l:
            a = base_l + col
        if rcol < end_r - base_r:
            b = base_r + rcol
        if a < 0 or b < 0:
            # None equals only None, which is what an absent cell must match.
            if a != b:
                return False
            k += 1
            continue
        var na = Int(left_len.unsafe_load(a))
        var nb = Int(right_len.unsafe_load(b))
        if na < 0 or nb < 0:
            if na != nb:
                return False
            k += 1
            continue
        if na != nb:
            return False
        if not cell_text_eq_bytes(
            left_data + left_off.unsafe_load(a),
            right_data + right_off.unsafe_load(b),
            na,
        ):
            return False
        k += 1
    return True


def right_width(
    row_cell: IPtr, rr: Int, right_keys: IPtr, nright_keys: Int, drop_right_key: Bool
) -> Int:
    """How many of the right row's cells the join emits."""
    var n = row_cell.unsafe_load(rr + 1) - row_cell.unsafe_load(rr)
    if not drop_right_key:
        return n
    var keep = n
    var k = 0
    while k < nright_keys:
        if right_keys.unsafe_load(k) < n:
            keep -= 1
        k += 1
    return keep


def emit_right_row(
    row_cell: IPtr,
    rr: Int,
    right_keys: IPtr,
    nright_keys: Int,
    drop_right_key: Bool,
    dst_cell: IPtr,
    w_in: Int,
) -> Int:
    """Copy the right row's cells, dropping the join key when asked."""
    var w = w_in
    var base = row_cell.unsafe_load(rr)
    var end = row_cell.unsafe_load(rr + 1)
    var k = 0
    while k < end - base:
        var skip = False
        if drop_right_key:
            var j = 0
            while j < nright_keys:
                if right_keys.unsafe_load(j) == k:
                    skip = True
                    break
                j += 1
        if not skip:
            dst_cell.unsafe_store(w, base + k)
            w += 1
        k += 1
    return w


def column_values_distinct(
    data: CPtr,
    cell_off: IPtr,
    cell_len: I32Ptr,
    row_cell: IPtr,
    nrows: Int,
    row_start: Int,
    col: Int,
    keymode: Int,
    vals: FPtr,
    nslots: Int,
    slots: IPtr,
    entry_cell: IPtr,
    entry_count: IPtr,
    entry_first: IPtr,
    entry_val: FPtr,
) -> Tuple[Int, Int]:
    """Fill the distinct-value table. Returns (distinct, cells hashed)."""
    var s = 0
    while s < nslots:
        slots.unsafe_store(s, -1)
        s += 1

    var used = 0
    var hashed = 0
    var r = row_start
    while r < nrows:
        var base = row_cell.unsafe_load(r)
        if col < row_cell.unsafe_load(r + 1) - base:
            var cell = base + col
            hashed += 1
            var n = Int(cell_len.unsafe_load(cell))
            var h = UInt64(0)
            if keymode != 0:
                h = hash_f64(vals.unsafe_load(r))
            else:
                var p = data + cell_off.unsafe_load(cell)
                h = hash_bytes(p, n if n > 0 else 0, FNV_OFFSET)
            var slot = Int(h % UInt64(nslots))
            var found = -1
            var probes = 0
            while probes <= nslots:
                var e = slots.unsafe_load(slot)
                if e == -1:
                    break
                if keymode != 0:
                    if entry_val.unsafe_load(e) == vals.unsafe_load(r):
                        found = e
                        break
                else:
                    var other = entry_cell.unsafe_load(e)
                    if n == Int(cell_len.unsafe_load(other)) and (
                        n < 0
                        or cell_text_eq_bytes(
                            data + cell_off.unsafe_load(cell),
                            data + cell_off.unsafe_load(other),
                            n,
                        )
                    ):
                        found = e
                        break
                slot += 1
                if slot == nslots:
                    slot = 0
                probes += 1
            if found >= 0:
                entry_count.unsafe_store(found, entry_count.unsafe_load(found) + 1)
            else:
                slots.unsafe_store(slot, used)
                entry_cell.unsafe_store(used, cell)
                entry_first.unsafe_store(used, cell)
                entry_count.unsafe_store(used, 1)
                entry_val.unsafe_store(used, vals.unsafe_load(r))
                used += 1
        r += 1
    return used, hashed


def hash_f64(v: Float64) -> UInt64:
    """A Float64 keyed on its bits, so equal values share a bucket."""
    var one = SIMD[DType.float64, 1](v)
    var h = UInt64(one[0])
    return (h ^ (h >> 32)) * FNV_PRIME


# `Counter(values).most_common(freq_count)`: the `freq_count` entries with the
# largest counts, ties broken by first appearance, which is what `heapq.nlargest`
# over an insertion-ordered Counter does.
def values_most_common(
    ndistinct: Int,
    entry_first: IPtr,
    entry_count: IPtr,
    freq_count: Int,
    out_cell: IPtr,
    out_count: IPtr,
) -> Int:
    var emitted = 0
    while emitted < freq_count:
        var best = -1
        var best_count = 0
        var e = 0
        while e < ndistinct:
            var c = entry_count.unsafe_load(e)
            if c > best_count:
                best_count = c
                best = e
            e += 1
        if best < 0:
            break
        out_cell.unsafe_store(emitted, entry_first.unsafe_load(best))
        out_count.unsafe_store(emitted, best_count)
        entry_count.unsafe_store(best, 0)
        emitted += 1
    return emitted


# ============================================================== csvkit/cleanup.py
# `RowChecker.checked_rows`'s tally: `for i in range(len_column_names): if i >=
# len(row) or row[i] == '': empty_counts[i] += 1`. This is the per-cell scan the
# --empty-columns and --remove-empty-columns checks rest on, and it is the part
# of the checker that scales with the size of the input.
def row_empty_counts(
    cell_len: I32Ptr,
    row_cell: IPtr,
    nrows: Int,
    ncols: Int,
    row_start: Int,
    counts: IPtr,
) -> Int:
    var r = row_start
    while r < nrows:
        var base = row_cell.unsafe_load(r)
        var end = row_cell.unsafe_load(r + 1)
        var i = 0
        while i < ncols:
            if i >= end - base:
                counts.unsafe_store(i, counts.unsafe_load(i) + 1)
            else:
                var n = Int(cell_len.unsafe_load(base + i))
                if n == 0 or n < 0:
                    counts.unsafe_store(i, counts.unsafe_load(i) + 1)
            i += 1
        r += 1
    return 0
