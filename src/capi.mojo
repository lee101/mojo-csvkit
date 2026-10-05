"""The C ABI the Python bindings call through.

Every buffer crosses as an `Int` address, because `@export` refuses parametric
functions and a pointer with an inferred origin is parametric. Callers own all
memory, including scratch -- nothing here allocates, so nothing here can leak.
The kernels keep upstream's names; the prefix here is only so the exported
symbol names do not collide.
"""

from ported import (
    Dialect,
    ValueSets,
    column_cast_number,
    column_values_distinct,
    count_run,
    csv_reader,

    csvcut,
    csvstack_headers,
    csvstack_project,
    csv_writerow,
    has_nulls_run,
    join_rows,
    make_default_headers,
    match_column_identifier,
    max_length_run,
    max_precision_run,
    max_run,
    mean_run,
    min_run,
    order_by,
    parse_column_identifiers,
    percentiles_run,
    row_empty_counts,
    stdev_run,
    sum_run,
    test_row,
    type_tester_run,
    values_most_common,
    variance_run,
)

comptime CPtr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int, AnyOrigin[mut=True]]
comptime I32Ptr = Pointer[Int32, AnyOrigin[mut=True]]
comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime U8Ptr = Pointer[UInt8, AnyOrigin[mut=True]]


def b(addr: Int) -> CPtr:
    return CPtr(unsafe_from_address=addr)


def i64(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def i32(addr: Int) -> I32Ptr:
    return I32Ptr(unsafe_from_address=addr)


def f64(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def u8(addr: Int) -> U8Ptr:
    return U8Ptr(unsafe_from_address=addr)


def dialect(
    delimiter: Int,
    quotechar: Int,
    escapechar: Int,
    doublequote: Int,
    skipinitialspace: Int,
    quoting: Int,
    strict: Int,
    lt_addr: Int,
    lt_len: Int,
) -> Dialect:
    return Dialect(
        delimiter,
        quotechar,
        escapechar,
        doublequote != 0,
        skipinitialspace != 0,
        quoting,
        strict != 0,
        lt_addr,
        lt_len,
    )


def valuesets(
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
    no_leading_zeroes: Int,
) -> ValueSets:
    return ValueSets(
        nulls,
        null_off,
        null_len,
        nnulls,
        true_vals,
        true_off,
        true_len,
        ntrue,
        false_vals,
        false_off,
        false_len,
        nfalse,
        currency,
        cur_off,
        cur_len,
        ncur,
        group_symbol,
        decimal_symbol,
        no_leading_zeroes != 0,
    )


# The 19 ValueSets arguments arrive as one Int64 block; see
# mojocsvkit/table.py, which packs them.
def vs_from(a: IPtr) -> ValueSets:
    return valuesets(
        a.unsafe_load(0), a.unsafe_load(1), a.unsafe_load(2), a.unsafe_load(3),
        a.unsafe_load(4), a.unsafe_load(5), a.unsafe_load(6), a.unsafe_load(7),
        a.unsafe_load(8), a.unsafe_load(9), a.unsafe_load(10), a.unsafe_load(11),
        a.unsafe_load(12), a.unsafe_load(13), a.unsafe_load(14),
        a.unsafe_load(15), a.unsafe_load(16), a.unsafe_load(17),
        a.unsafe_load(18),
    )


# =============================================================== the csv module
@export("csv_reader")
def x_csv_reader(
    src: Int,
    n: Int,
    scratch: Int,
    data: Int,
    cell_off: Int,
    cell_len: Int,
    row_cell: Int,
    row_line: Int,
    delimiter: Int,
    quotechar: Int,
    escapechar: Int,
    doublequote: Int,
    skipinitialspace: Int,
    quoting: Int,
    strict: Int,
    lt_addr: Int,
    lt_len: Int,
) abi("C") -> Int:
    return csv_reader(
        b(src), n, i64(scratch), b(data), i64(cell_off), i32(cell_len),
        i64(row_cell), i64(row_line),
        dialect(
            delimiter, quotechar, escapechar, doublequote, skipinitialspace,
            quoting, strict, lt_addr, lt_len,
        ),
    )


@export("csv_writerow")
def x_csv_writerow(
    delimiter: Int,
    quotechar: Int,
    escapechar: Int,
    doublequote: Int,
    skipinitialspace: Int,
    quoting: Int,
    lt_addr: Int,
    lt_len: Int,
    state: Int,
    rec: Int,
    data: Int,
    cells: Int,
    lens: Int,
    ncells_in_row: Int,
) abi("C") -> Int:
    return csv_writerow(
        dialect(
            delimiter, quotechar, escapechar, doublequote, skipinitialspace,
            quoting, 0, lt_addr, lt_len,
        ),
        i64(state), b(rec), b(data), i64(cells), i32(lens), ncells_in_row,
    )


# ============================================================ agate/type_tester
@export("type_tester_run")
def x_type_tester_run(
    data: Int,
    cell_off: Int,
    cell_len: Int,
    row_cell: Int,
    nrows: Int,
    ncols: Int,
    row_start: Int,
    order: Int,
    ntypes: Int,
    scratch: Int,
    out_types: Int,
    args: Int,
) abi("C") -> Int:
    var a = i64(args)
    return type_tester_run(
        b(data), i64(cell_off), i32(cell_len), i64(row_cell), nrows, ncols,
        row_start, i64(order), ntypes, vs_from(a),
        i64(scratch), i64(out_types),
    )


@export("column_cast_number")
def x_column_cast_number(
    data: Int,
    cell_off: Int,
    cell_len: Int,
    row_cell: Int,
    nrows: Int,
    row_start: Int,
    col: Int,
    vals: Int,
    isnull: Int,
    ndig: Int,
    expo: Int,
    special: Int,
    buf: Int,
    scratch: Int,
    args: Int,
) abi("C") -> Int:
    var a = i64(args)
    return column_cast_number(
        b(data), i64(cell_off), i32(cell_len), i64(row_cell), nrows, row_start,
        col, vs_from(a), f64(vals), u8(isnull), i64(ndig), i64(expo),
        u8(special), b(buf), i64(scratch),
    )


# =========================================================== agate/aggregations
@export("agg_count")
def x_agg_count(isnull: Int, nrows: Int) abi("C") -> Int:
    return count_run(u8(isnull), nrows)


@export("agg_has_nulls")
def x_agg_has_nulls(isnull: Int, nrows: Int) abi("C") -> Int:
    return has_nulls_run(u8(isnull), nrows)


@export("agg_sum")
def x_agg_sum(vals: Int, isnull: Int, nrows: Int) abi("C") -> Float64:
    return sum_run(f64(vals), u8(isnull), nrows)


@export("agg_mean")
def x_agg_mean(vals: Int, isnull: Int, nrows: Int) abi("C") -> Float64:
    return mean_run(f64(vals), u8(isnull), nrows)


@export("agg_variance")
def x_agg_variance(vals: Int, isnull: Int, nrows: Int) abi("C") -> Float64:
    return variance_run(f64(vals), u8(isnull), nrows)


@export("agg_stdev")
def x_agg_stdev(vals: Int, isnull: Int, nrows: Int) abi("C") -> Float64:
    return stdev_run(f64(vals), u8(isnull), nrows)


@export("agg_min")
def x_agg_min(
    cell_off: Int, cell_len: Int, row_cell: Int, nrows: Int, row_start: Int,
    col: Int, vals: Int, isnull: Int,
) abi("C") -> Int:
    return min_run(
        i64(cell_off), i32(cell_len), i64(row_cell), nrows, row_start, col,
        f64(vals), u8(isnull),
    )


@export("agg_max")
def x_agg_max(
    cell_off: Int, cell_len: Int, row_cell: Int, nrows: Int, row_start: Int,
    col: Int, vals: Int, isnull: Int,
) abi("C") -> Int:
    return max_run(
        i64(cell_off), i32(cell_len), i64(row_cell), nrows, row_start, col,
        f64(vals), u8(isnull),
    )


@export("agg_max_length")
def x_agg_max_length(
    data: Int, cell_off: Int, cell_len: Int, row_cell: Int, nrows: Int,
    row_start: Int, col: Int,
) abi("C") -> Int:
    return max_length_run(
        b(data), i64(cell_off), i32(cell_len), i64(row_cell), nrows, row_start, col
    )


@export("agg_max_precision")
def x_agg_max_precision(ndig: Int, expo: Int, isnull: Int, special: Int, nrows: Int) abi("C") -> Int:
    return max_precision_run(i64(ndig), i64(expo), u8(isnull), u8(special), nrows)


@export("agg_median")
def x_agg_median(
    vals: Int, isnull: Int, nrows: Int, row_cell: Int, col: Int, keys: Int,
    rows: Int, percentile: Int, row_start: Int, dst: Int,
) abi("C") -> Int:
    return percentiles_run(
        f64(vals), u8(isnull), nrows, i64(row_cell), col, f64(keys), i64(rows),
        percentile, row_start, i64(dst),
    )


@export("column_values_distinct")
def x_column_values_distinct(
    data: Int, cell_off: Int, cell_len: Int, row_cell: Int, nrows: Int,
    row_start: Int, col: Int, keymode: Int, vals: Int, nslots: Int, slots: Int,
    entry_cell: Int, entry_count: Int, entry_first: Int, entry_val: Int,
    dst: Int,
) abi("C") -> Int:
    var r = column_values_distinct(
        b(data), i64(cell_off), i32(cell_len), i64(row_cell), nrows, row_start,
        col, keymode, f64(vals), nslots, i64(slots), i64(entry_cell),
        i64(entry_count), i64(entry_first), f64(entry_val),
    )
    i64(dst).unsafe_store(0, r[0])
    i64(dst).unsafe_store(1, r[1])
    return r[0]


@export("values_most_common")
def x_values_most_common(
    ndistinct: Int, entry_first: Int, entry_count: Int, freq_count: Int,
    out_cell: Int, out_count: Int,
) abi("C") -> Int:
    return values_most_common(
        ndistinct, i64(entry_first), i64(entry_count), freq_count,
        i64(out_cell), i64(out_count),
    )


# ================================================================== csvkit/cli
@export("make_default_headers")
def x_make_default_headers(n: Int, dst: Int) abi("C") -> Int:
    return make_default_headers(n, b(dst))


@export("match_column_identifier")
def x_match_column_identifier(
    data: Int, cell_off: Int, cell_len: Int, header_base: Int,
    header_ncells: Int, id_addr: Int, id_len: Int, column_offset: Int,
) abi("C") -> Int:
    return match_column_identifier(
        b(data), i64(cell_off), i32(cell_len), header_base, header_ncells,
        id_addr, id_len, column_offset,
    )


@export("parse_column_identifiers")
def x_parse_column_identifiers(
    data: Int, cell_off: Int, cell_len: Int, header_base: Int,
    header_ncells: Int, ids_addr: Int, ids_len: Int, excluded_addr: Int,
    excluded_len: Int, column_offset: Int, ignore_unknown_columns: Int,
    dst: Int, scratch: Int, status: Int,
) abi("C") -> Int:
    var r = parse_column_identifiers(
        b(data), i64(cell_off), i32(cell_len), header_base, header_ncells,
        ids_addr, ids_len, excluded_addr, excluded_len, column_offset,
        ignore_unknown_columns != 0, i64(dst), i64(scratch),
    )
    i64(status).unsafe_store(0, r[1])
    return r[0]


# ============================================================== csvkit/cleanup
@export("row_empty_counts")
def x_row_empty_counts(
    cell_len: Int, row_cell: Int, nrows: Int, ncols: Int, row_start: Int,
    counts: Int,
) abi("C") -> Int:
    return row_empty_counts(
        i32(cell_len), i64(row_cell), nrows, ncols, row_start, i64(counts)
    )


@export("join_rows")
def x_join_rows(
    data: Int, cell_off: Int, cell_len: Int, sep: Int, sep_len: Int,
    in_cell: Int, in_off: Int, in_len: Int, nrows: Int, out_data: Int,
    out_off: Int, out_len: Int, dst: Int,
) abi("C") -> Int:
    var r = join_rows(
        b(data), i64(cell_off), i32(cell_len), b(sep), sep_len, i64(in_cell),
        i64(in_off), i64(in_len), nrows, b(out_data), i64(out_off),
        i32(out_len),
    )
    i64(dst).unsafe_store(0, r[0])
    i64(dst).unsafe_store(1, r[1])
    return r[0]


# ================================================================= csvkit/grep
@export("test_row")
def x_test_row(
    data: Int, cell_off: Int, cell_len: Int, row_cell: Int, nrows: Int,
    cols: Int, ncols: Int, needles: Int, needle_off: Int, needle_len: Int,
    any_match: Int, inverse: Int, empty: Int, keep: Int,
) abi("C") -> Int:
    return test_row(
        b(data), i64(cell_off), i32(cell_len), i64(row_cell), nrows,
        i64(cols), ncols, b(needles), i64(needle_off), i64(needle_len),
        any_match != 0, inverse != 0, b(empty), u8(keep),
    )


# ============================================================ csvkit/utilities
@export("csvcut")
def x_csvcut(
    data: Int, cell_off: Int, cell_len: Int, row_cell: Int, nrows: Int,
    row_start: Int, cols: Int, ncols: Int, delete_empty: Int, out_cell: Int,
    out_row: Int, out_ncells: Int,
) abi("C") -> Int:
    return csvcut(
        b(data), i64(cell_off), i32(cell_len), i64(row_cell), nrows, row_start,
        i64(cols), ncols, delete_empty != 0, i64(out_cell), i64(out_row),
        i64(out_ncells),
    )


@export("csvstack_headers")
def x_csvstack_headers(
    data: Int, cell_off: Int, cell_len: Int, head_off: Int, head_len: Int,
    ninputs: Int, out_cell: Int, scratch: Int,
) abi("C") -> Int:
    return csvstack_headers(
        b(data), i64(cell_off), i32(cell_len), i64(head_off), i64(head_len),
        ninputs, i64(out_cell), i64(scratch),
    )[0]


@export("csvstack_project")
def x_csvstack_project(
    cell_len: Int, row_cell: Int, nrows: Int, row_start: Int, colmap: Int,
    nstack: Int, group: Int, out_cell: Int, out_row: Int, out_ncells: Int,
) abi("C") -> Int:
    return csvstack_project(
        i32(cell_len), i64(row_cell), nrows, row_start, i64(colmap), nstack,
        group, i64(out_cell), i64(out_row), i64(out_ncells),
    )


@export("order_by")
def x_order_by(
    data: Int, cell_off: Int, cell_len: Int, row_cell: Int, nrows: Int,
    row_start: Int, cols: Int, ncols: Int, vals: Int, isnull: Int,
    numeric: Int, upper: Int, reverse: Int, dst: Int, upperbuf: Int,
    uppercap: Int,
) abi("C") -> Int:
    return order_by(
        b(data), i64(cell_off), i32(cell_len), i64(row_cell), nrows, row_start,
        i64(cols), ncols, f64(vals), u8(isnull), numeric != 0, upper != 0,
        reverse != 0, i64(dst), b(upperbuf), i64(uppercap),
    )
