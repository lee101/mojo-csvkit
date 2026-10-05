"""Parity between `mojocsvkit.csv` and CPython's `csv` module.

csvkit reads and writes through `agate.csv`, which is a thin wrapper around
CPython's `csv`; these are the cases that wrapper has to agree with.
"""

import csv as pycsv
import io

import pytest

from mojocsvkit import csv as mcsv

CASES = [
    b"a,b,c\n1,2,3\n",
    b"a,b,c\n1,2,3",
    b"a,b,c\n1,2,3\n\n4,5,6\n",
    b"",
    b"\n",
    b"\n\n",
    b"a",
    b"a\n",
    b'"a","b"\n',
    b'"a,b",c\n',
    b'"a""b",c\n',
    b'"a\nb",c\n',
    b'a,"b\nc",d\n',
    b"a,b,\n",
    b",,\n",
    b"a,,c\n",
    b'"",x\n',
    b"a,b\r\nc,d\r\n",
    b"a,b\rc,d\r",
    b" a , b \n",
    b"a ,b\n",
    b"a\tb\tc\n1\t2\t3\n",
    b"a;b;c\n1;2;3\n",
    b'"unterminated\n',
    b'"unterminated"',
    b'x,"y"z,w\n',
    b"a\\,b,c\n",
    b'"a\\"b",c\n',
    b"1,2,3\n4,5\n6,7,8,9\n",
    b"caf\xc3\xa9,2\n",
    b"1,2,3\n\x00\x01,2\n",
]

DIALECTS = [
    {},
    {"delimiter": "\t"},
    {"delimiter": ";"},
    {"quotechar": "'"},
    {"doublequote": False},
    {"skipinitialspace": True},
    {"escapechar": "\\"},
    {"delimiter": ",", "quotechar": None},
    {"lineterminator": "\n"},
]


def _reference(raw: bytes, **d):
    text = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return list(pycsv.reader(io.StringIO(text, newline=""), **d))


@pytest.mark.parametrize("raw", CASES, ids=lambda b: repr(b))
@pytest.mark.parametrize("dialect", DIALECTS, ids=lambda d: str(d) or "default")
def test_reader_matches_cpython(raw, dialect):
    assert list(mcsv.reader(raw, **dialect).rows()) == _reference(raw, **dialect)


@pytest.mark.parametrize("raw", CASES, ids=lambda b: repr(b))
def test_reader_row_boundaries(raw):
    """`row_cell` bounds line up with the rows the reader produced."""
    table = mcsv.reader(raw)
    expected = _reference(raw)
    assert table.nrows == len(expected)
    assert table.row(0) == expected[0] if expected else True


WRITE_CASES = [
    [["a", "b", "c"], ["1", "2", "3"]],
    [["a,b", 'c"d', "e\nf"]],
    [[""]],
    [[]],
    [["a", None, "c"]],
    [[" leading", "trailing "]],
    [["a\r\nb"]],
    [["\r"]],
    [["a", ""]],
    [["caf\u00e9", "\u4e2d\u6587"]],
    [["a", "b", "c", "d"]],
]

WRITE_DIALECTS = [
    {},
    {"quoting": pycsv.QUOTE_ALL},
    {"quoting": pycsv.QUOTE_NONE, "escapechar": "\\"},
    {"doublequote": False, "escapechar": "\\"},
    {"delimiter": "\t", "lineterminator": "\n"},
    {"lineterminator": "\r\n"},
    {"delimiter": " ", "skipinitialspace": True},
    {"quoting": pycsv.QUOTE_NONNUMERIC},
]


@pytest.mark.parametrize("rows", WRITE_CASES, ids=lambda r: repr(r))
@pytest.mark.parametrize("dialect", WRITE_DIALECTS, ids=lambda d: str(d) or "default")
def test_writer_matches_cpython(rows, dialect):
    try:
        buf = io.StringIO()
        w = pycsv.writer(buf, **dialect)
        for row in rows:
            w.writerow(row)
        expected, error = buf.getvalue(), None
    except Exception as exc:
        expected, error = None, type(exc).__name__
    try:
        got = mcsv.write(mcsv.from_rows(rows), **dialect).decode("utf-8")
        got_error = None
    except Exception as exc:
        got, got_error = None, type(exc).__name__

    assert (expected, error) == (got, got_error)


def test_round_trip_is_stable():
    raw = b"a,b\n1,\"x,y\"\n\"q\"\"r\",z\n"
    table = mcsv.reader(raw)
    assert mcsv.write(table, lineterminator="\n") == raw


def test_line_numbers():
    raw = b"a,b\n1,2\n3,4\n"
    table = mcsv.reader(raw, line_numbers=True)
    assert table.rows().__next__() == ["line_numbers", "a", "b"]
    assert list(table.rows())[1] == ["1", "1", "2"]