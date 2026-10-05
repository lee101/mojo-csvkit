"""Every covered utility, run against upstream csvkit on the same input.

The upstream package is the reference: same flags, same stdin, byte-compared
stdout (and stderr, where the utility reports errors).
"""

import io
import json
import sys

import pytest

csvkit_cli = pytest.importorskip("csvkit.cli")

from csvkit.utilities.csvclean import CSVClean as UpClean  # noqa: E402
from csvkit.utilities.csvcut import CSVCut as UpCut  # noqa: E402
from csvkit.utilities.csvformat import CSVFormat as UpFormat  # noqa: E402
from csvkit.utilities.csvgrep import CSVGrep as UpGrep  # noqa: E402
from csvkit.utilities.csvsort import CSVSort as UpSort  # noqa: E402
from csvkit.utilities.csvstack import CSVStack as UpStack  # noqa: E402
from csvkit.utilities.csvstat import CSVStat as UpStat  # noqa: E402

from mojocsvkit.utilities.csvclean import CSVClean  # noqa: E402
from mojocsvkit.utilities.csvcut import CSVCut  # noqa: E402
from mojocsvkit.utilities.csvformat import CSVFormat  # noqa: E402
from mojocsvkit.utilities.csvgrep import CSVGrep  # noqa: E402
from mojocsvkit.utilities.csvsort import CSVSort  # noqa: E402
from mojocsvkit.utilities.csvstack import CSVStack  # noqa: E402
from mojocsvkit.utilities.csvstat import CSVStat  # noqa: E402


def run(cls, argv, data, upstream=False):
    """Run a utility on `data` and return (stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    kwargs = {"output_file": out, "error_file": err}
    u = cls(argv, **kwargs)
    u.input_file = io.StringIO(data)
    try:
        u.main()
    except SystemExit:
        pass
    return out.getvalue(), err.getvalue()


SIMPLE = "a,b,c\n1,2,3\n4,5,6\n"

NUMERIC = """name,flag,num,money,mixed,nulls,frac
a,yes,1,"1,234.50",x,,1.5
b,no,2,"2,345.60",y,,2.50
c,true,3,"3,456.70",z,,3
d,0,4,"4,567.80",w,null,4.250
e,Y,5,"5,678.90",v,NA,5
"""

MESSY = "a,b,c\n1,2\n3,4,5,6\n,,\n7,8,9\n"


def compare(upstream_cls, ours_cls, argv, data):
    want, want_err = run(upstream_cls, argv, data, upstream=True)
    got, got_err = run(ours_cls, argv, data)
    assert got == want
    assert got_err == want_err


# ------------------------------------------------------------------- csvcut
@pytest.mark.parametrize("argv", [[], ["-c", "1,3"], ["-c", "2"], ["-c", "2-3"],
                                  ["-c", "b"], ["-C", "1"], ["-c", "1", "-x"]])
def test_csvcut(argv):
    compare(UpCut, CSVCut, argv, SIMPLE)


def test_csvcut_short_rows():
    compare(UpCut, CSVCut, ["-c", "1,2,3"], MESSY)


def test_csvcut_names():
    compare(UpCut, CSVCut, ["-n"], SIMPLE)


def test_csvcut_bad_column_raises():
    from mojocsvkit.exceptions import ColumnIdentifierError

    def go(cls):
        u = cls(["-c", "9"], output_file=io.StringIO())
        u.input_file = io.StringIO(SIMPLE)
        u.main()

    from csvkit.exceptions import ColumnIdentifierError as UpError

    with pytest.raises(ColumnIdentifierError):
        go(CSVCut)
    with pytest.raises(UpError):
        go(UpCut)


# ------------------------------------------------------------------ csvstat
def compare_json(upstream_cls, ours_cls, argv, data):
    """JSON output, compared numerically.

    The values are agate Decimals on one side and float64 on the other -- the
    one place the two representations differ -- so they are compared to a
    relative tolerance rather than byte for byte.
    """
    want, _ = run(upstream_cls, argv, data, upstream=True)
    got, _ = run(ours_cls, argv, data)
    assert _approx(json.loads(want), json.loads(got))


def _approx(a, b, rel=1e-9):
    if isinstance(a, dict):
        assert set(a) == set(b)
        return all(_approx(a[k], b[k], rel) for k in a)
    if isinstance(a, list):
        assert len(a) == len(b)
        return all(_approx(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == pytest.approx(b, rel=rel)
    return a == b


@pytest.mark.parametrize("argv", [["--csv"], ["--mean"], ["--median"],
                                  ["--stdev"], ["--sum"], ["--min"], ["--max"],
                                  ["--count"], ["--unique"], ["--freq"],
                                  ["--len"], ["--max-precision"], ["--type"],
                                  ["--nulls"], ["--non-nulls"]])
def test_csvstat(argv):
    compare(UpStat, CSVStat, argv, NUMERIC)


def test_csvstat_json():
    compare_json(UpStat, CSVStat, ["--json"], NUMERIC)


def test_csvstat_text_output():
    compare(UpStat, CSVStat, [], NUMERIC)


def test_csvstat_columns():
    compare(UpStat, CSVStat, ["--csv", "-c", "num"], NUMERIC)


def test_csvstat_freq_count():
    compare(UpStat, CSVStat, ["--csv", "--freq-count", "1"], NUMERIC)


# ------------------------------------------------------------------ csvgrep
@pytest.mark.parametrize("argv", [["-c", "1", "-m", "1"], ["-c", "1", "-m", "zzz"],
                                  ["-c", "2", "-m", "5"], ["-c", "1,2", "-m", "1"],
                                  ["-c", "1,2", "-m", "1", "-a"],
                                  ["-c", "1", "-m", "1", "-i"],
                                  ["-c", "3", "-r", "^[0-9]$"],
                                  ["-c", "1", "-m", "1", "-l"]])
def test_csvgrep(argv):
    compare(UpGrep, CSVGrep, argv, SIMPLE)


# ------------------------------------------------------------------ csvsort
@pytest.mark.parametrize("argv", [[], ["-c", "1"], ["-c", "1", "-r"],
                                  ["-c", "b"], ["-c", "1", "-i"], ["-c", "2"]])
def test_csvsort(argv):
    compare(UpSort, CSVSort, argv, SIMPLE)


def test_csvsort_text_and_numbers():
    data = "a,b\n3,x\n1,y\n2,z\n"
    compare(UpSort, CSVSort, ["-c", "a"], data)
    compare(UpSort, CSVSort, ["-c", "a", "-r"], data)


def test_csvsort_ignore_case():
    data = "a,b\nb,x\na,y\nB,z\n"
    compare(UpSort, CSVSort, ["-c", "a", "-i"], data)


# ----------------------------------------------------------------- csvstack
def test_csvstack(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("a,b\n1,2\n3,4\n")
    b.write_text("b,c\n5,6\n7,8\n")
    for argv in ([], ["-g", "x,y"], ["-g", "x,y", "-n", "src"]):
        paths = [str(a), str(b)]
        out, err = run(UpStack, argv + paths, "", upstream=True)
        m = run(CSVStack, argv + paths, "")
        assert m[0] == out
        assert m[1] == err


# ----------------------------------------------------------------- csvclean
@pytest.mark.parametrize("argv", [
    ["-a"], ["--length-mismatch"], ["--empty-columns"],
    ["--header-normalize-space"], ["--fill-short-rows"],
    ["--join-short-rows"],     ["-a", "--omit-error-rows"],
])
def test_csvclean(argv):
    compare(UpClean, CSVClean, argv, MESSY)


def test_csvclean_ragged(tmp_path):
    compare(UpClean, CSVClean, ["--length-mismatch"], MESSY)


# --------------------------------------------------------------- csvformat
@pytest.mark.parametrize("argv", [["-T"], ["-D", ";"], ["-U", "1"], ["-U", "3", "-P", "\\"],
                                  ["-B", "-P", "\\"], ["-A"], ["-E"], ["-M", "\r\n"]])
def test_csvformat(argv):
    compare(UpFormat, CSVFormat, argv, SIMPLE)