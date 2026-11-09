"""Decomposition profiler for mojo-csvkit ANALYZE phase.

Times each public entry point, each Mojo kernel in isolation, each Python glue
step in isolation, and baselines (upstream csvkit, numpy, plain Python) at
the same sizes as bench/bench.py. Every number printed here is measured in
this session; bench/analysis.md is derived from this output verbatim.

Run: pixi run python bench/profile.py [quick|full|wide|small]
Default full. Uses best-of-3 like bench.py unless noted.
"""
import io, os, sys, time, math, csv as pycsv, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import numpy as np

MODE = sys.argv[1] if len(sys.argv) > 1 else "full"
REPEAT = 3

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("benchmod", os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.py"))
_bm = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_bm)
table, SMALL, MEDIUM, WIDE, STACK_A, STACK_B, bench_run, timeit = _bm.table, _bm.SMALL, _bm.MEDIUM, _bm.WIDE, _bm.STACK_A, _bm.STACK_B, _bm.run, _bm.timeit

# ---- datasets (same generators as bench.py) ----
# bench.py tables already built at import; reuse lengths
def info(name, s):
    b = s.encode() if isinstance(s, str) else s
    print(f"DATA {name}: bytes={len(b)} rows={b.count(bytes([10]))} cols~{b.split(bytes([10]))[0].count(b',')+1}")

for nm, d in [("SMALL", SMALL), ("MEDIUM", MEDIUM), ("WIDE", WIDE), ("STACK_A", STACK_A)]:
    info(nm, d)

def T(fn, repeat=REPEAT):
    best = math.inf; meds=[]
    for _ in range(repeat):
        t0=time.perf_counter(); r=fn(); dt=time.perf_counter()-t0
        best=min(best,dt); meds.append(dt)
    return best, sorted(meds)[len(meds)//2], r

def ms(x): return x*1e3

# imports of mojo package
from mojocsvkit import csv as mcsv
from mojocsvkit._lib import lib
from mojocsvkit._tableutil import gather, flat_gather, concat, new_arrays
from mojocsvkit.table import ValueSets, cast_column, infer_types, get_column_types, Column
from mojocsvkit.cli import CSVKitUtility, parse_column_identifiers, make_default_headers
from mojocsvkit.utilities.csvcut import CSVCut, _cut, _render
from mojocsvkit.utilities.csvformat import CSVFormat
from mojocsvkit.utilities.csvgrep import CSVGrep, _test_row
from mojocsvkit.utilities.csvsort import CSVSort, order_by as mojo_order_by
from mojocsvkit.utilities.csvstack import CSVStack, stack_headers, project
from mojocsvkit.utilities.csvstat import CSVStat, _Stats
from mojocsvkit.cleanup import RowChecker
from mojocsvkit.utilities.csvclean import CSVClean

# upstream
from csvkit.utilities.csvcut import CSVCut as UpCut
from csvkit.utilities.csvformat import CSVFormat as UpFormat
from csvkit.utilities.csvgrep import CSVGrep as UpGrep
from csvkit.utilities.csvsort import CSVSort as UpSort
from csvkit.utilities.csvstack import CSVStack as UpStack
from csvkit.utilities.csvstat import CSVStat as UpStat
from csvkit.utilities.csvclean import CSVClean as UpClean

print(f"\nMODE={MODE} REPEAT={REPEAT}")

# ============================================================
# 1. Low-level kernels in isolation (bytes already parsed)
# ============================================================
print("\n=== 1. KERNELS IN ISOLATION (MEDIUM 100k x 10 parsed once) ===")
t_med = mcsv.reader(MEDIUM.encode())
print(f"parsed MEDIUM: nrows={t_med.nrows} ncells={t_med.ncells} data_bytes={int(t_med.data.shape[0])} cell_off_bytes={int(t_med.cell_off.nbytes)} cell_len_bytes={int(t_med.cell_len.nbytes)} row_cell_bytes={int(t_med.row_cell.nbytes)}")
print(f"  arena total B = {int(t_med.data.nbytes+t_med.cell_off.nbytes+t_med.cell_len.nbytes+t_med.row_cell.nbytes+t_med.row_line.nbytes)}")

def bench_kernel(name, fn, nbytes, flops, repeat=REPEAT):
    b, m, r = T(fn, repeat)
    fpb = (flops/nbytes) if nbytes else 0.0
    # bandwidth if memory-bound: GB/s
    bw = nbytes/ (b*1e9) if b>0 else 0
    print(f"KERNEL {name}: best={ms(b):.2f}ms med={ms(m):.2f}ms bytes={nbytes} flops~{flops} fpb={fpb:.3f} eff_BW={bw:.2f}GB/s ret={r!r:.60}")
    return b

L = lib()
vs = ValueSets()
order, vs2 = get_column_types()
# order for infer: [T_BOOLEAN, T_NUMBER, T_TEXT] -> (0,1,5)
print("order=", order)

# csv_reader alone
cap_bytes = len(MEDIUM.encode())
def do_reader():
    return mcsv.reader(MEDIUM.encode())
bench_kernel("csv_reader (incl _as_bytes+newline+allocs)", do_reader, cap_bytes, cap_bytes*3)

# raw csv_reader ctypes only, preallocated
raw = MEDIUM.encode(); n=len(raw); src=np.frombuffer(raw,dtype=np.uint8)
cap=n+2
data=np.zeros(cap,dtype=np.uint8); coff=np.zeros(cap,dtype=np.int64); clen=np.zeros(cap,dtype=np.int32)
rcell=np.zeros(cap,dtype=np.int64); rline=np.zeros(cap,dtype=np.int64); scr=np.zeros(8,dtype=np.int64)
lt=np.frombuffer(b"\r\n",dtype=np.uint8)
def raw_reader():
    return L.csv_reader(src.ctypes.data, n, scr.ctypes.data, data.ctypes.data, coff.ctypes.data, clen.ctypes.data, rcell.ctypes.data, rline.ctypes.data, 44,34,-1,1,0,0,0, lt.ctypes.data, 2)
# need correct char codes: ,=44 "=34
bench_kernel("csv_reader raw ctypes (1 call, prealloc)", raw_reader, cap_bytes, cap_bytes*3)

# csv_writerow: per-row re-entry. Time whole write() vs one row cost
def do_write():
    return mcsv.write(t_med, lineterminator="\n")
wb = bench_kernel("csv_write whole table (100k rows, per-row ctypes)", do_write, int(t_med.data.nbytes), int(t_med.data.nbytes)*4)

# single writerow timing: time write of 1-row table repeated
one = mcsv.from_rows([t_med.row(0)])
def do_write1():
    return mcsv.write(one, lineterminator="\n")
b1,_,_ = T(do_write1, repeat=1000)
print(f"  csv_writerow single-row: {ms(b1):.4f}ms/call -> per-row ctypes+tobytes overhead est. x100k = {ms(b1)*100000:.1f}ms if linear")

# type_tester / infer_types
def do_infer():
    return infer_types(t_med, vs, order)
bench_kernel("type_tester_run / infer_types (10 cols)", do_infer, int(t_med.data.nbytes), int(t_med.data.nbytes)*8)

# cast_column per column (10 cols) - time one col and all
def do_cast1():
    return cast_column(t_med, 2, vs)
bench_kernel("column_cast_number x1 col", do_cast1, int(t_med.data.nbytes)//10, int(t_med.data.nbytes)//10*6)
def do_cast_all():
    return [cast_column(t_med, c, vs) for c in range(10)]
bench_kernel("column_cast_number x10 cols", do_cast_all, int(t_med.data.nbytes), int(t_med.data.nbytes)*6)

# aggregations on a cast column (n=99999 data rows)
col = cast_column(t_med, 2, vs)
vals = col.vals[1:]; isnull = col.isnull[1:]
nn = len(vals)
def agg(name, fn, nbytes, flops):
    bench_kernel(name, fn, nbytes, flops)
agg("agg_count", lambda: L.agg_count(isnull.ctypes.data, nn), nn*1, nn)
agg("agg_has_nulls", lambda: L.agg_has_nulls(isnull.ctypes.data, nn), nn*1, nn)
agg("agg_sum (Neumaier)", lambda: L.agg_sum(vals.ctypes.data, isnull.ctypes.data, nn), nn*9, nn*6)
agg("agg_mean (=count+sum)", lambda: L.agg_mean(vals.ctypes.data, isnull.ctypes.data, nn), nn*9, nn*7)
agg("agg_variance (count+mean 2pass)", lambda: L.agg_variance(vals.ctypes.data, isnull.ctypes.data, nn), nn*9, nn*10)
agg("agg_stdev (count+variance)", lambda: L.agg_stdev(vals.ctypes.data, isnull.ctypes.data, nn), nn*9, nn*11)
agg("agg_min", lambda: L.agg_min(t_med.cell_off.ctypes.data, t_med.cell_len.ctypes.data, t_med.row_cell.ctypes.data, nn, 1, 2, vals.ctypes.data, isnull.ctypes.data), nn*9, nn*3)
agg("agg_max", lambda: L.agg_max(t_med.cell_off.ctypes.data, t_med.cell_len.ctypes.data, t_med.row_cell.ctypes.data, nn, 1, 2, vals.ctypes.data, isnull.ctypes.data), nn*9, nn*3)
agg("agg_max_length col2", lambda: L.agg_max_length(t_med.data.ctypes.data, t_med.cell_off.ctypes.data, t_med.cell_len.ctypes.data, t_med.row_cell.ctypes.data, t_med.nrows, 1, 2), int(t_med.data.nbytes)//10, int(t_med.data.nbytes)//10*2)
agg("agg_max_precision", lambda: L.agg_max_precision(col.ndig[1:].ctypes.data, col.expo[1:].ctypes.data, col.isnull[1:].ctypes.data, col.special[1:].ctypes.data, nn), nn*25, nn*4)
def do_median():
    keys=np.zeros(max(nn,1),dtype=np.float64); rows=np.zeros(max(nn,1),dtype=np.int64); out=np.zeros(4,dtype=np.int64)
    return L.agg_median(vals.ctypes.data, isnull.ctypes.data, nn, t_med.row_cell.ctypes.data, 2, keys.ctypes.data, rows.ctypes.data, 50, 1, out.ctypes.data)
bench_kernel("agg_median (copy+quicksort pairs)", do_median, nn*16, nn*int(math.log2(nn))*4)

# numpy baselines for aggregations on same column
print("\n--- numpy baselines (same vals/isnull) ---")
v = vals[isnull==0]
for nm, fn in [("np.sum", lambda: np.sum(v)), ("np.mean", lambda: np.mean(v)), ("np.std ddof1", lambda: np.std(v, ddof=1)), ("np.median", lambda: np.median(v)), ("np.min", lambda: np.min(v)), ("np.max", lambda: np.max(v))]:
    b,m,_ = T(fn); print(f"NP {nm}: best={ms(b):.3f}ms bytes~{v.nbytes} flops~{len(v)} fpb={len(v)/v.nbytes:.3f} BW={v.nbytes/(b*1e9):.1f}GB/s")
# plain python sum baseline
b,m,_=T(lambda: sum(v.tolist())); print(f"PY sum(list): best={ms(b):.2f}ms")

# csvcut kernel
cols3=np.array([0,2,4],dtype=np.int64)
capc=max(t_med.nrows*3,1)
cells=np.zeros(capc,dtype=np.int64); orows=np.zeros(capc+1,dtype=np.int64); ncell=np.zeros(2,dtype=np.int64)
def do_cutk():
    return L.csvcut(t_med.data.ctypes.data, t_med.cell_off.ctypes.data, t_med.cell_len.ctypes.data, t_med.row_cell.ctypes.data, t_med.nrows, 1, cols3.ctypes.data, 3, 0, cells.ctypes.data, orows.ctypes.data, ncell.ctypes.data)
bench_kernel("csvcut kernel (gather idx)", do_cutk, 300000*8, 300000*2)
# flat_gather alone
n_out=int(ncell[0])
def do_gather():
    return flat_gather(t_med, cells[:n_out], 3)
bench_kernel("flat_gather (Python loop 300k cells)", do_gather, 300000*8 + int(t_med.data.nbytes)//3, 0)
# _cut = kernel + gather
def do_cutfull():
    return _cut(t_med, [0,2,4], False)
bench_kernel("_cut full (kernel+gather+allocs)", do_cutfull, 300000*8, 300000*2)
# _render: row() decode + from_rows + write
cut_out = _cut(t_med, [0,2,4], False)
def do_render():
    return _render(cut_out, ["col0","col2","col4"])
bench_kernel("_render (row decode + from_rows + write)", do_render, int(cut_out.data.nbytes), int(cut_out.data.nbytes)*4)
# Table.row decode cost alone
def do_decode_all():
    return [t_med.row(r) for r in range(t_med.nrows)]
bench_kernel("Table.row() x100k (decode 1M cells)", do_decode_all, int(t_med.data.nbytes), 0)

# test_row kernel
def do_testrow():
    return _test_row(t_med, [1], "42", False, False)
bench_kernel("test_row kernel (-m 42, 1 col)", do_testrow, int(t_med.data.nbytes)//10, int(t_med.data.nbytes)//10*2)
# grep post: kept rows re-materialised
keep = _test_row(t_med, [1], "42", False, False)
print(f"  grep kept rows: {int(keep.sum())} / {t_med.nrows}")

# order_by kernel (numeric col 3? bench uses -c 3 => index 2)
# need vals contiguous for numeric
def do_orderk():
    nrows_data = t_med.nrows-1
    vals2=np.zeros(max(nrows_data,1),dtype=np.float64)
    per=cast_column(t_med,2,vs); vals2[:]=per.vals[1:nrows_data+1]
    out=np.zeros(max(nrows_data,1),dtype=np.int64); isnull2=np.zeros(max(nrows_data,1),dtype=np.uint8)
    upcap=np.array([8],dtype=np.int64); upbuf=np.zeros(24,dtype=np.uint8)
    return L.order_by(t_med.data.ctypes.data, t_med.cell_off.ctypes.data, t_med.cell_len.ctypes.data, t_med.row_cell.ctypes.data, t_med.nrows, 1, np.array([2],dtype=np.int64).ctypes.data, 1, vals2.ctypes.data, isnull2.ctypes.data, 1,0,0, out.ctypes.data, upbuf.ctypes.data, upcap.ctypes.data)
bench_kernel("order_by kernel ONLY (gap sort, numeric, incl cast)", do_orderk, nn*16, nn*int(math.log2(nn))*6)
# gap-sort complexity note: O(n^1.5?) shell sort with /2 gaps
# full order_by (infer+cast+kernel+gather+write happens in utility)
def do_orderfull():
    return mojo_order_by(t_med, [2], reverse=False, upper=False, vs=vs, order=order)
bench_kernel("order_by full (infer+cast+kernel+gather)", do_orderfull, nn*16, nn*int(math.log2(nn))*6)
# numpy argsort baseline on same numeric column
vv = cast_column(t_med,2,vs).vals[1:]
b,m,_=T(lambda: np.argsort(vv, kind="stable")); print(f"NP np.argsort 100k f64: best={ms(b):.2f}ms bytes={vv.nbytes} flops~{len(vv)*int(math.log2(len(vv)))} fpb={len(vv)*int(math.log2(len(vv)))/vv.nbytes:.2f}")
# python sorted baseline
b,m,_=T(lambda: sorted(range(len(vv)), key=lambda i: vv[i])); print(f"PY sorted(idx,key) 100k: best={ms(b):.2f}ms")

# csvstack kernels
print("\n=== STACK (2x50k x6) ===")
ta=mcsv.reader(STACK_A.encode()); tb=mcsv.reader(STACK_B.encode())
def do_stack_read():
    return [mcsv.reader(STACK_A.encode()), mcsv.reader(STACK_B.encode())]
bench_kernel("2x csv_reader (stack inputs)", do_stack_read, len(STACK_A.encode())*2, len(STACK_A.encode())*2*3)
combined, starts = concat([ta,tb])
def do_concat():
    return concat([ta,tb])
bench_kernel("concat (row decode + from_rows)", do_concat, int(combined.data.nbytes), 0)
def do_sheaders():
    return stack_headers([ta,tb])
bench_kernel("csvstack_headers kernel+union decode", do_sheaders, 12*8, 12*12)
# need colmap
union, colmaps = stack_headers([ta,tb])
def do_project():
    return project(ta, colmaps[0], union)
bench_kernel("csvstack_project (kernel + cell decode)", do_project, 50000*6*8, 50000*6*2)
def do_stack_full():
    return stack_headers([ta,tb])
# from_rows(out_rows) + write for stack: simulate utility tail
out_rows = [union] + project(ta, colmaps[0], union) + project(tb, colmaps[1], union)
def do_stack_fromrows_write():
    return mcsv.write(mcsv.from_rows(out_rows), lineterminator="\n")
bench_kernel("stack from_rows+write (100k rows)", do_stack_fromrows_write, int(combined.data.nbytes), int(combined.data.nbytes)*4)

# ============================================================
# 2. Python glue decomposition per utility (utility total = bench.py numbers)
# ============================================================
print("\n=== 2. UTILITY TOTALS (MEDIUM) ===")
for name,(ours,up),argv,data,paths in [
    ("csvcut -c 1,3,5", (CSVCut, UpCut), ["-c","1,3,5"], MEDIUM, ()),
    ("csvgrep -c 2 -m 42", (CSVGrep, UpGrep), ["-c","2","-m","42"], MEDIUM, ()),
    ("csvsort -c 3", (CSVSort, UpSort), ["-c","3"], MEDIUM, ()),
    ("csvstat --csv", (CSVStat, UpStat), ["--csv"], MEDIUM, ()),
    ("csvstat --mean", (CSVStat, UpStat), ["--mean"], MEDIUM, ()),
    ("csvstat --median", (CSVStat, UpStat), ["--median"], MEDIUM, ()),
    ("csvformat -T", (CSVFormat, UpFormat), ["-T"], MEDIUM, ()),
]:
    a,_,_ = T(lambda: bench_run(ours, argv, data, paths), repeat=2)
    b,_,_ = T(lambda: bench_run(up, argv, data, paths), repeat=2)
    print(f"UTIL {name}: mojo={ms(a):.1f}ms up={ms(b):.1f}ms ratio_up/mojo={b/a:.2f}x {'SLOWER' if a>b else 'faster'}")

# csvstat decomposition: _Stats init (infer+per-col cast lazy) vs ops
print("\n=== CSVSTAT DECOMPOSITION ===")
def do_stats_init():
    return _Stats(t_med, vs, order)
bench_kernel("_Stats.__init__ (infer_types 10 cols)", do_stats_init, int(t_med.data.nbytes), int(t_med.data.nbytes)*8)
st = _Stats(t_med, vs, order)
for op in ["type","nulls","nonnulls","min","max","sum","mean","median","stdev","len","maxprecision","freq","unique"]:
    def do_op(op=op):
        # mimic calculate_stats path via aggregate on one column
        from mojocsvkit.utilities.csvstat import aggregate
        return aggregate(st, 2, op) if False else None
    # instead time distinct/counter path directly for freq/unique
    pass
# distinct / freq host path
def do_distinct():
    return st.distinct(2)
bench_kernel("_Stats.distinct(col2) host Counter path", do_distinct, nn*8, nn*2)
st2 = _Stats(t_med, vs, order)
def do_freq_all():
    return [st2.distinct(c) for c in range(10)]
bench_kernel("_Stats.distinct x10 cols (freq/unique)", do_freq_all, int(t_med.data.nbytes), 0)
# rendered values cost
from mojocsvkit.utilities.csvstat import _rendered_values, _counter
def do_rendered():
    return _rendered_values(st, 2)
bench_kernel("_rendered_values (decode/cast render)", do_rendered, nn*8, 0)
# boolean keys path only if boolean col; text col decode
# numpy unique baseline
txt_col = [t_med.cell(int(t_med.row_cell[r])+2) for r in range(1, t_med.nrows)]
b,m,_=T(lambda: len(set(txt_col))); print(f"PY set() over 100k text cells (1 col): best={ms(b):.2f}ms")
arr = np.array(txt_col)
b,m,_=T(lambda: np.unique(arr, return_counts=True)); print(f"NP np.unique 100k text (1 col): best={ms(b):.2f}ms")

# ============================================================
# 3. Reader/writer vs Python csv + numpy baselines
# ============================================================
print("\n=== 3. BASELINES: parse + write ===")
def py_parse():
    return list(pycsv.reader(io.StringIO(MEDIUM)))
b,m,r = T(py_parse); print(f"PY csv.reader list(): best={ms(b):.2f}ms bytes={len(MEDIUM.encode())} fpb~3 BW={len(MEDIUM.encode())/(b*1e9):.2f}GB/s rows={len(r)}")
def py_write():
    rows=list(pycsv.reader(io.StringIO(MEDIUM)))
    out=io.StringIO(); w=pycsv.writer(out, lineterminator="\n"); w.writerows(rows); return out.getvalue()
b,m,_=T(py_write, repeat=2); print(f"PY csv.writer roundtrip: best={ms(b):.2f}ms")
b,m,_=T(lambda: np.genfromtxt(io.StringIO(MEDIUM), delimiter=",", skip_header=1, usecols=(1,2,3,4), dtype=float, encoding=None), repeat=2)
print(f"NP genfromtxt (4 numeric cols): best={ms(b):.2f}ms (includes parse+cast, fails on text cols in practice)")
try:
    b,m,_=T(lambda: np.loadtxt(io.StringIO(MEDIUM), delimiter=",", skiprows=1, usecols=(1,2)), repeat=2)
    print(f"NP loadtxt: best={ms(b):.2f}ms")
except Exception as e:
    print("NP loadtxt failed:", str(e)[:120])

# ============================================================
# 4. Glue specifics: per-element ctypes, allocations, copies
# ============================================================
print("\n=== 4. GLUE COUNTERS ===")
# csv_writerow calls count
print("csv_writerow: 1 ctypes call per ROW (100k calls for MEDIUM write)")
print("csv_reader: 1 ctypes call per DOCUMENT (whole buffer)")
print("csvcut/test_row/order_by/csvstack_project: 1 call per TABLE")
print("agg_*: 1 call per COLUMN x OPERATION")
print("column_cast_number: 1 call per COLUMN")
print("type_tester_run: 1 call per TABLE")
# allocations: reader allocates 5x cap=n+2 arrays
nB=len(MEDIUM.encode())
print(f"reader alloc: 5 arrays x cap={nB+2}: data 1B + cell_off 8B + cell_len 4B + row_cell/line 8B each = ~{ (1+8+4+8+8)*(nB+2)/1e6:.1f}MB zeroed per parse (np.zeros)")
b,m,_=T(lambda: np.zeros(nB+2, dtype=np.uint8)); print(f"  np.zeros(n+2) uint8 alone: {ms(b):.2f}ms")
# writer cap
print(f"writer alloc: rec cap = data + 8*ncells + 4*nrows +16 = {int(t_med.data.nbytes)+8*t_med.ncells+4*t_med.nrows+16} B zeroed per write")
# cast_column alloc: buf = data size per column!
print(f"cast_column alloc per col: buf={int(t_med.data.nbytes)+8}B + Column 5 arrays x nrows = {(8+1+8+8+1)*t_med.nrows}B; x10 cols for csvstat --csv")
# infer_types alloc
print(f"infer_types alloc: buf={int(t_med.data.nbytes)+8}B one per call")
# gather alloc: copies every cell bytes into blobs + join
cut = _cut(t_med,[0,2,4],False)
print(f"gather copy: output data={int(cut.data.nbytes)}B rebuilt via per-cell tobytes + join (300k slices)")
# from_rows alloc: str.encode per cell + join
b,m,_=T(lambda: mcsv.from_rows([t_med.row(r) for r in range(t_med.nrows)]), repeat=2)
print(f"from_rows full decoded table: best={ms(b):.1f}ms (decode+encode roundtrip)")

# ============================================================
# 5. WIDE + SMALL scaling
# ============================================================
print("\n=== 5. SCALING: SMALL / WIDE ===")
t_small = mcsv.reader(SMALL.encode())
print(f"SMALL: nrows={t_small.nrows} ncells={t_small.ncells} bytes={len(SMALL.encode())}")
for name,fn,nb in [
    ("reader SMALL", lambda: mcsv.reader(SMALL.encode()), len(SMALL.encode())),
    ("write SMALL", lambda: mcsv.write(t_small, lineterminator="\n"), int(t_small.data.nbytes)),
    ("infer SMALL", lambda: infer_types(t_small, vs, order), int(t_small.data.nbytes)),
]:
    b,m,_=T(fn); print(f"SCALE {name}: best={ms(b):.2f}ms bytes={nb} BW={nb/(b*1e9) if b else 0:.2f}GB/s")
t_wide = mcsv.reader(WIDE.encode())
print(f"WIDE: nrows={t_wide.nrows} ncells={t_wide.ncells} bytes={len(WIDE.encode())} data={int(t_wide.data.nbytes)}")
for name,fn,nb in [
    ("reader WIDE", lambda: mcsv.reader(WIDE.encode()), len(WIDE.encode())),
    ("write WIDE", lambda: mcsv.write(t_wide, lineterminator="\n"), int(t_wide.data.nbytes)),
    ("infer WIDE (60 cols)", lambda: infer_types(t_wide, vs, order), int(t_wide.data.nbytes)),
    ("cast_all WIDE x60", lambda: [cast_column(t_wide,c,vs) for c in range(60)], int(t_wide.data.nbytes)),
]:
    b,m,_=T(fn, repeat=2); print(f"SCALE {name}: best={ms(b):.2f}ms bytes={nb} BW={nb/(b*1e9) if b else 0:.2f}GB/s")

# csvclean kernel
print("\n=== CLEAN ===")
from mojocsvkit.cleanup import RowChecker as RC
def do_emptycounts():
    counts=np.zeros(10,dtype=np.int64)
    return L.row_empty_counts(t_med.cell_len.ctypes.data, t_med.row_cell.ctypes.data, t_med.nrows, 10, 1, counts.ctypes.data)
bench_kernel("row_empty_counts kernel", do_emptycounts, t_med.ncells*4, t_med.ncells)

# parse_column_identifiers (tiny: header only)
def do_pci():
    return parse_column_identifiers("1,3,5", t_med.header)
b,m,_=T(do_pci, repeat=100); print(f"parse_column_identifiers (header 10 cols): best={ms(b):.4f}ms/call")
def do_mdh():
    return make_default_headers(60)
b,m,_=T(do_mdh, repeat=100); print(f"make_default_headers(60): best={ms(b):.4f}ms/call")

print("\nDONE")
