# mojo-csvkit — ANALYZE phase

Measured map of where time goes. No algorithms changed in this phase; `src/`
and `tests/` untouched. Every number below comes from a run in this session
(2026-11-09, Linux 6.8.0-147-generic, x86_64 Intel Xeon E5-2697 v4 @ 2.30 GHz,
72 vCPU, CPython 3.13.14, numpy 2.5.1, Mojo 1.2.0.dev2026092905, upstream
csvkit importable). Reproduce with `pixi run bench` (flock-wrapped, ran alone)
and `pixi run python bench/profile.py full`.

`bench/profile.py` is an auxiliary profiler script; it is not part of the
deliverable. The only deliverable file is this document. All kernel timings
are best-of-3 (matching `bench/bench.py`); `cProfile` numbers are single-run
and quoted only for attribution, never as budget inputs.

## 1. Datasets (identical generators to `bench/bench.py`)

| dataset | bytes | rows | cols | cells | arena (`data`+offsets) |
|---|---|---|---|---|---|
| SMALL | 128,943 | 2,001 | 10 | 20,010 | 0.19 MB |
| MEDIUM | 6,733,304 | 100,001 | 10 | 1,000,010 | 19.3 MB (`data` 5,733,294 B) |
| WIDE | 8,173,654 | 20,001 | 60 | 1,200,060 | 22.9 MB (`data` 6,973,594 B) |
| STACK_A/B | 2,266,935 ea. | 50,001 ea. | 6 | 300,006 ea. | 13.5 MB combined |

## 2. End-to-end utility totals (`pixi run bench`, best-of-3, this session)

| case | mojo-csvkit | upstream csvkit | ratio (up/mojo) | verdict |
|---|---|---|---|---|
| csvcut -c 1,3,5 (100k x 10) | 2,794.1 ms | 278.6 ms | 0.10x | **SLOWER** |
| csvgrep -c 2 -m 42 (100k x 10) | 217.3 ms | 178.9 ms | 0.82x | **SLOWER** |
| csvsort -c 3 (100k x 10) | 5,262.7 ms | 6,117.3 ms | 1.16x | faster |
| csvsort -c 3 -r (100k x 10) | 5,310.7 ms | 5,916.3 ms | 1.11x | faster |
| csvstat --csv (100k x 10) | 8,356.4 ms | 13,477.7 ms | 1.61x | faster |
| csvstat --mean (100k x 10) | 2,832.3 ms | 5,408.0 ms | 1.91x | faster |
| csvstat --median (100k x 10) | 2,774.5 ms | 5,719.4 ms | 2.06x | faster |
| csvstat --stdev (100k x 10) | 2,745.6 ms | 6,124.9 ms | 2.23x | faster |
| csvstat --freq (100k x 10) | 7,987.9 ms | 5,635.7 ms | 0.71x | **SLOWER** |
| csvstat --csv (20k x 60) | 10,315.2 ms | 14,237.9 ms | 1.38x | faster |
| csvformat -T (100k x 10) | 3,422.9 ms | 427.7 ms | 0.12x | **SLOWER** |
| csvformat -U 1 (100k x 10) | 3,341.8 ms | 417.8 ms | 0.13x | **SLOWER** |
| csvstack (2 x 50k x 6) | 3,765.7 ms | 685.6 ms | 0.18x | **SLOWER** |

Rows marked **SLOWER** are retained, not dropped. Pattern: every win is a
numeric pass (inference, cast, aggregation, sort); every loss is a path that
re-materialises Python `str` per cell or re-enters ctypes per row.

## 3. Ranked hotspot table

Ranked by absolute ms on MEDIUM (100k x 10) unless noted. Columns as
requested. `flops` is a measured-cost proxy (op count from the kernel's loop
structure, not a hardware counter); `flops/byte` is flops ÷ bytes touched.
"Glue vs kernel" is where the measured wall time actually sits.

| # | function (entry point) | input size | time | bytes touched | flops (~) | flops/byte | bound | glue vs kernel | next phase |
|---|---|---|---|---|---|---|---|---|---|
| H1 | `csvstat` `--csv`/`--freq` host distinct+freq (`_Stats.distinct`, `_rendered_values`, `_counter`, `_boolean_keys`) | 100k x 10, 1.0M cells | 4,500 ms (distinct x10); 2,261 ms (get_freq x10, second render) | 5.7 MB text + per-cell `Decimal` objects | ~2M hash/compare ops, O(n) Python | ~0.3 (object-dominated) | Python-object bound (neither classic) | **Python glue, 100%**; Mojo `values_most_common` kernel present but uncalled | **none** (host rewrite is not a Mojo kernel phase; do not SIMD/parallel/GPU it) |
| H2 | `order_by` full (`infer` + `cast` + gap-sort kernel + `gather` + `write`) | 100k rows, 1 key col | 3,407 ms (isolated) / 5,263 ms (utility) | 1.6 MB sort keys + 5.7 MB table | ~9.6M (`n log n`) | 6.0 | compute (comparison) in kernel; glue elsewhere | mixed — see split below | **none** for kernel (already wins vs upstream); glue is H5/H6 |
| H3 | `csv_writerow` via `write()` (1 ctypes call **per row**) | 100k rows, 5.73 MB | 1,629 ms (whole-table write); 11.5 µs/call single row → ~1,150 ms of the 1,629 ms is call+kernel per row | 5.73 MB in, ~6.7 MB out; 2 passes over every byte (measure then copy in `join_append_data`) | ~23M (±3 comparisons/byte × 2 passes) | 4.0 | **memory** on paper; **latency/call-overhead** in practice (16–20 µs/row flat from 2k to 100k rows) | **ctypes glue + kernel**: ~70% per-row call path, ~30% Python loop/`tobytes` | **unroll** (batch rows per call; single pass). GPU: **no** — 4.0 flops/byte is above the ~2 threshold but the bottleneck is per-call serialisation, not bandwidth; a bigger batch fixes it on CPU. |
| H4 | `type_tester_run` / `infer_types` (10 cols) | 1.0M cells, 5.73 MB | 1,273 ms (1.25 µs/cell, flat 2k→100k→WIDE) | 5.73 MB | ~46M (Boolean+Number+Text tries per live hypothesis) | 8.0 | compute (but over-tested: order `[0,1,1,5]` tests Number twice) | **kernel** (single ctypes call per table, 1.27 s inside the kernel) | **unroll** (deduplicate hypotheses; early-exit already exists — win is smaller). **parallel**: **no** — rows are serial (hypothesis intersection), but columns are independent (10-way). GPU: **no** despite 8.0 flops/byte — the operator is argmin-over-prefix with data-dependent early death; launch/transfer cost exceeds the 1.3 s CPU cost only if parallelised over columns on CPU first. |
| H5 | `gather` / `flat_gather` (Python per-cell re-pack) | 300k cells (cut) / 1.0M cells (sort) | 559 ms (cut, 300k) / 1,814 ms (sort, 1.0M) | 4.3 MB (cut) | 0 (copies) | 0.0 | memory | **Python glue, 100%** (`tobytes()` per cell + `b"".join`) | **none** (this is the glue bug; the fix is a Mojo-side gather into a preallocated arena — still "none" of unroll/simd/parallel/gpu for the kernel itself) |
| H6 | `Table.row()` decode + `from_rows()` re-encode roundtrip | 1.0M cells | 933 ms (decode) + 1,913 ms (`from_rows` full table) | 5.73 MB × 3 passes (decode, encode, join) | 0 (copies/transcodes) | 0.0 | memory | **Python glue, 100%** (`str` per cell, `encode` per cell) | **none** (same as H5 — B-table columnar path removes this entirely) |
| H7 | `column_cast_number` x10 cols | 1.0M cells, 5.73 MB | 1,105 ms (121 ms/col, 1.20 µs/row/col) | 5.73 MB in + 2.6 MB/col out + **5.73 MB `buf` zeroed per column** | ~34M (strip, null-test, currency peel, `decimal_parse`) | 6.0 | compute (string parse, branchy) | **kernel** (1 call/col); per-col alloc in glue | **unroll** (reuse one `buf`; fold into `infer_types` where the same parse already runs). SIMD: **no** (variable-length parse). GPU: **no** — 6.0 flops/byte but a serial digit-by-digit parse; CPU parallel over columns is the right axis. |
| H8 | `csv_reader` (whole-buffer) | 6.73 MB, 100k rows | 109 ms full / 91 ms raw (12.8 ns/byte flat) | 6.73 MB in; ~195 MB of `np.zeros` arena allocated+zeroed per parse | ~20M (1 state-machine step/byte) | 3.0 | **latency-bound serial chain** (not bandwidth: 0.06–0.08 GB/s vs 17 GB/s numpy sum) | **kernel** incl. allocs; raw-vs-full delta 18 ms = newline+alloc glue | **none**. SIMD/parallel/GPU all **no**: the 9-state machine is inherently sequential (quoted-field state), 3.0 flops/byte buys nothing on a serial dependence chain. numpy `genfromtxt` (430 ms) and `loadtxt` (70 ms, numeric-only) are not drop-in baselines (no dialect, no ragged rows). Python `csv.reader` is 181 ms — the Mojo reader already beats it. |
| H9 | `_boolean_keys` (host Boolean cast, run for every Boolean column's first `column()` call) | 200k cells (2 cols × 100k) | 1,048 ms cumulative in `--mean` profile (0.41 s tottime) | 200k `str` objects | ~2M (`replace`+`strip`+`lower`+set test per cell) | ~1 | Python-object bound | **Python glue, 100%** (a Mojo `boolean_cast` kernel exists but is never called on this path) | **none** (call the existing kernel; not a new SIMD/GPU phase) |
| H10 | `agg_median` / `percentiles_run` (copy + introsort) | 100k rows, 1 col | 9.4 ms | 1.6 MB (keys+rows) | ~6.4M (`n log n`) | 4.0 | compute | **kernel** | **parallel**: **no** (sort is the op; `np.median` is 1.5 ms — a radix/select would win, but that's an algorithm swap, out of scope). GPU: **no** — 4.0 flops/byte but 9 ms total; transfer alone exceeds it. Verdict: **none** (absolute cost is noise). |
| H11 | Reductions `agg_sum/mean/variance/stdev/min/max/count/has_nulls/max_length/max_precision` | 100k rows, 1 col | 0.06–1.38 ms each (median 0.29 ms) | 0.1–2.5 MB | ~0.6M each | 0.16–2.0 | memory | **kernel** | **none**. All lose pairwise to numpy (`np.sum` 0.047 ms vs Mojo `agg_sum` 0.29 ms = 6×; `np.min` 0.034 ms vs 0.15 ms) but win vs upstream Decimal by 10–50× — they are not the problem. GPU: **no** — every one is ≤2.0 flops/byte and ≤1.4 ms; the whole 13-op × 10-col `csvstat --csv` aggregation budget is <100 ms. |
| H12 | `test_row` (`csvgrep -m`) | 100k rows, 1 col | 1.32 ms | 0.57 MB | ~1.1M (naive substring scan) | 2.0 | memory | **kernel** (1 call/table) | **parallel**: candidate **yes** (per-row embarrassingly parallel, 1.3 ms → sub-ms); SIMD: **no** (short needles, 2.0 flops/byte). GPU: **no** — 2.0 flops/byte is the boundary and 1.3 ms never amortises a copy. Utility is still **SLOWER** (217 vs 179 ms) because reader (109 ms) + kept-row re-encode dominate, not this kernel. |
| H13 | `csvcut` kernel (gather indices) | 300k out-cells | 0.49 ms | 2.4 MB | ~0.6M | 0.25 | memory | **kernel** | **none** (kernel is 0.02% of the 2,794 ms utility; `_render`+`gather`+`write` = H3+H5+H6). |
| H14 | `csvstack_project` kernel | 300k cells | 0.7 ms | 2.4 MB | ~0.6M | 0.25 | memory | **kernel** | **none** (utility loss is `concat` 1,092 ms + `project` decode 354 ms + `from_rows+write` 1,991 ms = H5+H6+H3). |
| H15 | `csvstack_headers` kernel + `concat()` | 2 × 50k x 6 | kernel ~0 ms; `concat` 1,092 ms | 3.9 MB | ~0 | 0.0 | memory | **Python glue** (`concat` decodes every row of both tables to re-encode them) | **none** (pointer-offset, not a kernel). |
| H16 | `order_by` kernel alone (gap sort, numeric, pre-cast) | 100k rows | 64 ms (10k: 3.3 ms; 2k: 0.4 ms; superlinear, shell-gap) | 1.6 MB | ~9.6M | 6.0 | compute | **kernel** | **parallel**: **no** (comparison sort; `np.argsort` is 11 ms and Python `sorted` 76 ms — the gap sort is 6× numpy but still beats upstream's 6,117 ms utility). Algorithm swap (merge/radix) is out of ANALYZE scope; GPU **no** (6.0 flops/byte but 64 ms and gather-bound). |
| H17 | `row_empty_counts` (csvclean) | 1.0M cells | 0.81 ms | 4.0 MB | ~1.0M | 0.25 | memory | **kernel** | **none**. (`csvclean checked_rows` itself decodes all rows in Python: 1,062 ms — glue, H6-class.) |
| H18 | `parse_column_identifiers` / `make_default_headers` | header only (10–60 cols) | 0.036 ms / 0.022 ms per call | <1 KB | <1k | ~1 | — | **kernel** (1 call) | **none** (noise floor). |
| H19 | `_value_at` + `Decimal(text)` per cell in `_rendered_values` | 1.2M calls in `--csv` | ~3.3 s cumulative (cProfile) inside H1 | 1.2M `str` + `Decimal` constructions | — | — | Python-object | **Python glue** | **none** (dies with H1). `_base_of` is O(n) per call but unused on this path — flagged by reading, not by measurement; do not chase it. |

### Hotspot order-by split (one row per requested column, ranked by ms)

The table above is the ranked list; this is the same data ordered for
actionability (largest first):

1. H1 host freq/distinct — 4,500 ms — Python glue — **none**
2. H2 order_by full — 3,407 ms — mixed (gather 1,814 + kernel 64 + infer 1,230 + cast 118) — **none**
3. H3 writerow per-row ctypes — 1,629 ms — glue+kernel — **unroll** (batch)
4. H4 type_tester_run — 1,273 ms — kernel — **unroll** (dedupe)
5. H6 decode/re-encode — 933 + 1,913 ms — Python glue — **none**
6. H7 cast x10 — 1,105 ms — kernel — **unroll** (buffer reuse)
7. H5 gather — 559 ms (cut) — Python glue — **none**
8. H8 csv_reader — 109 ms — kernel — **none** (already beats Python `csv`)
9. H9 _boolean_keys — ~1,000 ms — Python glue — **none**
10. H10–H19 — all <10 ms each — **none** (H12 parallel candidate only)

## 4. Baselines (same bytes, this session)

**Upstream csvkit** (importable): see §2. Wins 1.11–2.23× on numeric paths;
loses 0.10–0.82× on every materialising path.

**Plain Python `csv`** (same MEDIUM bytes):

| baseline | time | notes |
|---|---|---|
| `csv.reader` → `list()` | 180.9 ms | Mojo `csv_reader` raw 91 ms **beats it** |
| `csv.reader` + `csv.writer` roundtrip | 369.1 ms | Mojo `write` alone (1,629 ms) is 4.4× slower — entirely the per-row ctypes re-entry + slice/`tobytes` per row |
| `sum(list(floats))` 100k | 3.93 ms | Mojo `agg_sum` 0.29 ms **beats it** 14× |
| `sorted(idx, key)` 100k | 76.0 ms | Mojo `order_by` kernel 64 ms **beats it** |
| `set()` 100k str | 4.96 ms | host `_counter` path is ~40× this per column — the O(n²) `ranked.index()` + double render is the cost, not hashing |

**numpy** (same column/table):

| baseline | time | Mojo kernel | Mojo time | ratio |
|---|---|---|---|---|
| `np.sum` 100k f64 | 0.047 ms | `agg_sum` | 0.29 ms | 6× slower |
| `np.mean` | 0.053 ms | `agg_mean` | 0.37 ms | 7× slower |
| `np.std(ddof=1)` | 0.236 ms | `agg_stdev` | 0.64 ms | 3× slower |
| `np.median` | 1.46 ms | `agg_median` | 9.4 ms | 6× slower |
| `np.min` / `np.max` | 0.034 / 0.032 ms | `agg_min`/`agg_max` | 0.15 / 0.14 ms | 4× slower |
| `np.argsort` 100k f64 | 11.1 ms | `order_by` kernel | 64 ms | 6× slower |
| `np.char.find` 100k str | 3.09 ms | `test_row` | 1.32 ms | **2.3× faster** |
| `np.genfromtxt` (4 numeric cols) | 429.5 ms | — | — | not comparable (no dialect/ragged) |
| `np.loadtxt` (2 cols) | 70.5 ms | — | — | not comparable (numeric-only, no header/ragged) |
| `np.unique` 100k str | 23.6 ms | host distinct | ~450 ms/col | 19× slower |

Takeaway: no Mojo reduction beats numpy pairwise, but every one beats
upstream agate's `Decimal` + Python-object path by an order of magnitude —
which is why the utilities that stay in the arena win and the ones that
leave it (decode → `str` → `from_rows` → `write`) lose.

## 5. Python-level loops re-entering ctypes, allocations, copies

### 5a. Per-element / per-row ctypes re-entry

| site | calls per MEDIUM run | evidence |
|---|---|---|
| `csv.py:write` → `csv_writerow` | **100,001** (1/row) | 11.5 µs/call; `write` SMALL 36.2 ms → MEDIUM 1,685 ms → WIDE 408 ms (20.4 µs/row at 349 B/row) — flat per-row cost independent of row width proves call overhead, not byte work |
| `csv.py:write` slice `.ctypes.data` | 200k slices + 100k `tobytes()` per write | 1.47 µs/slice, 0.46 µs/`tobytes` each — pure glue inside the 1,629 ms |
| `csvclean` `RowChecker.checked_rows` | 100k `table.row()` calls | 1,062 ms for decode-only pass |
| `csvstat` `_rendered_values` → `_value_at` → `cell()` | 1.2M `cell()` calls in `--csv` | 3,410 ms in `cell()` alone (cProfile) |
| Everything else is 1 call/table, 1 call/col, or 1 call/document | — | `csv_reader`, `type_tester_run`, `csvcut`, `test_row`, `order_by`, `csvstack_*`, `join_rows` |

Single-call ctypes overhead is small in absolute terms (`agg_count` 100k
elems in 46.7 µs = 0.5 ns/elem) — the problem is never one call, it is
100k calls through a Python loop with two numpy slice views and a
`tobytes()` per call.

### 5b. Per-call allocations and copies

| site | allocation per call | total per MEDIUM run |
|---|---|---|
| `csv.reader` | 5 arrays × `cap=n+2`: `data` 1 B + `cell_off` 8 B + `cell_len` 4 B + `row_cell`/`row_line` 8 B = **~195 MB** `np.zeros` | 195 MB zeroed per parse (0.37 ms for one 6.7M `uint8` zero — allocation is not the reader's cost; the serial kernel is) |
| `cast_column` | `buf` = **full table `data` size** (5.73 MB) **per column** | 57 MB for `--csv` (10 cols) + `Column` 2.6 MB/col |
| `infer_types` | `buf` 5.73 MB + scratch | 5.73 MB per `_Stats` construction |
| `csv.write` | `rec` buffer `data + 8·ncells + 4·nrows` = **14.1 MB** | 14.1 MB zeroed per write |
| `csvcut`/`csvstack_project`/`order_by` out arrays | `cap = nrows·ncols` int64 | 2.4 MB (cut), 2.4 MB (stack), 0.8 MB (sort) |
| `gather` / `from_rows` | per-cell `tobytes()` slice + `b"".join` | 300k–1.2M object copies per utility — the dominant copy in cut/sort/stack/format |
| `agg_median` | `keys`+`rows` 1.6 MB | per call (9.4 ms — still noise) |
| `_Stats.column()` Boolean path | `_boolean_keys` allocates + 100k `str` ops | ~200 MB of transient strings in `--mean` |

### 5c. Copy-multiplier summary (bytes moved per input byte)

| utility | input bytes | bulk copies | effective multiplier |
|---|---|---|---|
| reader alone | 6.73 MB | parse into arena (1×) | 1× |
| csvformat | 6.73 MB | decode (1×) + `from_rows` encode (1×) + `write` 2-pass (2×) | **~4× + 1M `str` objects** |
| csvcut | 6.73 MB | `gather` (1×) + decode (1×) + `from_rows` (1×) + `write` (2×) | **~5× + 300k–1M objects** |
| csvsort | 6.73 MB | `gather` permutation (1×) + `write` (2×) | ~3× + infer + cast |
| csvstack | 4.53 MB | `concat` decode+encode (2×) + `project` decode (1×) + `from_rows` (1×) + `write` (2×) | **~6× + 1.2M objects** |
| csvstat `--csv` | 6.73 MB | infer cast (1×) + `cast_column` (1×) + `_rendered_values` ×2 (2×) | ~4× + 2M `Decimal`/`str` |

## 6. Memory-bound vs compute-bound (roofline, measured)

Practical single-core roof on this host from `np.sum` (17.0 GB/s, 0.125
flops/byte) and `np.min` (23.6 GB/s); DDR4 socket bandwidth is far higher
with all 72 threads, but no kernel here is threaded, so the single-core
roof is the honest comparator.

| kernel | flops/byte | eff. bandwidth | verdict |
|---|---|---|---|
| `csv_reader` | 3.0 | 0.07 GB/s | **serial latency chain** — 250× below the roof despite 3.0 fpb. Not memory-bound, not compute-bound: dependency-bound. |
| `type_tester_run` | 8.0 | ~0.005 GB/s | compute-bound (string casts), but branchy/over-tested, not FLOPs. |
| `column_cast_number` | 6.0 | ~0.005 GB/s | compute-bound (parse). |
| `csv_writerow` | 4.0 | ~0.007 GB/s | call-overhead-bound; the 2-pass `join_append_data` doubles byte traffic for nothing. |
| `order_by` / `agg_median` | 6.0 / 4.0 | ~0.2 GB/s | compute-bound (comparisons) — the only truly compute-bound kernels, and both are <100 ms. |
| `agg_sum/min/max/count/row_empty_counts/csvcut/csvstack_project` | 0.16–0.25 | 1.4–12 GB/s | memory-bound and near-roof — healthy, no action. |
| `test_row` | 2.0 | 0.43 GB/s | memory-bound (short-needle scan), healthy. |
| numpy reductions | 0.125 | 17–25 GB/s | the roof — Mojo reductions sit at 1.4–6 GB/s (4× gap, all <1.4 ms absolute). |

## 7. GPU verdicts (each with the flops-per-byte figure behind it)

Rule applied: GPU only if flops/byte ≳ 2 **and** absolute time amortises
copy-in/copy-out **and** the operator is data-parallel. All three must hold.

| kernel | flops/byte | time | verdict |
|---|---|---|---|
| `csv_reader` | 3.0 | 109 ms | **no** — 3.0 fpb but a 9-state serial byte chain; cannot parallelise without an algorithm change. |
| `type_tester_run` | 8.0 | 1,273 ms | **no** — 8.0 fpb but prefix-intersection with early death; columns give 10-way CPU parallelism, GPU transfer of 19 MB arena per launch still loses to dedupe+parallel on CPU. |
| `column_cast_number` | 6.0 | 121 ms/col | **no** — 6.0 fpb but serial digit parse; per-column CPU parallel is the axis. |
| `csv_writerow` | 4.0 | 1,629 ms | **no** — 4.0 fpb (above threshold) but the bottleneck is 100k serial calls, not bandwidth; batching on CPU removes ~1.1 s with zero transfer. GPU is the wrong fix for a call-count bug. |
| `order_by` kernel | 6.0 | 64 ms | **no** — 6.0 fpb but comparison sort over 100k keys; `np.argsort` already does this in 11 ms on CPU. |
| `agg_median` | 4.0 | 9.4 ms | **no** — 4.0 fpb, 9.4 ms; one copy alone exceeds the kernel. |
| `test_row` | 2.0 | 1.3 ms | **no** — 2.0 fpb is the boundary and 1.3 ms never amortises. |
| All reductions | 0.16–2.0 | ≤1.4 ms | **no** — every one is below 2.0 fpb. |

**Overall GPU verdict: no kernel qualifies.** The two highest-fpb kernels
(inference 8.0, cast 6.0) are serial string parses; the highest-absolute
costs (H1–H9) are Python glue with ~0 fpb, which no device accelerates.
A GPU phase would spend its entire budget on copy-in/copy-out for kernels
that are already ≤10 ms or irreducibly serial.

## 8. What the next phases should do (in order)

1. **Stop materialising `str` per cell on the write path** (H3+H5+H6+H15,
   ~5–6 s across cut/format/stack/sort utilities). Write directly from the
   arena; batch `csv_writerow` over row ranges. This is the entire
   csvcut/csvformat/csvstack loss (0.10–0.18×) and ~60% of csvsort's
   remaining time.
2. **Fix the csvstat freq/unique host path** (H1+H9, ~4.5 s). Either call
   the existing `column_values_distinct`/`values_most_common`/`boolean_cast`
   kernels or deduplicate the double `_rendered_values` pass. This is the
   entire `--freq` loss (0.71×) and ~45% of `--csv`.
3. **Dedupe type-tester hypotheses + reuse cast buffers** (H4+H7, ~2.4 s
   across csvstat/sort). The order list `[0,1,1,5]` tests Number twice;
   `cast_column` zeroes a full-table `buf` per column.
4. **Batch the writer** (H3, **unroll**). Single-pass `join_append_data`
   (drop the measure pass) + one ctypes call per block of rows.
5. **Nothing else.** Reader, reductions, `test_row`, `csvcut`/`csvstack`
   kernels, `agg_median`, `row_empty_counts`, header helpers are all
   ≤110 ms each and already beat or match their Python/upstream baselines.
   Do not SIMD, parallelise, or GPU them in the next phase; the evidence
   above (fpb ≤2 or absolute <10 ms or serial) does not support it.
   `test_row` (H12) is the single **parallel** candidate if a parallel
   phase must pick something: 1.3 ms, per-row independent, 2.0 fpb.

## 9. Caveats

- FLOP counts are loop-structure estimates, not PMU counters; they are used
  only for flops/byte ratios and GPU thresholds, where ±2× changes no
  verdict except `test_row` (flagged as the boundary case).
- `np.loadtxt`/`genfromtxt` baselines are numeric-only and lack csvkit's
  dialect, quoting, ragged-row and header semantics — quoted for scale, not
  as drop-in replacements.
- cProfile inflates absolute ms (~10–30%); it was used only to rank
  attribution inside a utility, never as a budget number.
- No code was changed in this phase. `git status` should show only the new
  `bench/analysis.md` (and the auxiliary `bench/profile.py` profiler, which
  may be deleted — it is not part of the deliverable).
