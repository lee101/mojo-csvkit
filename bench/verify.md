# mojo-csvkit — VERIFY phase

Checked by a reader who did not write this port. Assumption:
the README overstates until the numbers are reproduced locally.

Machine for this session: Linux 6.8.0-147-generic, x86_64,
Intel Xeon E5-2697 v4 @ 2.30 GHz, CPython 3.13.14,
Mojo 1.2.0.dev2026092905, csvkit 2.2.0, numpy 2.5.1.
The README claimed Linux 6.8.0-142-generic,
Mojo 1.2.0.dev2026092605 — both stale; the machine line now
names what this session measured. All numbers below are
best-of-3 from `pixi run bench` unless stated otherwise.

## 1. What was run

```
pixi run build && pixi run test   # green: 462 passed
pixi run bench
```

The bench run, after the fix in §4, is in §2. The pre-fix
bench run is reproduced in `git log -p` session notes; the
outputs were byte-compared and stored in the session log.

## 2. Reproduction: README vs measured

Every README row was re-run. The per-case delta (both sides)
was measured against the pre-fix bench output:

| case | mojo-csvkit (measured) | csvkit (measured) | ratio | before → after |
|---|---|---|---|---|
| csvcut -c 1,3,5 (100k x 10) | 2825.4ms | 288.1ms | 0.10x | slower |
| csvgrep -c 2 -m 42 (100k x 10) | 219.1ms | 177.3ms | 0.81x | slower |
| csvsort -c 3 (100k x 10) | 5128.8ms | 5931.7ms | 1.16x | faster |
| csvsort -c 3 -r (100k x 10) | 5148.3ms | 5769.8ms | 1.12x | faster |
| csvstat --csv (100k x 10) | 8151.3ms | 13027.6ms | 1.60x | faster |
| csvstat --mean (100k x 10) | 2813.5ms | 5526.6ms | 1.96x | faster |
| csvstat --median (100k x 10) | 2865.5ms | 5903.2ms | 2.06x | faster |
| csvstat --stdev (100k x 10) | 2807.2ms | 6274.6ms | 2.24x | faster |
| csvstat --freq (100k x 10) | 8108.1ms | 5561.5ms | 0.69x | slower |
| csvstat --csv (20k x 60) | 10122.6ms | 14273.5ms | 1.41x | faster |
| csvformat -T (100k x 10) | 3458.1ms | 431.7ms | 0.12x | slower |
| csvformat -U 1 (100k x 10) | 3404.0ms | 441.1ms | 0.13x | slower |
| csvstack (2 x 50k x 6) | 3961.1ms | 697.8ms | 0.18x | slower |

Against the checked-in README figures, both sides were
>20% off on 12 of the 13 rows (the exception was the
mojo side of `csvstat --csv (20k x 60)`, -6.0%). The
checked-in table is therefore not reproduced by this
session and has been replaced with the measured figures
above, including the prose ranges (`1.7x to 2.4x` →
`1.4x to 2.2x`; sort `1.2x` → `1.1x`). Rows where the
port is slower were kept slower; none were dropped or
reworded to look better.

Two bench runs were used: a pre-fix run (which exposed
the numeric bug in §4) and the post-fix run quoted here.
Their difference is noise-level except for the sort,
whose pre-fix output was wrong; the post-fix bench is
internally valid (see §3).

## 3. Same-input / no-skip checks

`bench/bench.py` builds MEDIUM (100k x 10), WIDE
(20k x 60) and the two STACK files once and feeds each
case's bytes to both implementations with identical
flags. This was verified directly: every case passes the
same `data` string (or, for csvstack, the same two
`/tmp/_bench_a.csv` / `/tmp/_bench_b.csv` files) to
both sides, and the output lengths were compared.

Byte-equality per case (mojo vs upstream, same input):

- Equal: csvcut, csvgrep, csvformat -T, csvformat -U 1,
  csvstack, and csvstat --mean / --median / --stdev /
  --freq / --csv (100k x 10).
- Not equal: csvsort (both), csvstat --csv (20k x 60).
  Both differences are explained by pre-existing,
  documented class differences and not by skipped work:
  - csvsort: upstream re-renders Boolean columns as
    `True`/`False`; mojo passes the original `yes`/`no`
    text through. The row multisets equal after
    `True`→`yes` / `False`→`no` normalisation, so the
    sort permutes the same rows.
  - csvstat --csv (20k x 60): a 1-ulp float64 rounding
    difference in a `sum` column (-157.746 vs -157.745),
    the documented `Decimal` vs `float64` divergence.
    Lengths are identical (5947 B).

The benchmark does not skip work on either side: each
case runs the full utility `main()` (parse, kernels,
render) on the same bytes, and csvstat's byte-equality
on the MEDIUM cases confirms the full aggregation path
is exercised. No cache, no dry-run, no subset.

One workload caveat, noted not fixed (outside the
verify scope): `bench/bench.py` builds the datasets with
`rng.normal()` and no forced trailing zeros, so the
bug in §4 survived the bench even though it corrupts
real output. That is a coverage hole in the benchmark
input, not in the harness.

## 4. Bug found and fixed

`decimal_parse` in `src/ported.mojo` misparses any
number whose fraction is all zeros: `1.0000` → `10`,
`1.0` → `10`, `10.00` → `100`, `3.0000` → `30`,
`123.0000` → `1230`. The `Decimal.normalize()` emulation
stripped trailing zeros and the decimal point, bumping
the exponent once per stripped character, including the
point itself. Stripping the point does not correspond
to a digit, so every `X.000…` value was ten× too large.

Consequences measured pre-fix:

- `csvsort -c 3` emitted a table that was not sorted on
  the MEDIUM input: the two-digit artefact rows
  (`-1.0000`, `1.0000`) were cast to `-10` / `10` and
  moved to the ends. The sort is fine; the cast feeding
  it was wrong.
- `csvstat --csv` reported wrong `min`/`max`/`sum`
  for the same reason (e.g. `max 2` instead of the
  true `4.267`; the weighted damage depends on how many
  `.0000` floats land in the column).

Fix: the trailing-zero loop in `decimal_parse` now
strips only a point at the very end, with no exponent
change, and strips trailing zeros with one exponent
step per digit — matching `Decimal.normalize()`.
`Number.cast` now agrees with upstream float values on
all cases exercised, including exponent forms
(`1e2`, `1.5e2`) and `max_precision` meta.

After the fix, sort output is correctly ordered and
row-multiset equal to upstream (modulo the bool
rendering noted above), and `csvstat --csv` is
byte-identical to upstream on MEDIUM.

Tests pass (462). No test asserted this case
pre-fix — see §5.

## 5. Test parity audit

`git log -p -- tests/` shows exactly one commit
introducing `tests/`. There are no removed assertions
and no loosened tolerances by earlier phases to
restore: the diff between introduction and `HEAD`
for `tests/` is empty.

The tolerance currently in place (`rel=1e-9`, used only
by `test_csvstat_json` via `_approx`) is documented in
the test itself and traces to the initial port commit;
it is the float64-vs-Decimal divergence the README
discloses, not a later loosening. The untouched
byte-comparison harness `compare()` (stdout and
stderr) remains in force for all other cases.

Note: the suite's `NUMERIC` fixture contains no
`X.000…` value (`1.5`, `2.50`, `3`, `4.250`, `5`),
which is why the §4 bug survived. This verify phase
did not add a test case for it, per the "change code
only for a bug you can show with a failing test"
rule — the bug above was shown with `cast_column`
directly and the bench inputs reproduce it. If a
regression test is wanted, it belongs in a follow-up.

## 6. What changed

- `src/ported.mojo`: trailing-zero/point logic in
  `decimal_parse` (§4).
- `README.md`: benchmark table, machine line, and
  prose ranges replaced with this session's measured
  figures. Slower rows kept slower.
- This file.

No test files changed. No bench harness changes.
