"""Shared plumbing for the utility modules: gathering, projecting and reading
back the cell-index arrays the Mojo kernels write."""

from __future__ import annotations

import numpy as np

from ._lib import lib
from .csv import Table, from_rows

I = np.int64


def gather(table: Table, cells: np.ndarray, rows: np.ndarray) -> Table:
    """Turn a (cell index, row index) pair from a kernel back into a Table.

    `rows` is the kernel's row-start array with its trailing bound, so it holds
    `nrows + 1` entries, as `Table.row_cell` does.
    """
    nrows = int(rows.shape[0]) - 1
    cell_off = np.zeros(cells.shape[0], dtype=np.int64)
    cell_len = np.zeros(cells.shape[0], dtype=np.int32)
    row_cell = np.zeros(nrows + 1, dtype=np.int64)
    blobs: list[bytes] = []
    pos = 0
    i = 0
    for r in range(nrows):
        row_cell[r] = i
        for _ in range(int(rows[r + 1]) - int(rows[r])):
            c = int(cells[i])
            if c < 0:
                cell_off[i] = pos
                cell_len[i] = -1
            else:
                cell_off[i] = pos
                cell_len[i] = int(table.cell_len[c])
                blobs.append(table.data[
                    int(table.cell_off[c]):int(table.cell_off[c]) + int(table.cell_len[c])
                ].tobytes())
                pos += int(table.cell_len[c])
            i += 1
    row_cell[nrows] = i
    row_line = np.zeros(max(nrows, 1), dtype=np.int64)
    data = np.frombuffer(b"".join(blobs), dtype=np.uint8) if blobs \
        else np.zeros(0, dtype=np.uint8)
    return Table(data, cell_off, cell_len, row_cell, row_line, nrows)


def flat_gather(table: Table, cells: np.ndarray, width: int) -> Table:
    """As `gather`, for kernels that emit a fixed-width row."""
    nrows = int(cells.shape[0]) // max(width, 1)
    rows = np.arange(nrows + 1, dtype=np.int64) * width
    return gather(table, cells, rows)


def new_arrays(n: int) -> dict:
    return {
        "cells": np.zeros(max(n, 1), dtype=np.int64),
        "rows": np.zeros(max(n, 1) + 1, dtype=np.int64),
        "ncells": np.zeros(2, dtype=np.int64),
    }


def read_input(path: str, encoding: str = "utf-8-sig") -> bytes:
    with open(path, "rb") as f:
        raw = f.read()
    # Upstream opens in text mode with universal newlines; the same translation
    # happens here, before the Mojo parser sees the bytes.
    return raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def cells_of(table: Table, r: int) -> range:
    return range(int(table.row_cell[r]), int(table.row_cell[r + 1]))

def concat(tables):
    """Every table in one arena, plus the index where each one starts.

    A kernel that compares cells across tables needs one addressing space; this
    gives it one without copying anything twice.
    """
    rows = []
    starts = []
    for t in tables:
        starts.append(len(rows))
        rows.extend(t.row(r) for r in range(t.nrows))
    return from_rows(rows), starts
