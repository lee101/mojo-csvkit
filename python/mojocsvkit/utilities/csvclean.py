#!/usr/bin/env python

"""csvclean: report and fix common errors in a CSV file.

`RowChecker.checked_rows` is upstream's, transcribed. The per-cell tally behind
`--empty-columns` and `--remove-empty-columns` runs in the Mojo
`row_empty_counts` kernel, and the short-row merge behind `--join-short-rows`
runs in `join_rows`.
"""

import sys

import numpy as np

from .._lib import lib
from ..cleanup import Error, RowChecker, join_rows
from ..cli import CSVKitUtility
from ..csv import Table, from_rows


class CSVClean(CSVKitUtility):
    description = "Report and fix common errors in a CSV file."
    override_flags = ["L", "I"]

    def add_arguments(self):
        self.argparser.add_argument(
            "--length-mismatch", dest="length_mismatch", action="store_true",
            help="Report data rows that are shorter or longer than the header row.")
        self.argparser.add_argument(
            "--empty-columns", dest="empty_columns", action="store_true",
            help="Report empty columns as errors.")
        self.argparser.add_argument(
            "-a", "--enable-all-checks", dest="enable_all_checks",
            action="store_true", help="Enable all error reporting.")
        self.argparser.add_argument(
            "--omit-error-rows", dest="omit_error_rows", action="store_true",
            help="Omit data rows that contain errors, from standard output.")
        self.argparser.add_argument(
            "--label", dest="label",
            help='Add a "label" column to standard error.')
        self.argparser.add_argument(
            "--header-normalize-space", dest="header_normalize_space",
            action="store_true",
            help="Strip leading and trailing whitespace and replace sequences "
                 "of whitespace characters by a single space in the header.")
        self.argparser.add_argument(
            "--join-short-rows", dest="join_short_rows", action="store_true",
            help="Merges short rows into a single row.")
        self.argparser.add_argument(
            "--separator", dest="separator", default="\n",
            help="The string with which to join short rows. Defaults to a newline.")
        self.argparser.add_argument(
            "--fill-short-rows", dest="fill_short_rows", action="store_true",
            help="Fill short rows with the missing cells.")
        self.argparser.add_argument(
            "--fillvalue", dest="fillvalue",
            help="The value with which to fill short rows. Defaults to none.")
        self.argparser.add_argument(
            "--remove-empty-columns", dest="remove_empty_columns",
            action="store_true",
            help="Remove empty columns from standard output.")

    def main(self):
        if self.additional_input_expected():
            sys.stderr.write(
                "No input file or piped data provided. Waiting for standard input:\n")

        if (
            not self.args.length_mismatch
            and not self.args.empty_columns
            and not self.args.enable_all_checks
            and not self.args.header_normalize_space
            and not self.args.join_short_rows
            and not self.args.fill_short_rows
            and not self.args.remove_empty_columns
        ):
            self.argparser.error(
                "No checks or fixes were enabled. See available options with: "
                "csvclean --help")

        if self.args.join_short_rows and self.args.fill_short_rows:
            self.argparser.error(
                "The --join-short-rows and --fill-short-rows options are mutually "
                "exclusive.")

        default = self.args.enable_all_checks
        table = self.read_table()

        checker = RowChecker(
            table,
            length_mismatch=default or self.args.length_mismatch,
            empty_columns=default or self.args.empty_columns,
            header_normalize_space=self.args.header_normalize_space,
            join_short_rows=self.args.join_short_rows,
            separator=self.args.separator,
            fill_short_rows=self.args.fill_short_rows,
            fillvalue=self.args.fillvalue,
            remove_empty_columns=self.args.remove_empty_columns,
            zero_based=self.args.zero_based,
            omit_error_rows=self.args.omit_error_rows,
        )

        from .. import csv as mcsv

        rows = list(checker.checked_rows())
        self.output_file.write(
            mcsv.write(from_rows([checker.output_column_names] + rows),
                       lineterminator="\n").decode("utf-8", "replace"))

        if checker.errors:
            fieldnames = ["line_number", "msg"] + list(checker.column_names)
            if self.args.label:
                fieldnames.insert(0, "label")
            label = self.args.label
            error_rows = [fieldnames]
            for error in checker.errors:
                row = [error.line_number, error.msg] + list(error.row)
                if self.args.label:
                    row.insert(0, label)
                error_rows.append(row)
            self.error_file.write(
                mcsv.write(from_rows(error_rows),
                           lineterminator="\n").decode("utf-8", "replace"))
            sys.exit(1)


def launch_new_instance():
    utility = CSVClean()
    utility.run()


if __name__ == "__main__":
    launch_new_instance()