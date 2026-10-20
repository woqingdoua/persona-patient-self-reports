#!/usr/bin/env python3
"""Prepare an explicit candidate-report column for the retained graders.

This is a code-release utility added after the historical experiments. It does
not reconstruct or certify which report column was evaluated in those runs.
The retained graders read ``report``; generation scripts instead produce named
columns such as ``report_gpt4o_mini`` and ``report_gpt5_mini``. This utility makes
that mapping explicit and preserves an existing ``report`` as ``source_report``.

Only CSV reshaping is performed: no model calls, scoring, sampling, or result
generation. Existing files are never overwritten. Run only on data that you
are authorized to process; the resulting CSV remains subject to its original
data-access and redistribution restrictions.
"""

import argparse
import csv
from pathlib import Path


def prepare_evaluation_input(input_path, output_path, report_column):
    """Write a distinct grader-input CSV and return the number of rows.

    Every row and every existing column value is retained, except that the
    explicit candidate column replaces ``report``. Any previous ``report``
    value is moved into a new ``source_report`` column. If that would overwrite
    an existing ``source_report``, validation fails instead.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output must be different files.")
    if output_path.exists() or output_path.is_symlink():
        raise ValueError("Output already exists; this utility never overwrites files.")
    if not report_column or not report_column.strip():
        raise ValueError("--report-column must name an explicit existing column.")

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("Input has no CSV header.")
        if any(not name for name in fieldnames):
            raise ValueError("Input contains an empty column name.")
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError("Input contains duplicate column names.")
        if report_column not in fieldnames:
            raise ValueError(f"Requested report column is missing: {report_column}")

        preserve_original = "report" in fieldnames
        if preserve_original and "source_report" in fieldnames:
            raise ValueError(
                "Cannot preserve the existing report: source_report already exists."
            )
        output_fields = list(fieldnames)
        if preserve_original:
            output_fields.insert(output_fields.index("report"), "source_report")
        else:
            output_fields.append("report")

        rows = []
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"CSV row {row_number} has the wrong number of fields.")
            candidate = row[report_column]
            if not candidate.strip():
                raise ValueError(f"CSV row {row_number} has an empty candidate report.")
            if preserve_original:
                row["source_report"] = row["report"]
            row["report"] = candidate
            rows.append(row)

    # Validate every row before creating an output. Exclusive creation also
    # prevents races from overwriting a file created after the existence check.
    with output_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Existing input CSV.")
    parser.add_argument("--output", required=True, type=Path, help="New output CSV.")
    parser.add_argument(
        "--report-column",
        required=True,
        help="Column containing the exact candidate text to pass to the grader.",
    )
    args = parser.parse_args(argv)
    try:
        row_count = prepare_evaluation_input(
            args.input, args.output, args.report_column
        )
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))
    print(
        f"Prepared {row_count} rows. Candidate column: {args.report_column}. "
        "No model evaluation was performed."
    )


if __name__ == "__main__":
    main()
