"""Offline tests using invented nonclinical CSV fixtures only."""

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


UTILITY_PATH = Path(__file__).resolve().parents[1] / "tools" / "prepare_evaluation_input.py"
SPEC = importlib.util.spec_from_file_location("prepare_evaluation_input", UTILITY_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareEvaluationInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.csv"
        self.output = self.root / "prepared.csv"

    def write_fixture(self, fields, rows):
        with self.source.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(rows)

    def read_output(self):
        with self.output.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return reader.fieldnames, list(reader)

    def prepare(self, column="generated"):
        return MODULE.prepare_evaluation_input(self.source, self.output, column)

    def test_preserves_original_and_other_columns_and_order(self):
        self.write_fixture(
            ["fixture_id", "report", "generated", "extra"],
            [
                ["fixture-b", "Original B", "Candidate B", "001"],
                ["fixture-a", "Original A", "Candidate A", "002"],
            ],
        )
        original_bytes = self.source.read_bytes()
        self.assertEqual(self.prepare(), 2)
        fields, rows = self.read_output()
        self.assertEqual(
            fields, ["fixture_id", "source_report", "report", "generated", "extra"]
        )
        self.assertEqual([row["fixture_id"] for row in rows], ["fixture-b", "fixture-a"])
        self.assertEqual([row["report"] for row in rows], ["Candidate B", "Candidate A"])
        self.assertEqual(
            [row["source_report"] for row in rows], ["Original B", "Original A"]
        )
        self.assertEqual([row["extra"] for row in rows], ["001", "002"])
        self.assertEqual(rows[0]["generated"], "Candidate B")
        self.assertEqual(self.source.read_bytes(), original_bytes)

    def test_without_original_report_adds_only_report(self):
        self.write_fixture(
            ["fixture_id", "generated", "source_report"],
            [["fixture-a", "Candidate A", "Already retained source"]],
        )
        self.prepare()
        fields, rows = self.read_output()
        self.assertEqual(fields, ["fixture_id", "generated", "source_report", "report"])
        self.assertEqual(rows[0]["source_report"], "Already retained source")
        self.assertEqual(rows[0]["report"], "Candidate A")

    def test_unicode_multiline_and_whitespace_are_preserved(self):
        candidate = '  Invented “quotation”,\nsecond line — α.  '
        self.write_fixture(["report", "generated"], [["original", candidate]])
        self.prepare()
        _, rows = self.read_output()
        self.assertEqual(rows[0]["report"], candidate)
        self.assertEqual(rows[0]["generated"], candidate)

    def test_rejects_source_report_collision(self):
        self.write_fixture(
            ["report", "source_report", "generated"], [["old", "retained", "new"]]
        )
        with self.assertRaisesRegex(ValueError, "source_report already exists"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_rejects_missing_candidate_column(self):
        self.write_fixture(["report"], [["original"]])
        with self.assertRaisesRegex(ValueError, "column is missing"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_rejects_empty_candidates_before_writing_any_rows(self):
        for candidate in ("", " \t\n"):
            with self.subTest(candidate=repr(candidate)):
                self.write_fixture(["generated"], [["valid"], [candidate]])
                with self.assertRaisesRegex(ValueError, "empty candidate"):
                    self.prepare()
                self.assertFalse(self.output.exists())

    def test_rejects_existing_output_without_touching_it(self):
        self.write_fixture(["generated"], [["candidate"]])
        self.output.write_text("keep me\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "never overwrites"):
            self.prepare()
        self.assertEqual(self.output.read_text(encoding="utf-8"), "keep me\n")

    def test_rejects_input_output_identity(self):
        self.write_fixture(["generated"], [["candidate"]])
        original_bytes = self.source.read_bytes()
        with self.assertRaisesRegex(ValueError, "different files"):
            MODULE.prepare_evaluation_input(self.source, self.source, "generated")
        self.assertEqual(self.source.read_bytes(), original_bytes)

    def test_rejects_duplicate_headers_and_irregular_rows(self):
        fixtures = [
            (["generated", "generated"], [["a", "b"]], "duplicate"),
            (["generated", ""], [["a", "b"]], "empty column"),
            (["generated", "extra"], [["a"]], "wrong number"),
            (["generated"], [["a", "b"]], "wrong number"),
        ]
        for fields, rows, message in fixtures:
            with self.subTest(message=message, fields=fields):
                self.write_fixture(fields, rows)
                with self.assertRaisesRegex(ValueError, message):
                    self.prepare()
                self.assertFalse(self.output.exists())

    def test_explicit_existing_report_can_be_preserved_without_text_changes(self):
        self.write_fixture(["fixture_id", "report"], [["fixture-a", "candidate"]])
        self.assertEqual(self.prepare("report"), 1)
        _, rows = self.read_output()
        self.assertEqual(rows[0]["source_report"], "candidate")
        self.assertEqual(rows[0]["report"], "candidate")


if __name__ == "__main__":
    unittest.main()
