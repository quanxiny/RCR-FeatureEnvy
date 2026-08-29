import csv
import tempfile
import unittest
from pathlib import Path

from scripts.aggregate_results import objective_name, write_union_csv


class AggregateResultsTests(unittest.TestCase):
    def test_cl5_controls_and_local_objective_are_distinct(self):
        self.assertEqual(
            objective_name({"method": "CL5", "lambda_local": "0"}),
            "Focal-only-control")
        self.assertEqual(
            objective_name({"method": "CL5", "lambda_local": "0.1"}),
            "CL5-graph-pair-local")
        self.assertEqual(
            objective_name({"method": "CL6"}),
            "CL6-directed-method-class")
        self.assertEqual(objective_name({"method": "CL2"}), "CL2")

    def test_union_csv_keeps_late_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"
            write_union_csv(path, [{"a": 1}, {"a": 2, "b": 3}])
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [
                {"a": "1", "b": ""},
                {"a": "2", "b": "3"},
            ])


if __name__ == "__main__":
    unittest.main()
