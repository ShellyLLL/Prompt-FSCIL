import unittest

import torch

from models.base.continual_vsmr_metrics import (
    evaluate_session_predictions,
    summarize_session_metrics,
    format_session_table,
)


class TestContinualVSMRSessionMetrics(unittest.TestCase):
    def test_macro_accuracy_is_per_class_not_sample_weighted(self):
        # Class 0 has many more examples, but contributes exactly one class score.
        labels = torch.tensor([0] * 10 + [1] * 2 + [2] * 2)
        predictions = torch.tensor([0] * 8 + [1] * 2 + [1] + [2] + [1] + [2])
        result = evaluate_session_predictions(
            labels, predictions, base_class=2, seen_classes=3,
            session=1, way=1,
        )
        expected_macro = (0.8 + 0.5 + 0.5) / 3.0
        self.assertAlmostEqual(result["class_average_accuracy"], expected_macro, places=6)
        self.assertAlmostEqual(result["sample_accuracy"], 12 / 14, places=6)
        self.assertAlmostEqual(result["base_class_average_accuracy"], 0.65, places=6)
        self.assertAlmostEqual(result["current_new_class_average_accuracy"], 0.5, places=6)
        self.assertEqual(len(result["confusion_matrix"]), 3)

    def test_aa_is_mean_of_all_session_macro_accuracies_and_pd_is_first_minus_last(self):
        records = [
            {"session": 0, "class_average_accuracy": 0.90, "sample_accuracy": 0.91},
            {"session": 1, "class_average_accuracy": 0.80, "sample_accuracy": 0.82},
            {"session": 2, "class_average_accuracy": 0.60, "sample_accuracy": 0.65},
        ]
        summary = summarize_session_metrics(records)
        self.assertAlmostEqual(summary["AA"], (0.90 + 0.80 + 0.60) / 3)
        self.assertAlmostEqual(summary["PD"], 0.90 - 0.60)
        self.assertAlmostEqual(summary["AA_percent"], 76.6666666, places=4)
        self.assertAlmostEqual(summary["PD_percentage_points"], 30.0, places=6)
        self.assertIn("AA = 76.67%", format_session_table(records))
        self.assertIn("PD = 30.00 percentage points", format_session_table(records))

    def test_summary_rejects_missing_or_repeated_sessions(self):
        with self.assertRaises(RuntimeError):
            summarize_session_metrics([
                {"session": 0, "class_average_accuracy": 0.8, "sample_accuracy": 0.8},
                {"session": 2, "class_average_accuracy": 0.7, "sample_accuracy": 0.7},
            ])

    def test_evaluation_rejects_missing_seen_class(self):
        with self.assertRaises(RuntimeError):
            evaluate_session_predictions(
                torch.tensor([0, 0, 1, 1]),
                torch.tensor([0, 0, 1, 1]),
                base_class=2, seen_classes=3, session=1, way=1,
            )


if __name__ == "__main__":
    unittest.main()
