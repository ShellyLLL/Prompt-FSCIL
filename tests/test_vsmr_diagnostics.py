import unittest

import torch
import torch.nn.functional as F

from models.base.vsmr_diagnostics import (
    component_set_diagnostics,
    prompt_gradient_diagnostics,
)


class TestVSMRDiagnostics(unittest.TestCase):
    def test_component_diagnostics_are_finite_and_class_conditional(self):
        torch.manual_seed(101)
        parts = F.normalize(torch.randn(12, 4, 16), dim=-1)
        attention = torch.softmax(torch.randn(12, 4, 25), dim=-1)
        labels = torch.tensor([0] * 6 + [1] * 6)
        result = component_set_diagnostics(
            parts, attention, labels, epsilon=0.1, iterations=20,
            max_samples=12, max_pairs=64,
        )
        required = [
            "component_pairwise_cosine_mean",
            "attention_entropy_normalized_mean",
            "attention_overlap_mean",
            "patch_coverage_top10_union_mean",
            "within_class_component_ot_distance_mean",
            "between_class_component_ot_distance_mean",
            "between_minus_within_component_ot_distance",
        ]
        for key in required:
            self.assertIsNotNone(result[key])
            self.assertTrue(torch.isfinite(torch.tensor(result[key])))
        self.assertEqual(result["component_count"], 4)
        self.assertGreaterEqual(result["patch_coverage_top10_union_mean"], 0.0)
        self.assertLessEqual(result["patch_coverage_top10_union_mean"], 1.0)

    def test_stratified_sampling_keeps_within_and_between_class_pairs(self):
        torch.manual_seed(103)
        parts = F.normalize(torch.randn(200, 3, 8), dim=-1)
        attention = torch.softmax(torch.randn(200, 3, 16), dim=-1)
        labels = torch.arange(100).repeat_interleave(2)
        result = component_set_diagnostics(
            parts, attention, labels, epsilon=0.1, iterations=10,
            max_samples=48, max_pairs=128,
        )
        self.assertEqual(result["diagnostic_samples"], 48)
        self.assertIsNotNone(result["within_class_component_ot_distance_mean"])
        self.assertIsNotNone(result["between_class_component_ot_distance_mean"])

    def test_prompt_gradient_diagnostics_match_weighted_gradient_ratio(self):
        torch.manual_seed(102)
        prompt = torch.nn.Parameter(torch.randn(4, 8))
        classification = (prompt - 1.0).square().mean()
        history = (prompt + 0.5).square().mean()
        result = prompt_gradient_diagnostics(
            classification, history, prompt, history_weight=0.2
        )
        self.assertGreater(result["classification_prompt_grad_norm"], 0.0)
        self.assertGreater(result["history_prompt_grad_norm_unweighted"], 0.0)
        self.assertAlmostEqual(
            result["weighted_history_to_classification_grad_ratio"],
            result["history_prompt_grad_norm_weighted"]
            / result["classification_prompt_grad_norm"],
            places=6,
        )
        self.assertTrue(-1.0001 <= result["history_classification_grad_cosine"] <= 1.0001)


if __name__ == "__main__":
    unittest.main()
