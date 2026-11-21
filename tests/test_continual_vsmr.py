import unittest
import torch
import torch.nn as nn
from types import SimpleNamespace

from models.base.continual_vsmr_model import ContinualVSMRViT
from models.base.vsmr_modules import QueryConditionedReconstructor, VisualSemanticDecomposer
from models.base.vsmr_losses import sinkhorn_transport, ot_distance
from models.base.vsmr_resynthesis import query_conditioned_reconstruction

from models.base.continual_vsmr import (
    FrozenFeatureMemory, loo_class_prototypes, prompt_consistency_loss,
    distillation_kl,
)



class DummyPatchEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.patch_size = (16, 16)
        self.projection = nn.Linear(3, dim)

    def forward(self, images):
        pooled = torch.nn.functional.adaptive_avg_pool2d(images, (2, 2))
        pooled = pooled.flatten(2).transpose(1, 2)
        return self.projection(pooled)


class DummyViTEncoder(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.num_features = dim
        self.patch_embed = DummyPatchEmbed(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 5, dim))
        self.pos_drop = nn.Identity()
        self.blocks = nn.ModuleList([nn.Linear(dim, dim), nn.Linear(dim, dim)])
        self.norm = nn.LayerNorm(dim)
        self.default_cfg = {
            "mean": (0.5, 0.5, 0.5),
            "std": (0.5, 0.5, 0.5),
            "interpolation": "bicubic",
            "crop_pct": 0.9,
        }

    def _pos_embed(self, patches):
        tokens = torch.cat([self.cls_token.expand(patches.shape[0], -1, -1), patches], dim=1)
        return self.pos_drop(tokens + self.pos_embed)


class TestContinualVSMRArchitecture(unittest.TestCase):
    def test_frozen_backbone_and_prompt_only_incremental_gradient(self):
        torch.manual_seed(11)
        model = ContinualVSMRViT(
            SimpleNamespace(
                scratch=False, vsmr_num_parts=4,
                vsmr_prompt_temperature=1.0,
                vsmr_global_weight=0.5,
                vsmr_local_weight_eval=0.5,
            ),
            encoder=DummyViTEncoder(),
        )
        images = torch.randn(3, 3, 224, 224)
        output = model(images)
        self.assertEqual(output["parts"].shape, (3, 4, 8))
        self.assertEqual(output["patch_features"].shape, (3, 4, 8))
        self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
        self.assertTrue(torch.isfinite(output["parts"]).all())

        model.freeze_for_incremental()
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertFalse(model.decomposer.training)
        self.assertFalse(model.reconstructor.training)
        self.assertFalse(model.fusion.training)
        model.assert_incremental_parameter_contract()
        prompt = model.decomposer.shared_prompts
        before = prompt.detach().clone()
        loss = model.decompose_features(
            output["cls_features"].detach(), output["patch_features"].detach()
        )["parts"].square().mean()
        loss.backward()
        self.assertIsNotNone(prompt.grad)
        self.assertTrue(torch.isfinite(prompt.grad).all())
        for name, parameter in model.named_parameters():
            if name != "decomposer.shared_prompts":
                self.assertIsNone(parameter.grad, name)
        optimizer = torch.optim.SGD([prompt], lr=0.1)
        optimizer.step()
        self.assertFalse(torch.equal(before, prompt.detach()))
        self.assertEqual(tuple(prompt.shape), (4, 8))

    def test_query_conditioned_reconstruction_has_prompt_and_query_gradient(self):
        torch.manual_seed(12)
        dim, parts, patches = 8, 4, 6
        decomposer = VisualSemanticDecomposer(embed_dim=dim, num_parts=parts)
        reconstructor = QueryConditionedReconstructor(embed_dim=dim)
        cls = torch.randn(3, dim)
        patch = torch.randn(3, patches, dim)
        _, query_parts, _ = decomposer(cls, patch)
        class_parts = torch.randn(3, 2, parts, dim, requires_grad=True)
        scores, reconstructed, transport, attention = query_conditioned_reconstruction(
            query_parts, patch, class_parts, reconstructor,
            epsilon=0.1, iterations=30, scale=4.0,
        )
        self.assertEqual(scores.shape, (3, 2))
        self.assertEqual(reconstructed.shape, (3, 2, parts, dim))
        self.assertEqual(transport.shape, (3, 2, parts, parts))
        self.assertEqual(attention.shape, (3, 2, parts, patches))
        scores.sum().backward()
        self.assertIsNotNone(decomposer.shared_prompts.grad)
        self.assertIsNotNone(class_parts.grad)
        self.assertTrue(torch.isfinite(decomposer.shared_prompts.grad).all())

    def test_sinkhorn_uniform_marginals_and_permutation_invariant_ot(self):
        torch.manual_seed(13)
        cost = torch.rand(2, 4, 4)
        transport = sinkhorn_transport(cost, epsilon=0.1, iterations=100)
        target = torch.full((2, 4), 0.25)
        self.assertTrue(torch.allclose(transport.sum(-1), target, atol=1e-4))
        self.assertTrue(torch.allclose(transport.sum(-2), target, atol=1e-4))
        parts = torch.nn.functional.normalize(torch.randn(1, 4, 8), dim=-1)
        permuted = parts[:, [2, 0, 3, 1]]
        distance = ot_distance(parts, permuted, epsilon=0.1, iterations=100)
        self.assertLess(float(distance.item()), 1e-4)

class TestContinualVSMRUtilities(unittest.TestCase):
    def test_feature_memory_stores_real_cls_and_patch_and_rejects_duplicate(self):
        memory = FrozenFeatureMemory(max_base_per_class=3)
        cls = torch.randn(4, 8)
        patches = torch.randn(4, 5, 8)
        labels = torch.tensor([0, 0, 1, 1])
        memory.append(cls, patches, labels, ["a", "b", "c", "d"], 0, is_base=True)
        cached_cls, cached_patches, cached_labels = memory.tensors("cpu")
        self.assertEqual(cached_cls.shape, (4, 8))
        self.assertEqual(cached_patches.shape, (4, 5, 8))
        self.assertTrue(torch.allclose(cached_cls, cls, atol=2e-3))
        self.assertTrue(torch.equal(cached_labels, labels))
        with self.assertRaises(RuntimeError):
            memory.append(cls[:1], patches[:1], labels[:1], ["a"], 1)

    def test_base_budget_is_enforced(self):
        memory = FrozenFeatureMemory(max_base_per_class=2)
        cls = torch.randn(3, 4)
        patches = torch.randn(3, 5, 4)
        labels = torch.tensor([0, 0, 0])
        memory.append(cls, patches, labels, ["a", "b", "c"], 0, is_base=True)
        self.assertEqual(len(memory), 2)

    def test_loo_candidate_excludes_query_from_own_class(self):
        torch.manual_seed(3)
        parts = torch.nn.functional.normalize(torch.randn(4, 3, 6), dim=-1)
        labels = torch.tensor([0, 0, 1, 1])
        candidates = loo_class_prototypes(
            parts, labels, torch.tensor([0, 2]), torch.tensor([0, 1]),
            epsilon=0.1, iterations=30, barycenter_iterations=2)
        self.assertEqual(candidates.shape, (2, 2, 3, 6))
        # For query 0, its own-class candidate is constructed from sample 1.
        expected = parts[1]
        self.assertTrue(torch.allclose(candidates[0, 0], expected, atol=1e-5))

    def test_prompt_consistency_is_permutation_invariant_and_differentiable(self):
        torch.manual_seed(4)
        teacher = torch.nn.functional.normalize(torch.randn(2, 4, 8), dim=-1)
        student = teacher[:, [2, 0, 3, 1]].clone().requires_grad_()
        loss = prompt_consistency_loss(student, teacher, epsilon=0.1, iterations=40)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(student.grad)
        self.assertTrue(torch.isfinite(student.grad).all())


if __name__ == "__main__":
    unittest.main()
