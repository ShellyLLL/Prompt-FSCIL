import unittest
import tempfile
import os
import torch
import torch.nn as nn
from types import SimpleNamespace

from models.base.continual_vsmr_model import ContinualVSMRViT
from models.base.continual_vsmr_trainer import ContinualVSMRFSCILTrainer
from models.base.vsmr_modules import QueryConditionedReconstructor, VisualSemanticDecomposer
from models.base.vsmr_losses import sinkhorn_transport, ot_distance
from models.base.vsmr_resynthesis import query_conditioned_reconstruction

from models.base.continual_vsmr import (
    FrozenFeatureMemory, loo_class_prototypes, prompt_consistency_loss,
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



class TinyImageDataset(torch.utils.data.Dataset):
    def __init__(self, labels, seed):
        self.targets = list(labels)
        self.transform = lambda image: image
        generator = torch.Generator().manual_seed(seed)
        self.images = [torch.randn(3, 224, 224, generator=generator) for _ in labels]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], self.targets[index]


class TestContinualVSMRTrainingPipeline(unittest.TestCase):
    def test_synthetic_base_incremental_evaluation_and_complete_report(self):
        torch.manual_seed(21)
        args = SimpleNamespace(
            base_class=2, way=2, shot=2, sessions=2,
            epochs_base=1, epochs_new=1,
            num_workers=0, test_batch_size=8,
            vsmr_classes_per_batch=2, vsmr_samples_per_class=2,
            continual_classes_per_batch=2, continual_samples_per_class=2,
            continual_base_memory_per_class=2,
            vsmr_lr=1e-3, vsmr_weight_decay=0.0,
            continual_prompt_lr=1e-3, continual_prompt_weight_decay=0.0,
            vsmr_ot_epsilon=0.1, vsmr_sinkhorn_iterations=15,
            vsmr_barycenter_iterations=2, vsmr_logit_scale=4.0,
            vsmr_local_weight=1.0, vsmr_joint_weight=1.0,
            continual_local_weight=1.0, continual_joint_weight=1.0,
            continual_history_consistency_weight=0.1,
        )
        trainer = ContinualVSMRFSCILTrainer.__new__(ContinualVSMRFSCILTrainer)
        trainer.args = args
        trainer.device = torch.device("cpu")
        trainer.seed = 21
        trainer.model = ContinualVSMRViT(
            SimpleNamespace(
                scratch=False, vsmr_num_parts=3,
                vsmr_prompt_temperature=1.0,
                vsmr_global_weight=0.5,
                vsmr_local_weight_eval=0.5,
            ),
            encoder=DummyViTEncoder(),
        )
        trainer.epsilon = 0.1
        trainer.sinkhorn_steps = 15
        trainer.barycenter_steps = 2
        trainer.logit_scale = 4.0
        trainer.memory_per_base_class = 2
        trainer.memory = FrozenFeatureMemory(2)
        trainer.base_reference = {}
        trainer.previous_reference = {}
        trainer.global_prototypes = None
        trainer.local_prototypes = None
        trainer.classes = None
        trainer.session_records = []
        trainer.confusion_matrices = {}
        trainer._train_transform = lambda: (lambda image: image)
        trainer._eval_transform = lambda: (lambda image: image)
        trainer._append_log = lambda record: None

        with tempfile.TemporaryDirectory() as directory:
            trainer.save_dir = directory
            base = TinyImageDataset([0, 0, 0, 1, 1, 1], seed=22)
            trainer._train_base(base)
            self.assertEqual(len(trainer.memory), 4)
            self.assertEqual(trainer.classes.tolist(), [0, 1])
            self.assertEqual(trainer.global_prototypes.shape, (2, 8))
            self.assertEqual(trainer.local_prototypes.shape, (2, 3, 8))
            trainer._evaluate(TinyImageDataset([0, 1], seed=23), 0)

            support = TinyImageDataset([2, 2, 3, 3], seed=24)
            trainer._incremental_session(support, 1)
            self.assertEqual(len(trainer.memory), 8)
            self.assertEqual(trainer.classes.tolist(), [0, 1, 2, 3])
            self.assertEqual(trainer.global_prototypes.shape, (4, 8))
            self.assertEqual(trainer.local_prototypes.shape, (4, 3, 8))
            self.assertEqual(
                [name for name, parameter in trainer.model.named_parameters() if parameter.requires_grad],
                ["decomposer.shared_prompts"],
            )
            trainer._evaluate(TinyImageDataset([0, 1, 2, 3], seed=25), 1)
            summary = trainer._write_final_report(1.0)
            self.assertEqual(summary["num_sessions"], 2)
            self.assertAlmostEqual(
                summary["AA"],
                sum(summary["session_class_average_accuracy"]) / 2.0,
            )
            self.assertAlmostEqual(
                summary["PD"],
                summary["session_class_average_accuracy"][0]
                - summary["session_class_average_accuracy"][1],
            )
            self.assertTrue(os.path.isfile(os.path.join(directory, "session_metrics.csv")))
            self.assertTrue(os.path.isfile(os.path.join(directory, "summary.json")))
            self.assertTrue(os.path.isfile(os.path.join(directory, "continual_vsmr_final.pth")))

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
        parts = model.decompose_features(
            output["cls_features"].detach(), output["patch_features"].detach()
        )["parts"]
        probe = torch.randn_like(parts)
        loss = (parts * probe).sum()
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
