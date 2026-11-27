import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base.continual_vsmr import (
    FrozenFeatureMemory,
    normalized_class_means,
    leave_one_out_cosine_logits,
    cosine_consistency_loss,
)
from models.base.continual_vsmr_model import (
    ContinualVSMRViT,
    PromptGuidedVisualContext,
)
from models.base.continual_vsmr_trainer import ContinualVSMRFSCILTrainer


class DummyPatchEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.patch_size = (16, 16)
        self.projection = nn.Linear(3, dim)

    def forward(self, images):
        pooled = F.adaptive_avg_pool2d(images, (2, 2)).flatten(2).transpose(1, 2)
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
            "mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5),
            "interpolation": "bicubic", "crop_pct": 0.9,
        }

    def _pos_embed(self, patches):
        tokens = torch.cat([
            self.cls_token.expand(patches.shape[0], -1, -1), patches
        ], dim=1)
        return self.pos_drop(tokens + self.pos_embed)


class TinyImageDataset(torch.utils.data.Dataset):
    def __init__(self, labels, seed=1):
        self.targets = list(labels)
        self.transform = lambda image: image
        generator = torch.Generator().manual_seed(seed)
        self.images = [
            torch.randn(3, 224, 224, generator=generator) for _ in labels
        ]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.targets[index]


class TestVSMRPromptArchitecture(unittest.TestCase):
    def test_single_prompt_and_incremental_parameter_contract(self):
        model = ContinualVSMRViT(
            SimpleNamespace(scratch=False, vsmr_prompt_temperature=1.0,
                            continual_global_init=0.5),
            encoder=DummyViTEncoder(),
        )
        cls, patches = model.backbone(torch.randn(3, 3, 224, 224))
        output = model.forward_features(cls, patches)
        self.assertEqual(output["global_features"].shape, (3, 8))
        self.assertEqual(output["local_features"].shape, (3, 8))
        self.assertEqual(output["attention"].shape, (3, 4))
        model.freeze_for_incremental()
        trainable = [
            n for n, p in model.named_parameters() if p.requires_grad
        ]
        self.assertEqual(trainable, ["context.shared_prompt"])
        prompt_before = model.context.shared_prompt.detach().clone()
        model.forward_features(cls, patches)["local_features"].sum().backward()
        self.assertIsNotNone(model.context.shared_prompt.grad)
        for name, parameter in model.named_parameters():
            if name != "context.shared_prompt":
                self.assertIsNone(parameter.grad, name)
        torch.optim.SGD([model.context.shared_prompt], lr=0.1).step()
        self.assertFalse(torch.equal(prompt_before, model.context.shared_prompt.detach()))

    def test_projected_context_matches_raw_context(self):
        torch.manual_seed(2)
        context = PromptGuidedVisualContext(embed_dim=8)
        cls = torch.randn(4, 8)
        patches = torch.randn(4, 5, 8)
        keys, values = context.project_patches(patches)
        a = context(cls, patches)[1]
        b, _ = context.aggregate_projected(keys, values)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))


class TestVSMRPrototypeUtilities(unittest.TestCase):
    def test_feature_memory_fp16_budget_and_duplicate_guard(self):
        memory = FrozenFeatureMemory(5)
        cls = torch.randn(6, 8)
        keys = torch.randn(6, 4, 8)
        values = torch.randn(6, 4, 8)
        labels = torch.tensor([0, 0, 0, 0, 0, 0])
        memory.append(cls, keys, values, labels, list("abcdef"), 0, is_base=True)
        self.assertEqual(len(memory), 5)
        self.assertEqual(memory.records[0].keys.dtype, torch.float16)
        out_cls, out_keys, out_values, out_labels = memory.tensors("cpu")
        self.assertEqual(out_cls.shape, (5, 8))
        self.assertEqual(out_keys.shape, (5, 4, 8))
        self.assertTrue(torch.equal(out_labels, torch.zeros(5, dtype=torch.long)))
        with self.assertRaises(RuntimeError):
            memory.append(cls[:1], keys[:1], values[:1], labels[:1], ["a"], 1)

    def test_loo_excludes_query_and_has_gradient(self):
        features = F.normalize(torch.tensor([
            [1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.2, 0.8]
        ]), dim=-1).requires_grad_()
        labels = torch.tensor([0, 0, 1, 1])
        scores = leave_one_out_cosine_logits(
            features, labels, torch.arange(4), torch.tensor([0, 1]), scale=1.0
        )
        self.assertEqual(scores.shape, (4, 2))
        # Query 0's own-class prototype is exactly sample 1.
        self.assertAlmostEqual(float(scores[0, 0]), float(features[0] @ features[1]), places=5)
        scores.sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_history_consistency_is_differentiable_and_reference_detached(self):
        reference = F.normalize(torch.randn(5, 8), dim=-1).requires_grad_()
        student = (reference.detach() + 0.1 * torch.randn(5, 8)).requires_grad_()
        loss = cosine_consistency_loss(student, reference)
        loss.backward()
        self.assertIsNotNone(student.grad)
        self.assertIsNone(reference.grad)
        self.assertTrue(torch.isfinite(student.grad).all())

    def test_class_mean_prototypes_are_normalized(self):
        x = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]])
        y = torch.tensor([0, 0, 1, 1])
        classes, prototypes = normalized_class_means(x, y)
        self.assertEqual(classes.tolist(), [0, 1])
        self.assertTrue(torch.allclose(prototypes.norm(dim=-1), torch.ones(2)))


class TestVSMRSyntheticPipeline(unittest.TestCase):
    def test_base_incremental_checkpoint_and_session_metrics(self):
        torch.manual_seed(11)
        args = SimpleNamespace(
            base_class=2, way=1, shot=5, sessions=2,
            epochs_base=1, epochs_new=2, num_workers=0, test_batch_size=8,
            continual_feature_batch_size=8, continual_eval_batch_size=8,
            continual_classes_per_batch=2, continual_samples_per_class=2,
            continual_base_memory_per_class=5,
            vsmr_lr=1e-3, vsmr_weight_decay=0.0,
            continual_prompt_lr=1e-3, continual_prompt_weight_decay=0.0,
            continual_history_rep_weight=0.1, continual_history_proto_weight=0.1,
            continual_global_loss_weight=1.0, continual_local_loss_weight=1.0,
            continual_joint_loss_weight=1.0, continual_global_init=0.5,
            vsmr_prompt_temperature=1.0, vsmr_logit_scale=4.0,
        )
        trainer = ContinualVSMRFSCILTrainer.__new__(ContinualVSMRFSCILTrainer)
        trainer.args = args
        trainer.device = torch.device("cpu")
        trainer.seed = 11
        trainer.model = ContinualVSMRViT(
            SimpleNamespace(scratch=False, vsmr_prompt_temperature=1.0,
                            continual_global_init=0.5),
            encoder=DummyViTEncoder(),
        )
        trainer.logit_scale = 4.0
        trainer.memory_per_class = 5
        trainer.memory = FrozenFeatureMemory(5)
        trainer.sample_birth = {}
        trainer.prototype_birth = {}
        trainer.global_prototypes = None
        trainer.local_prototypes = None
        trainer.classes = None
        trainer.session_records = []
        trainer.confusion_matrices = {}
        trainer.timings = {}
        trainer._append_log = lambda record: None
        # Synthetic tensors are already transformed; bypass PIL-only transforms.
        trainer._eval_transform = lambda: (lambda image: image)
        with tempfile.TemporaryDirectory() as directory:
            trainer.save_dir = directory
            base = TinyImageDataset([0] * 6 + [1] * 6, seed=12)
            trainer._train_base(base)
            self.assertEqual(len(trainer.memory), 10)
            self.assertEqual(trainer.classes.tolist(), [0, 1])
            self.assertEqual(trainer.global_prototypes.shape, (2, 8))
            self.assertEqual(trainer.local_prototypes.shape, (2, 8))
            global_birth = trainer.global_prototypes.clone()
            prototype_birth = {
                key: value.clone() for key, value in trainer.prototype_birth.items()
            }
            sample_birth = {
                key: value.clone() for key, value in trainer.sample_birth.items()
            }
            base_checkpoint = torch.load(
                os.path.join(directory, "continual_vsmr_base.pth"),
                map_location="cpu", weights_only=False,
            )
            self.assertTrue(base_checkpoint["memory_included"])
            self.assertEqual(len(base_checkpoint["feature_memory"]), 10)
            trainer._evaluate(TinyImageDataset([0, 1], seed=13), 0)

            support = TinyImageDataset([2] * 5, seed=14)
            trainer._incremental_session(support, 1)
            self.assertEqual(len(trainer.memory), 15)
            self.assertEqual(trainer.classes.tolist(), [0, 1, 2])
            # Global prototypes and all historical Birth References are immutable.
            self.assertTrue(torch.equal(trainer.global_prototypes[:2], global_birth))
            for class_id, reference in prototype_birth.items():
                self.assertTrue(torch.equal(trainer.prototype_birth[class_id], reference))
            for sample_id, reference in sample_birth.items():
                self.assertTrue(torch.equal(trainer.sample_birth[sample_id], reference))
            self.assertIn(2, trainer.prototype_birth)
            self.assertEqual(
                [n for n, p in trainer.model.named_parameters() if p.requires_grad],
                ["context.shared_prompt"],
            )
            trainer._evaluate(TinyImageDataset([0, 1, 2], seed=15), 1)
            self.assertTrue(os.path.isfile(os.path.join(directory, "continual_vsmr_base.pth")))
            self.assertTrue(os.path.isfile(os.path.join(directory, "continual_vsmr_S1.pth")))


if __name__ == "__main__":
    unittest.main()
