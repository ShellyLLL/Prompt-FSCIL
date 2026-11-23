"""End-to-end Continual VSMR-Prompt FSCIL trainer.

Base learns the full visual decomposition/reconstruction model on real base
images. Each incremental session updates only the fixed shared Prompt using
real K-shot support features and replayed frozen CLS/Patch feature memory.
"""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from tqdm import tqdm

from dataloader.data_utils import get_base_dataloader, get_new_dataloader, set_up_datasets
from .continual_vsmr import (
    FrozenFeatureMemory,
    loo_class_prototypes,
    prompt_consistency_loss,
    decompose_cached,
)
from .continual_vsmr_model import ContinualVSMRViT
from .continual_vsmr_metrics import (
    evaluate_session_predictions,
    summarize_session_metrics,
    write_session_csv,
    write_confusion_csv,
    format_session_table,
)
from .vsmr_losses import (
    global_logits,
    leave_one_out_global,
    ot_barycenter,
    local_logits,
)
from .vsmr_resynthesis import (
    local_leave_one_out_query_logits,
    query_conditioned_reconstruction,
    query_conditioned_fixed_index_reconstruction,
)
from .vsmr_diagnostics import (
    component_set_diagnostics,
    prompt_gradient_diagnostics,
)


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        if len(item) == 2:
            image, label = item
        elif len(item) == 3:
            image, label, _ = item
        else:
            raise ValueError("Dataset must return (image,label[,metadata]).")
        return image, int(label), int(index)


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, classes_per_batch, samples_per_class, seed):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.seed = int(seed)
        self.epoch = 0
        self.mapping = {
            int(cls): np.flatnonzero(self.labels == cls).tolist()
            for cls in np.unique(self.labels)
        }
        if self.classes_per_batch < 2 or self.samples_per_class < 2:
            raise ValueError("Balanced batches require >=2 classes and samples/class.")
        if len(self.mapping) < self.classes_per_batch:
            raise ValueError("Not enough classes for a balanced batch.")
        if any(len(indices) < self.samples_per_class for indices in self.mapping.values()):
            raise ValueError("A base class has too few images for a balanced batch.")
        self.num_batches = max(
            1, int(np.ceil(len(self.labels) /
                           (self.classes_per_batch * self.samples_per_class)))
        )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        classes = np.asarray(sorted(self.mapping))
        for _ in range(self.num_batches):
            selected = rng.choice(classes, self.classes_per_batch, replace=False)
            batch = []
            for cls in selected:
                batch.extend(rng.choice(
                    self.mapping[int(cls)], self.samples_per_class, replace=False
                ).tolist())
            rng.shuffle(batch)
            yield batch


class ContinualVSMRFSCILTrainer:
    def __init__(self, args):
        args.vit = True
        self.args = set_up_datasets(args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = max(0, int(getattr(args, "seed", 1)))
        self._seed_everything(self.seed)
        self.model = ContinualVSMRViT(self.args).to(self.device)
        self.epsilon = float(getattr(args, "vsmr_ot_epsilon", 0.05))
        self.sinkhorn_steps = int(getattr(args, "vsmr_sinkhorn_iterations", 30))
        self.barycenter_steps = int(getattr(args, "vsmr_barycenter_iterations", 5))
        self.logit_scale = float(getattr(args, "vsmr_logit_scale", 16.0))
        self.memory_per_base_class = int(getattr(args, "continual_base_memory_per_class", 5))
        self.memory = FrozenFeatureMemory(self.memory_per_base_class)
        self.base_reference = {}
        self.previous_reference = {}
        self.global_prototypes = None
        self.local_prototypes = None
        self.classes = None
        self.session_records = []
        self.confusion_matrices = {}
        root = getattr(args, "out", None) or "ContinualVSMR"
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        root = root if os.path.isabs(root) else os.path.join(repo_root, "checkpoint", root)
        self.save_dir = os.path.join(
            root, self.args.dataset, "ContinualVSMR",
            "{}_seed{}".format(time.strftime("%Y%m%d_%H%M%S"), self.seed)
        )
        os.makedirs(self.save_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.save_dir, "metrics.jsonl")
        # Persist only settings that define this method. train.py is shared by
        # legacy baselines, so their unrelated defaults must not pollute the
        # reproducibility record for Continual VSMR-Prompt.
        active_config_keys = {
            "method", "project", "dataset", "dataroot", "index_list_root",
            "base_class", "way", "shot", "sessions", "seed", "gpu",
            "epochs_base", "epochs_new", "num_workers", "test_batch_size",
            "out", "vit", "vsmr_num_parts", "vsmr_prompt_temperature",
            "vsmr_global_weight", "vsmr_local_weight_eval",
            "vsmr_classes_per_batch", "vsmr_samples_per_class",
            "vsmr_lr", "vsmr_weight_decay", "vsmr_ot_epsilon",
            "vsmr_sinkhorn_iterations", "vsmr_barycenter_iterations",
            "vsmr_local_weight", "vsmr_joint_weight", "vsmr_logit_scale",
            "vsmr_diagnostics_max_samples", "vsmr_diagnostics_max_pairs",
            "continual_base_memory_per_class", "continual_prompt_lr",
            "continual_prompt_weight_decay", "continual_local_weight",
            "continual_joint_weight", "continual_history_consistency_weight",
            "continual_classes_per_batch", "continual_samples_per_class",
        }
        active_config = {
            key: value for key, value in vars(self.args).items()
            if key in active_config_keys
        }
        self._write_json(os.path.join(self.save_dir, "config.json"), active_config)
        print("[ContinualVSMR] output={}".format(self.save_dir))

    @staticmethod
    def _seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def _write_json(path, value):
        with open(path, "w") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, default=str)

    def _append_log(self, record):
        with open(self.metrics_path, "a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("[ContinualVSMR] " + json.dumps(record, ensure_ascii=False, sort_keys=True))

    def _interpolation(self):
        names = {
            "nearest": transforms.InterpolationMode.NEAREST,
            "bilinear": transforms.InterpolationMode.BILINEAR,
            "bicubic": transforms.InterpolationMode.BICUBIC,
            "lanczos": transforms.InterpolationMode.LANCZOS,
        }
        name = str(self.model.backbone.input_interpolation).lower()
        if name not in names:
            raise ValueError("Unsupported ViT interpolation: {}".format(name))
        return names[name]

    def _eval_transform(self):
        backbone = self.model.backbone
        resize = int(round(224.0 / float(backbone.input_crop_pct)))
        return transforms.Compose([
            transforms.Resize(resize, interpolation=self._interpolation()),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(backbone.input_mean, backbone.input_std),
        ])

    def _train_transform(self):
        backbone = self.model.backbone
        return transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=self._interpolation()),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(backbone.input_mean, backbone.input_std),
        ])

    def _loader(self, dataset, batch_size, batch_sampler=None):
        workers = int(getattr(self.args, "num_workers", 4))
        common = {
            "num_workers": workers,
            "pin_memory": self.device.type == "cuda",
            "persistent_workers": workers > 0,
        }
        wrapped = IndexedDataset(dataset)
        if batch_sampler is not None:
            return DataLoader(wrapped, batch_sampler=batch_sampler, **common)
        return DataLoader(wrapped, batch_size=int(batch_size), shuffle=False, **common)

    @torch.no_grad()
    def _extract_dataset(self, dataset, batch_size=None):
        original = getattr(dataset, "transform", None)
        if original is None:
            raise RuntimeError("Dataset must expose a transform.")
        dataset.transform = self._eval_transform()
        loader = self._loader(dataset, batch_size or int(self.args.test_batch_size))
        cls_all, patches_all, labels_all, ids_all = [], [], [], []
        self.model.eval()
        try:
            for images, labels, indices in tqdm(loader, desc="Frozen feature extraction"):
                cls, patches = self.model.backbone(images.to(self.device, non_blocking=True))
                cls_all.append(cls.cpu())
                patches_all.append(patches.cpu())
                labels_all.append(torch.as_tensor(labels, dtype=torch.long).cpu())
                ids_all.extend([int(i) for i in indices.tolist()])
        finally:
            dataset.transform = original
        if not labels_all:
            raise RuntimeError("Cannot extract features from an empty dataset.")
        return (
            torch.cat(cls_all), torch.cat(patches_all),
            torch.cat(labels_all), ids_all,
        )

    def _append_dataset_to_memory(self, dataset, session, is_base=False):
        cls, patches, labels, indices = self._extract_dataset(dataset)
        full_cls, full_patches, full_labels = cls, patches, labels
        if is_base:
            # Deterministic, reproducible class-balanced exemplars; never use test data.
            selected = []
            labels_np = labels.numpy()
            rng = np.random.default_rng(self.seed)
            for class_id in range(int(self.args.base_class)):
                candidates = np.flatnonzero(labels_np == class_id)
                if candidates.size < self.memory_per_base_class:
                    raise RuntimeError(
                        "Base class {} has {} training images; {} are required.".format(
                            class_id, candidates.size, self.memory_per_base_class
                        )
                    )
                chosen = np.sort(rng.choice(
                    candidates, self.memory_per_base_class, replace=False
                ))
                selected.extend(chosen.tolist())
            selected = torch.tensor(sorted(selected), dtype=torch.long)
            cls, patches, labels = cls[selected], patches[selected], labels[selected]
            indices = [indices[int(i)] for i in selected.tolist()]
        self.memory.append(
            cls, patches, labels,
            ["session{}:index{}".format(session, i) for i in indices],
            session, is_base=is_base,
        )
        if is_base:
            return full_cls, full_patches, full_labels
        return cls, patches, labels

    def _all_memory(self):
        cls, patches, labels = self.memory.tensors(self.device)
        return cls, patches, labels

    def _global_features(self, cls_features):
        return F.normalize(self.model.decomposer.global_projection(cls_features), dim=-1)

    def _current_parts(self, cls_features, patches):
        global_features, parts, _ = decompose_cached(
            self.model, cls_features, patches
        )
        return global_features, parts

    @torch.no_grad()
    def _log_component_diagnostics(self, stage, cls_features, patch_features, labels):
        """Log training-memory component statistics without changing optimization."""
        self.model.eval()
        _, parts, attention = self.model.decomposer(cls_features, patch_features)
        record = component_set_diagnostics(
            parts, attention, labels, epsilon=self.epsilon,
            iterations=self.sinkhorn_steps,
            max_samples=int(getattr(self.args, "vsmr_diagnostics_max_samples", 48)),
            max_pairs=int(getattr(self.args, "vsmr_diagnostics_max_pairs", 512)),
        )
        record.update({"stage": "component_diagnostics", "session": int(stage)})
        self._append_log(record)
        return record

    def _prototype_bank(self, cls_features, patch_features, labels):
        global_features, parts = self._current_parts(cls_features, patch_features)
        classes = torch.unique(labels, sorted=True)
        global_prototypes, local_prototypes = [], []
        for class_id in classes.tolist():
            mask = labels == int(class_id)
            if int(mask.sum()) < 2:
                raise RuntimeError(
                    "Class {} has fewer than two memory exemplars for LOO.".format(class_id)
                )
            global_prototypes.append(F.normalize(global_features[mask].mean(0), dim=-1))
            local_prototypes.append(ot_barycenter(
                parts[mask], self.epsilon, self.sinkhorn_steps, self.barycenter_steps
            ))
        return classes, torch.stack(global_prototypes), torch.stack(local_prototypes), parts

    def _batch_logits(self, cls_features, patch_features, labels):
        output = self.model.decompose_features(cls_features, patch_features)
        global_features, parts = output["global_features"], output["parts"]
        loo_global, inverse, classes = leave_one_out_global(global_features, labels)
        class_means = []
        for class_id in classes:
            class_means.append(F.normalize(
                global_features[labels == class_id].mean(0), dim=-1
            ))
        global_scores = global_logits(
            global_features, torch.stack(class_means), self.logit_scale
        )
        global_scores = global_scores.scatter(
            1, inverse[:, None],
            (global_features * loo_global).sum(-1, keepdim=True) * self.logit_scale
        )
        local_scores, local_classes = local_leave_one_out_query_logits(
            parts, patch_features, labels, self.model.reconstructor,
            epsilon=self.epsilon, iterations=self.sinkhorn_steps,
            logit_scale=self.logit_scale,
            barycenter_iterations=self.barycenter_steps,
        )
        if not torch.equal(classes, local_classes):
            raise RuntimeError("Global/local LOO class ordering mismatch.")
        joint_scores = self.model.fusion(global_scores, local_scores)
        return output, global_scores, local_scores, joint_scores, inverse

    def _train_base(self, trainset):
        trainset.transform = self._train_transform()
        labels = np.asarray(trainset.targets, dtype=np.int64)
        expected = np.arange(int(self.args.base_class))
        if not np.array_equal(np.unique(labels), expected):
            raise RuntimeError("Base training labels must be contiguous [0, base_class).")
        classes_per_batch = min(
            int(getattr(self.args, "vsmr_classes_per_batch", 16)),
            int(self.args.base_class),
        )
        sampler = BalancedBatchSampler(
            labels, classes_per_batch,
            int(getattr(self.args, "vsmr_samples_per_class", 4)), self.seed
        )
        loader = self._loader(trainset, 1, batch_sampler=sampler)
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=float(getattr(self.args, "vsmr_lr", 2e-4)),
            weight_decay=float(getattr(self.args, "vsmr_weight_decay", 1e-4)),
        )
        epochs = int(self.args.epochs_base)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs)
        )
        w_local = float(getattr(self.args, "vsmr_local_weight", 1.0))
        w_joint = float(getattr(self.args, "vsmr_joint_weight", 1.0))
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            self.model.train()
            totals = {"loss": 0.0, "global": 0.0, "local": 0.0,
                      "joint": 0.0}
            steps = 0
            for images, batch_labels, _ in tqdm(loader, desc="Base {}/{}".format(epoch + 1, epochs)):
                images = images.to(self.device, non_blocking=True)
                labels = torch.as_tensor(batch_labels, device=self.device).long()
                output, global_scores, local_scores, joint_scores, targets = self._batch_logits(
                    *self.model.backbone(images), labels
                )
                global_loss = F.cross_entropy(global_scores, targets)
                local_loss = F.cross_entropy(local_scores, targets)
                joint_loss = F.cross_entropy(joint_scores, targets)
                loss = global_loss + w_local * local_loss + w_joint * joint_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite Base loss.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for name, parameter in self.model.named_parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        if not torch.isfinite(parameter.grad).all():
                            raise FloatingPointError("Non-finite gradient: " + name)
                optimizer.step()
                totals["loss"] += float(loss.detach())
                totals["global"] += float(global_loss.detach())
                totals["local"] += float(local_loss.detach())
                totals["joint"] += float(joint_loss.detach())
                steps += 1
            scheduler.step()
            record = {"stage": "base_train", "epoch": epoch + 1}
            record.update({key: value / max(steps, 1) for key, value in totals.items()})
            record["fusion_alpha"] = float(self.model.fusion.alpha.detach())
            record["lr"] = scheduler.get_last_lr()[0]
            self._append_log(record)

        self.model.freeze_for_incremental()
        base_cls, base_patches, base_labels = self._append_dataset_to_memory(
            trainset, 0, is_base=True
        )
        cls, patches, labels = self._all_memory()
        # Prototype construction is a post-Base cache operation, not another
        # optimization step. Do not retain an autograd graph in the S0 bank.
        with torch.no_grad():
            self.classes, _, self.local_prototypes, base_parts = (
                self._prototype_bank(cls, patches, labels)
            )
            self.local_prototypes = self.local_prototypes.detach()
            # Base global prototypes use every real Base training image, while
            # local OT memory uses the deterministic per-class feature budget.
            base_global_features = self._global_features(base_cls.to(self.device))
            base_global_labels = base_labels.to(self.device)
            self.global_prototypes = torch.stack([
                F.normalize(base_global_features[base_global_labels == class_id].mean(0), dim=-1)
                for class_id in range(int(self.args.base_class))
            ]).detach()
        self._log_component_diagnostics(0, cls, patches, labels)
        self.base_reference = {
            record.sample_id: part.detach().cpu().clone()
            for record, part in zip(self.memory.records, base_parts)
        }
        self.previous_reference = {
            record.sample_id: part.detach().cpu().clone()
            for record, part in zip(self.memory.records, base_parts)
        }
        torch.save({
            "model": self.model.state_dict(),
            "global_prototypes": self.global_prototypes.cpu(),
            "local_prototypes": self.local_prototypes.cpu(),
            "memory_metadata": self.memory.metadata(),
            "feature_memory": self.memory.state_dict(),
            "base_reference": self.base_reference,
        }, os.path.join(self.save_dir, "continual_vsmr_base.pth"))
        self._append_log({
            "stage": "base_memory",
            "base_memory_per_class": self.memory_per_base_class,
            "memory_samples": len(self.memory),
            "memory_bytes": int(sum(
                record.cls.numel() * record.cls.element_size()
                + record.patches.numel() * record.patches.element_size()
                for record in self.memory.records
            )),
        })

    def _balanced_memory_indices(self, labels, classes_per_batch, samples_per_class, epoch):
        rng = np.random.default_rng(self.seed + 100003 * int(epoch))
        labels_np = labels.detach().cpu().numpy()
        mapping = {
            int(cls): np.flatnonzero(labels_np == cls)
            for cls in np.unique(labels_np)
        }
        classes = np.asarray(sorted(mapping))
        cnum = min(int(classes_per_batch), len(classes))
        snum = min(int(samples_per_class), min(len(v) for v in mapping.values()))
        if cnum < 2 or snum < 2:
            raise RuntimeError("Incremental replay needs >=2 classes and >=2 samples/class.")
        batches = max(1, int(np.ceil(len(labels_np) / (cnum * snum))))
        for _ in range(batches):
            chosen = rng.choice(classes, cnum, replace=False)
            batch = []
            for class_id in chosen:
                batch.extend(rng.choice(mapping[int(class_id)], snum, replace=False).tolist())
            rng.shuffle(batch)
            yield torch.tensor(batch, dtype=torch.long, device=labels.device)

    def _make_reference(self, cls_features, patch_features, labels, ids):
        with torch.no_grad():
            parts = self._current_parts(cls_features, patch_features)[1]
        return {
            ids[i]: parts[i].detach().cpu().clone()
            for i in range(len(ids))
        }

    def _incremental_session(self, support_set, session):
        expected = np.arange(
            int(self.args.base_class) + (session - 1) * int(self.args.way),
            int(self.args.base_class) + session * int(self.args.way),
        )
        support_labels = np.asarray(support_set.targets, dtype=np.int64)
        counts = {int(c): int((support_labels == c).sum()) for c in expected}
        if not np.array_equal(np.unique(support_labels), expected):
            raise RuntimeError("Session {} support classes mismatch: {}".format(session, counts))
        if any(value != int(self.args.shot) for value in counts.values()):
            raise RuntimeError("Session {} requires exact K-shot: {}".format(session, counts))

        previous_memory_size = len(self.memory)
        self._append_dataset_to_memory(support_set, session, is_base=False)
        expected_added = int(self.args.way) * int(self.args.shot)
        if len(self.memory) - previous_memory_size != expected_added:
            raise RuntimeError("Incremental memory did not append exactly the K-shot samples.")

        cls, patches, labels = self._all_memory()
        # Add only the current K-shot classes to the immutable global prototype
        # bank. Previously admitted global prototypes are never recomputed.
        new_global_features = self._global_features(
            cls[previous_memory_size:]
        )
        new_global_labels = labels[previous_memory_size:]
        new_global_prototypes = []
        for class_id in expected.tolist():
            mask = new_global_labels == int(class_id)
            new_global_prototypes.append(
                F.normalize(new_global_features[mask].mean(0), dim=-1)
            )
        self.global_prototypes = torch.cat([
            self.global_prototypes,
            torch.stack(new_global_prototypes),
        ], dim=0)
        previous_prompt = self.model.decomposer.shared_prompts.detach().clone()
        prompt = self.model.decomposer.shared_prompts
        self.model.assert_incremental_parameter_contract()

        # Historical references are immutable and exist only for samples admitted
        # before this Session. New support samples acquire references after training.
        historical_count = previous_memory_size
        reference_parts = []
        reference_indices = []
        for index in range(historical_count):
            record = self.memory.records[index]
            if record.sample_id not in self.base_reference and record.sample_id not in self.previous_reference:
                raise RuntimeError("Missing permanent historical reference: " + record.sample_id)
            reference = (
                self.base_reference[record.sample_id]
                if record.sample_id in self.base_reference
                else self.previous_reference[record.sample_id]
            )
            reference_parts.append(reference.to(self.device))
            reference_indices.append(index)
        reference_parts = torch.stack(reference_parts)
        reference_indices = torch.tensor(reference_indices, dtype=torch.long, device=self.device)

        optimizer = torch.optim.AdamW(
            [prompt],
            lr=float(getattr(self.args, "continual_prompt_lr", 1e-4)),
            weight_decay=float(getattr(self.args, "continual_prompt_weight_decay", 1e-4)),
        )
        epochs = int(getattr(self.args, "epochs_new", 20))
        w_local = float(getattr(self.args, "continual_local_weight", 1.0))
        w_joint = float(getattr(self.args, "continual_joint_weight", 1.0))
        w_history = float(getattr(self.args, "continual_history_consistency_weight", 0.1))
        classes_per_batch = int(getattr(self.args, "continual_classes_per_batch", 16))
        samples_per_class = int(getattr(self.args, "continual_samples_per_class", 4))

        for epoch in range(epochs):
            self.model.train()
            totals = {
                "total": 0.0, "local": 0.0, "joint": 0.0, "history": 0.0,
                "classification_prompt_grad_norm": 0.0,
                "history_prompt_grad_norm_unweighted": 0.0,
                "history_prompt_grad_norm_weighted": 0.0,
                "weighted_history_to_classification_grad_ratio": 0.0,
                "history_classification_grad_cosine": 0.0,
            }
            steps = 0
            for indices in self._balanced_memory_indices(
                labels, classes_per_batch, samples_per_class,
                epoch + session * epochs
            ):
                query_cls = cls[indices]
                query_patches = patches[indices]
                query_labels = labels[indices]
                output = self.model.decompose_features(query_cls, query_patches)

                all_classes = torch.unique(labels, sorted=True)
                all_parts = self._current_parts(cls, patches)[1]
                candidate_parts = loo_class_prototypes(
                    all_parts, labels, indices, all_classes,
                    self.epsilon, self.sinkhorn_steps, self.barycenter_steps,
                )
                global_scores = global_logits(
                    output["global_features"], self.global_prototypes, self.logit_scale
                )
                local_scores, _, _, _ = query_conditioned_reconstruction(
                    output["parts"], query_patches, candidate_parts,
                    self.model.reconstructor, self.epsilon,
                    self.sinkhorn_steps, self.logit_scale,
                )
                joint_scores = self.model.fusion(global_scores, local_scores)
                class_to_column = {
                    int(class_id): position
                    for position, class_id in enumerate(all_classes.tolist())
                }
                targets = torch.tensor(
                    [class_to_column[int(label)] for label in query_labels.tolist()],
                    dtype=torch.long, device=self.device,
                )
                local_loss = F.cross_entropy(local_scores, targets)
                joint_loss = F.cross_entropy(joint_scores, targets)

                # Re-decompose historical replay features in the student Prompt
                # and compare them to their immutable first-admission references.
                history_parts = all_parts[reference_indices]
                history_loss = prompt_consistency_loss(
                    history_parts,
                    reference_parts,
                    epsilon=self.epsilon,
                    iterations=self.sinkhorn_steps,
                )
                classification_objective = w_local * local_loss + w_joint * joint_loss
                gradient_diagnostics = prompt_gradient_diagnostics(
                    classification_objective, history_loss, prompt, w_history
                )
                loss = classification_objective + w_history * history_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite incremental loss.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if prompt.grad is None or not torch.isfinite(prompt.grad).all():
                    raise FloatingPointError("Shared Prompt gradient is absent or non-finite.")
                for name, parameter in self.model.named_parameters():
                    if name != "decomposer.shared_prompts" and parameter.grad is not None:
                        raise RuntimeError("Frozen parameter received gradient: " + name)
                optimizer.step()

                totals["total"] += float(loss.detach())
                totals["local"] += float(local_loss.detach())
                totals["joint"] += float(joint_loss.detach())
                totals["history"] += float(history_loss.detach())
                for key, value in gradient_diagnostics.items():
                    totals[key] += float(value)
                steps += 1

            record = {"stage": "incremental_train", "session": session, "epoch": epoch + 1}
            record.update({key: value / max(steps, 1) for key, value in totals.items()})
            record["prompt_delta_from_previous"] = float(
                (prompt.detach() - previous_prompt).norm()
            )
            record.update(gradient_diagnostics)
            self._append_log(record)

        # Recompute every seen-class prototype in the current Prompt space.
        # Historical references remain untouched; only this Session's samples
        # receive their first and permanent component references.
        with torch.no_grad():
            self.classes, _, self.local_prototypes, final_parts = (
                self._prototype_bank(cls, patches, labels)
            )
            historical_drift = prompt_consistency_loss(
                final_parts[reference_indices],
                reference_parts,
                epsilon=self.epsilon,
                iterations=self.sinkhorn_steps,
            )
            for index in range(previous_memory_size, len(self.memory)):
                sample_id = self.memory.records[index].sample_id
                self.previous_reference[sample_id] = final_parts[index].cpu().clone()

        self._log_component_diagnostics(session, cls, patches, labels)
        self._append_log({
            "stage": "memory_update",
            "session": session,
            "memory_samples": len(self.memory),
            "classes": int(self.classes.numel()),
            "prompt_delta_from_previous": float((prompt.detach() - previous_prompt).norm()),
            "historical_reference_count": historical_count,
            "new_reference_count": len(self.memory) - previous_memory_size,
            "historical_component_drift": float(historical_drift),
        })

    @torch.no_grad()
    def _evaluate(self, testset, session):
        original = getattr(testset, "transform", None)
        if original is None:
            raise RuntimeError("Test dataset must expose transform.")
        testset.transform = self._eval_transform()
        loader = self._loader(testset, int(self.args.test_batch_size))
        prediction_sets = {
            "joint_ot_query": [], "global_only": [], "local_ot_query": [],
            "ot_without_query_reconstruction": [], "fixed_index_query": [],
        }
        labels_all = []
        self.model.eval()
        try:
            for images, labels, _ in tqdm(loader, desc="Evaluate S{}".format(session)):
                output = self.model(images.to(self.device, non_blocking=True))
                global_scores = global_logits(
                    output["global_features"], self.global_prototypes, self.logit_scale
                )
                local_scores, _, _, _ = query_conditioned_reconstruction(
                    output["parts"], output["patch_features"], self.local_prototypes,
                    self.model.reconstructor, self.epsilon, self.sinkhorn_steps,
                    self.logit_scale,
                )
                fixed_scores, _, _ = query_conditioned_fixed_index_reconstruction(
                    output["parts"], output["patch_features"], self.local_prototypes,
                    self.model.reconstructor, scale=self.logit_scale,
                )
                no_query_scores = local_logits(
                    output["parts"], self.local_prototypes,
                    epsilon=self.epsilon, iterations=self.sinkhorn_steps,
                    scale=self.logit_scale,
                )
                joint_scores = self.model.fusion(global_scores, local_scores)
                score_sets = {
                    "joint_ot_query": joint_scores,
                    "global_only": global_scores,
                    "local_ot_query": local_scores,
                    "ot_without_query_reconstruction": no_query_scores,
                    "fixed_index_query": self.model.fusion(global_scores, fixed_scores),
                }
                for key, scores in score_sets.items():
                    prediction_sets[key].append(scores.argmax(dim=1).cpu())
                labels_all.append(torch.as_tensor(labels, dtype=torch.long).cpu())
        finally:
            testset.transform = original
        labels = torch.cat(labels_all)
        predictions = torch.cat(prediction_sets["joint_ot_query"])
        metrics = evaluate_session_predictions(
            labels, predictions, int(self.args.base_class),
            int(self.args.base_class + session * int(self.args.way) if session else self.args.base_class),
            session, int(self.args.way),
        )
        ablation_metrics = {}
        for key, batch_predictions in prediction_sets.items():
            branch_predictions = torch.cat(batch_predictions)
            branch_result = evaluate_session_predictions(
                labels, branch_predictions, int(self.args.base_class),
                int(self.args.base_class + session * int(self.args.way) if session else self.args.base_class),
                session, int(self.args.way),
            )
            ablation_metrics[key] = {
                "class_average_accuracy": branch_result["class_average_accuracy"],
                "sample_accuracy": branch_result["sample_accuracy"],
                "base_class_average_accuracy": branch_result["base_class_average_accuracy"],
                "old_novel_class_average_accuracy": branch_result["old_novel_class_average_accuracy"],
                "current_new_class_average_accuracy": branch_result["current_new_class_average_accuracy"],
            }
        metrics["inference_ablations"] = ablation_metrics
        metrics["stage"] = "evaluation"
        metrics["method"] = "ContinualVSMR-Prompt"
        self.confusion_matrices[session] = metrics.pop("confusion_matrix")
        self.session_records.append(metrics)
        self._append_log(metrics)
        write_confusion_csv(
            os.path.join(self.save_dir, "confusion_S{}.csv".format(session)),
            self.confusion_matrices[session],
        )
        return metrics

    def _write_final_report(self, elapsed):
        expected_sessions = int(self.args.sessions)
        actual_sessions = len(self.session_records)
        if actual_sessions != expected_sessions:
            raise RuntimeError(
                "Refusing to report partial FSCIL metrics: expected {} sessions "
                "(S0..S{}), but evaluated {}.".format(
                    expected_sessions, expected_sessions - 1, actual_sessions
                )
            )
        summary = summarize_session_metrics(self.session_records)
        summary.update({
            "stage": "summary",
            "method": "ContinualVSMR-Prompt",
            "elapsed_minutes": float(elapsed / 60.0),
            "base_memory_per_class": self.memory_per_base_class,
            "memory_samples_final": len(self.memory),
            "AA_definition": "Arithmetic mean of class-average accuracy over S0..ST.",
            "PD_definition": "S0 class-average accuracy minus ST class-average accuracy; percentage-point difference.",
        })
        self._append_log(summary)
        write_session_csv(
            os.path.join(self.save_dir, "session_metrics.csv"),
            self.session_records,
        )
        self._write_json(
            os.path.join(self.save_dir, "session_metrics.json"),
            self.session_records,
        )
        self._write_json(
            os.path.join(self.save_dir, "summary.json"), summary
        )
        with open(os.path.join(self.save_dir, "summary.txt"), "w") as handle:
            handle.write(format_session_table(self.session_records) + "\n")
            handle.write("\nSample-weighted Top-1 is reported separately and is not used for AA/PD.\n")
        torch.save({
            "model": self.model.state_dict(),
            "global_prototypes": self.global_prototypes.cpu(),
            "local_prototypes": self.local_prototypes.cpu(),
            "memory_metadata": self.memory.metadata(),
            "feature_memory": self.memory.state_dict(),
            "base_reference": self.base_reference,
            "previous_reference": self.previous_reference,
            "session_records": self.session_records,
            "summary": summary,
        }, os.path.join(self.save_dir, "continual_vsmr_final.pth"))
        print("\n" + format_session_table(self.session_records))
        return summary

    def train(self):
        started = time.time()
        trainset, _, testloader = get_base_dataloader(self.args)
        self._train_base(trainset)
        self._evaluate(testloader.dataset, 0)
        for session in range(1, int(self.args.sessions)):
            support_set, _, testloader = get_new_dataloader(self.args, session)
            self._incremental_session(support_set, session)
            self._evaluate(testloader.dataset, session)
        summary = self._write_final_report(time.time() - started)
        print("[ContinualVSMR] summary={}".format(os.path.join(self.save_dir, "summary.txt")))
        return summary
