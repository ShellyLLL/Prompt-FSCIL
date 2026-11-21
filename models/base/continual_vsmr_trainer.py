"""End-to-end Continual VSMR-Prompt FSCIL trainer.

Base learns the full visual decomposition/reconstruction model on real base
images. Each incremental session updates only the fixed shared Prompt using
real K-shot support features and replayed frozen CLS/Patch feature memory.
"""
import csv
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
    distillation_kl,
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
    attention_overlap_loss,
)
from .vsmr_resynthesis import (
    local_leave_one_out_query_logits,
    query_conditioned_reconstruction,
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
        self._write_json(os.path.join(self.save_dir, "config.json"), vars(self.args))
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
        return cls, patches, labels

    def _all_memory(self):
        cls, patches, labels = self.memory.tensors(self.device)
        return cls, patches, labels

    def _global_features(self, cls_features):
        return F.normalize(self.model.decomposer.global_projection(cls_features), dim=-1)

    def _current_parts(self, cls_features, patches):
        output = decompose_cached(self.model, cls_features, patches)
        return output["global_features"], output["parts"]

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
        w_overlap = float(getattr(self.args, "vsmr_overlap_weight", 0.01))
        for epoch in range(epochs):
            sampler.set_epoch(epoch)
            self.model.train()
            totals = {"loss": 0.0, "global": 0.0, "local": 0.0,
                      "joint": 0.0, "overlap": 0.0}
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
                overlap_loss = attention_overlap_loss(output["attention"])
                loss = global_loss + w_local * local_loss + w_joint * joint_loss + w_overlap * overlap_loss
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
                totals["overlap"] += float(overlap_loss.detach())
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
        self.classes, self.global_prototypes, self.local_prototypes, base_parts = (
            self._prototype_bank(cls, patches, labels)
        )
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
        old_memory_size = len(self.memory)
        self._append_dataset_to_memory(support_set, session, is_base=False)
        if len(self.memory) - old_memory_size != int(self.args.way) * int(self.args.shot):
            raise RuntimeError("Incremental memory did not append exactly the K-shot samples.")

        cls, patches, labels = self._all_memory()
        _, _, _, teacher_parts = self._prototype_bank(cls, patches, labels)
        teacher_prompt = self.model.decomposer.shared_prompts.detach().clone()
        previous_parts = teacher_parts.detach()
        previous_reference = {
            record.sample_id: previous_parts[index].detach().cpu().clone()
            for index, record in enumerate(self.memory.records)
        }
        old_classes = torch.arange(int(self.args.base_class) + (session - 1) * int(self.args.way),
                                   device=self.device)
        old_mask = torch.isin(labels, old_classes)
        old_indices = torch.nonzero(old_mask, as_tuple=False).flatten()
        old_cls = cls[old_indices]
        old_patches = patches[old_indices]
        old_labels = labels[old_indices]
        old_parts = previous_parts[old_indices]
        old_ids = [self.memory.records[int(i)].sample_id for i in old_indices.cpu().tolist()]
        old_base_mask = old_labels < int(self.args.base_class)
        old_inc_mask = ~old_base_mask

        self.model.assert_incremental_parameter_contract()
        prompt = self.model.decomposer.shared_prompts
        previous_prompt = teacher_prompt.to(self.device)
        optimizer = torch.optim.AdamW(
            [prompt],
            lr=float(getattr(self.args, "continual_prompt_lr", 1e-4)),
            weight_decay=float(getattr(self.args, "continual_prompt_weight_decay", 1e-4)),
        )
        epochs = int(getattr(self.args, "epochs_new", 20))
        w_local = float(getattr(self.args, "continual_local_weight", 1.0))
        w_joint = float(getattr(self.args, "continual_joint_weight", 1.0))
        w_prev = float(getattr(self.args, "continual_previous_consistency_weight", 0.1))
        w_base = float(getattr(self.args, "continual_base_consistency_weight", 0.1))
        w_kd = float(getattr(self.args, "continual_distillation_weight", 0.5))
        w_anchor = float(getattr(self.args, "continual_prompt_anchor_weight", 1e-3))
        w_overlap = float(getattr(self.args, "vsmr_overlap_weight", 0.01))
        temperature = float(getattr(self.args, "continual_distillation_temperature", 2.0))
        classes_per_batch = int(getattr(self.args, "continual_classes_per_batch", 16))
        samples_per_class = int(getattr(self.args, "continual_samples_per_class", 4))

        # Previous-session teacher outputs are fixed for this Session.
        with torch.no_grad():
            old_teacher_parts = old_parts
            old_teacher_local, _, _, _ = query_conditioned_reconstruction(
                old_parts, old_patches,
                self._prototype_bank(cls, patches, labels)[2][old_classes],
                self.model.reconstructor, self.epsilon, self.sinkhorn_steps,
                self.logit_scale,
            ) if False else (None, None, None, None)

        for epoch in range(epochs):
            self.model.train()
            totals = {key: 0.0 for key in (
                "total", "local", "joint", "previous_consistency",
                "base_consistency", "distillation", "anchor", "overlap",
            )}
            steps = 0
            for indices in self._balanced_memory_indices(
                labels, classes_per_batch, samples_per_class,
                epoch + session * epochs
            ):
                q_cls, q_patches, q_labels = cls[indices], patches[indices], labels[indices]
                output = self.model.decompose_features(q_cls, q_patches)
                all_global = self._global_features(cls)
                all_classes, all_global_proto, all_local_proto, all_parts = self._prototype_bank(
                    cls, patches, labels
                )
                candidate_parts = loo_class_prototypes(
                    all_parts, labels, indices, all_classes,
                    self.epsilon, self.sinkhorn_steps, self.barycenter_steps,
                )
                global_scores_full = global_logits(
                    output["global_features"], all_global_proto, self.logit_scale
                )
                local_scores, _, _, _ = query_conditioned_reconstruction(
                    output["parts"], q_patches, candidate_parts,
                    self.model.reconstructor, self.epsilon,
                    self.sinkhorn_steps, self.logit_scale,
                )
                joint_scores = self.model.fusion(global_scores_full, local_scores)
                target_map = {int(c): i for i, c in enumerate(all_classes.tolist())}
                targets = torch.tensor(
                    [target_map[int(v)] for v in q_labels.tolist()],
                    dtype=torch.long, device=self.device
                )
                local_loss = F.cross_entropy(local_scores, targets)
                joint_loss = F.cross_entropy(joint_scores, targets)

                old_batch = torch.isin(q_labels, old_classes)
                prev_consistency = prompt_consistency_loss(
                    output["parts"][old_batch],
                    previous_parts[indices][old_batch],
                    epsilon=self.epsilon, iterations=self.sinkhorn_steps,
                )
                base_batch = old_batch & (q_labels < int(self.args.base_class))
                base_consistency = prompt_consistency_loss(
                    output["parts"][base_batch],
                    torch.stack([
                        self.base_reference[self.memory.records[int(i)].sample_id].to(self.device)
                        for i in indices[base_batch].cpu().tolist()
                    ]) if bool(base_batch.any()) else output["parts"][base_batch].detach(),
                    epsilon=self.epsilon, iterations=self.sinkhorn_steps,
                )
                anchor_loss = (prompt - previous_prompt).square().mean()

                # Distill old-class predictions with a frozen previous Prompt.
                old_query = torch.nonzero(old_batch, as_tuple=False).flatten()
                if old_query.numel() > 0 and old_classes.numel() > 0:
                    with torch.no_grad():
                        teacher_output = self.model.decompose_features(
                            q_cls[old_query], q_patches[old_query]
                        )
                        teacher_all_parts = previous_parts
                        teacher_prototypes = []
                        for old_class in old_classes.tolist():
                            mask = labels == int(old_class)
                            teacher_prototypes.append(ot_barycenter(
                                teacher_all_parts[mask], self.epsilon,
                                self.sinkhorn_steps, self.barycenter_steps
                            ))
                        teacher_prototypes = torch.stack(teacher_prototypes)
                        teacher_local, _, _, _ = query_conditioned_reconstruction(
                            teacher_output["parts"], q_patches[old_query],
                            teacher_prototypes, self.model.reconstructor,
                            self.epsilon, self.sinkhorn_steps, self.logit_scale,
                        )
                        teacher_global = global_logits(
                            teacher_output["global_features"],
                            self.global_prototypes[:old_classes.numel()],
                            self.logit_scale,
                        )
                        teacher_joint = self.model.fusion(teacher_global, teacher_local)
                    student_joint_old = joint_scores[old_query, :old_classes.numel()]
                    teacher_joint_old = teacher_joint
                    kd_loss = distillation_kl(
                        student_joint_old, teacher_joint_old, temperature
                    )
                else:
                    kd_loss = joint_loss * 0.0

                overlap_loss = attention_overlap_loss(output["attention"])
                loss = (
                    w_local * local_loss + w_joint * joint_loss
                    + w_prev * prev_consistency + w_base * base_consistency
                    + w_kd * kd_loss + w_anchor * anchor_loss
                    + w_overlap * overlap_loss
                )
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
                for key, value in {
                    "total": loss, "local": local_loss, "joint": joint_loss,
                    "previous_consistency": prev_consistency,
                    "base_consistency": base_consistency, "distillation": kd_loss,
                    "anchor": anchor_loss, "overlap": overlap_loss,
                }.items():
                    totals[key] += float(value.detach())
                steps += 1

            record = {"stage": "incremental_train", "session": session, "epoch": epoch + 1}
            record.update({key: value / max(steps, 1) for key, value in totals.items()})
            record["prompt_delta_from_previous"] = float(
                (prompt.detach() - previous_prompt).norm()
            )
            self._append_log(record)

        # Permanent Base references are immutable. Newly admitted class references
        # are established only after this Session has finished training.
        with torch.no_grad():
            _, _, _, final_parts = self._prototype_bank(cls, patches, labels)
            for index, memory_record in enumerate(self.memory.records):
                if memory_record.label >= int(self.args.base_class) + (session - 1) * int(self.args.way):
                    if memory_record.label < int(self.args.base_class) + session * int(self.args.way):
                        self.previous_reference[memory_record.sample_id] = final_parts[index].cpu().clone()
            self.classes, self.global_prototypes, self.local_prototypes = (
                self._prototype_bank(cls, patches, labels)[:3]
            )
        self._append_log({
            "stage": "memory_update", "session": session,
            "memory_samples": len(self.memory),
            "classes": int(self.classes.numel()),
            "prompt_delta_from_previous": float((prompt.detach() - previous_prompt).norm()),
        })

    @torch.no_grad()
    def _evaluate(self, testset, session):
        original = getattr(testset, "transform", None)
        if original is None:
            raise RuntimeError("Test dataset must expose transform.")
        testset.transform = self._eval_transform()
        loader = self._loader(testset, int(self.args.test_batch_size))
        predictions, labels_all = [], []
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
                logits = self.model.fusion(global_scores, local_scores)
                predictions.append(logits.argmax(dim=1).cpu())
                labels_all.append(torch.as_tensor(labels, dtype=torch.long).cpu())
        finally:
            testset.transform = original
        predictions = torch.cat(predictions)
        labels = torch.cat(labels_all)
        metrics = evaluate_session_predictions(
            labels, predictions, int(self.args.base_class),
            int(self.args.base_class + session * int(self.args.way) if session else self.args.base_class),
            session, int(self.args.way),
        )
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
