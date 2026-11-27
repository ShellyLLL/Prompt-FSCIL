"""End-to-end efficient Continual VSMR-Prompt FSCIL trainer.

The frozen ViT is run once per training image. Base CLS/Patch features are
cached on CPU; incremental memory stores FP16 CLS and projected Patch K/V.
Incremental optimization updates only one shared Prompt and never reruns ViT.
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
    normalized_class_means,
    leave_one_out_cosine_logits,
    cosine_consistency_loss,
)
from .continual_vsmr_model import ContinualVSMRViT
from .continual_vsmr_metrics import (
    evaluate_session_predictions,
    summarize_session_metrics,
    write_session_csv,
    write_confusion_csv,
    format_session_table,
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
    """Deterministic class-balanced batches without replacement within a batch."""

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
            raise ValueError("Balanced LOO batches require >=2 classes and samples/class.")
        if len(self.mapping) < self.classes_per_batch:
            raise ValueError("Not enough classes for a balanced batch.")
        if any(len(v) < self.samples_per_class for v in self.mapping.values()):
            raise ValueError("A class has too few samples for a balanced batch.")
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
            chosen = rng.choice(classes, self.classes_per_batch, replace=False)
            batch = []
            for cls in chosen:
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
        self.logit_scale = float(getattr(args, "vsmr_logit_scale", 16.0))
        self.memory_per_class = int(getattr(args, "continual_base_memory_per_class", 5))
        if self.memory_per_class != 5:
            raise ValueError("VSMR V2.0 uses a fixed memory budget of exactly 5 samples/class.")
        self.memory = FrozenFeatureMemory(self.memory_per_class)
        self.sample_birth = {}
        self.prototype_birth = {}
        self.global_prototypes = None
        self.local_prototypes = None
        self.classes = None
        self.session_records = []
        self.confusion_matrices = {}
        self.timings = {}
        self.test_cache = None
        root = getattr(args, "out", None) or "ContinualVSMR"
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        root = root if os.path.isabs(root) else os.path.join(repo_root, "checkpoint", root)
        self.save_dir = os.path.join(
            root, self.args.dataset, "ContinualVSMR_V2",
            "{}_seed{}".format(time.strftime("%Y%m%d_%H%M%S"), self.seed)
        )
        os.makedirs(self.save_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.save_dir, "metrics.jsonl")
        active_keys = {
            "method", "project", "dataset", "dataroot", "index_list_root",
            "base_class", "way", "shot", "sessions", "seed", "gpu",
            "epochs_base", "epochs_new", "num_workers", "test_batch_size",
            "out", "vit", "vsmr_prompt_temperature", "vsmr_logit_scale",
            "vsmr_lr", "vsmr_weight_decay",
            "continual_base_memory_per_class", "continual_prompt_lr",
            "continual_prompt_weight_decay", "continual_history_rep_weight",
            "continual_history_proto_weight", "continual_global_init",
            "continual_global_loss_weight", "continual_local_loss_weight",
            "continual_joint_loss_weight", "continual_classes_per_batch",
            "continual_samples_per_class", "continual_feature_batch_size",
            "continual_eval_batch_size",
        }
        self._write_json(
            os.path.join(self.save_dir, "config.json"),
            {k: v for k, v in vars(self.args).items() if k in active_keys},
        )
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

    def _loader(self, dataset, batch_size):
        workers = int(getattr(self.args, "num_workers", 4))
        return DataLoader(
            IndexedDataset(dataset), batch_size=int(batch_size), shuffle=False,
            num_workers=workers, pin_memory=self.device.type == "cuda",
            persistent_workers=workers > 0,
        )

    @torch.no_grad()
    def _extract_dataset(self, dataset, batch_size=None, store_projected=True):
        original = getattr(dataset, "transform", None)
        if original is None:
            raise RuntimeError("Dataset must expose a transform.")
        dataset.transform = self._eval_transform()
        loader = self._loader(
            dataset, batch_size or int(getattr(self.args, "continual_feature_batch_size", 128))
        )
        started = time.time()
        size = len(dataset)
        cls_cache = labels_cache = key_cache = value_cache = patch_cache = None
        ids_all = [None] * size
        self.model.eval()
        try:
            for images, labels, indices in tqdm(loader, desc="Frozen ViT feature extraction"):
                images = images.to(self.device, non_blocking=True)
                cls, patches = self.model.backbone(images)
                indices_cpu = indices.long().cpu()
                if cls_cache is None:
                    dim = int(cls.shape[-1])
                    cls_cache = torch.empty((size, dim), dtype=torch.float16)
                    labels_cache = torch.empty((size,), dtype=torch.long)
                    if store_projected:
                        projected_shape = (size, int(patches.shape[1]), int(patches.shape[2]))
                        key_cache = torch.empty(projected_shape, dtype=torch.float16)
                        value_cache = torch.empty_like(key_cache)
                    else:
                        patch_cache = torch.empty(
                            (size,) + tuple(patches.shape[1:]), dtype=torch.float16
                        )
                cls_cache[indices_cpu] = cls.detach().cpu().half()
                labels_cache[indices_cpu] = torch.as_tensor(labels, dtype=torch.long).cpu()
                for row, index in enumerate(indices_cpu.tolist()):
                    ids_all[index] = int(index)
                if store_projected:
                    keys, values = self.model.context.project_patches(patches)
                    key_cache[indices_cpu] = keys.detach().cpu().half()
                    value_cache[indices_cpu] = values.detach().cpu().half()
                else:
                    patch_cache[indices_cpu] = patches.detach().cpu().half()
        finally:
            dataset.transform = original
        if cls_cache is None or labels_cache is None:
            raise RuntimeError("Cannot extract features from an empty dataset.")
        self.timings["feature_extraction_seconds"] = (
            self.timings.get("feature_extraction_seconds", 0.0) + time.time() - started
        )
        result = {"cls": cls_cache, "labels": labels_cache, "ids": ids_all}
        if store_projected:
            result["keys"], result["values"] = key_cache, value_cache
        else:
            result["patches"] = patch_cache
        return result

    def _all_memory(self):
        return self.memory.tensors(self.device)

    def _build_prototypes(self, global_features, local_features, labels):
        classes, global_proto = normalized_class_means(global_features, labels)
        local_classes, local_proto = normalized_class_means(local_features, labels, classes)
        if not torch.equal(classes, local_classes):
            raise RuntimeError("Global/local prototype class order differs.")
        return classes, global_proto, local_proto

    def _train_base(self, trainset):
        """Extract frozen ViT once, then train the lightweight Prompt on cached features."""
        started = time.time()
        trainset.transform = self._eval_transform()
        cache = self._extract_dataset(trainset, store_projected=False)
        labels = cache["labels"].numpy()
        expected = np.arange(int(self.args.base_class))
        if not np.array_equal(np.unique(labels), expected):
            raise RuntimeError("Base training labels must be contiguous [0, base_class).")
        sampler = BalancedBatchSampler(
            labels,
            min(int(getattr(self.args, "continual_classes_per_batch", 16)), int(self.args.base_class)),
            int(getattr(self.args, "continual_samples_per_class", 4)),
            self.seed,
        )
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=float(getattr(self.args, "vsmr_lr", 2e-4)),
            weight_decay=float(getattr(self.args, "vsmr_weight_decay", 1e-4)),
        )
        epochs = int(self.args.epochs_base)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        wg = float(getattr(self.args, "continual_global_loss_weight", 1.0))
        wl = float(getattr(self.args, "continual_local_loss_weight", 1.0))
        wj = float(getattr(self.args, "continual_joint_loss_weight", 1.0))
        batch_size = int(getattr(self.args, "continual_feature_batch_size", 128))
        for epoch in range(epochs):
            epoch_started = time.perf_counter()
            sampler.set_epoch(epoch)
            self.model.train()
            totals = {"total": 0.0, "global": 0.0, "local": 0.0, "joint": 0.0}
            steps = 0
            for raw_indices in sampler:
                indices = torch.as_tensor(raw_indices, dtype=torch.long)
                cls = cache["cls"][indices].to(self.device, dtype=torch.float32, non_blocking=True)
                patches = cache["patches"][indices].to(self.device, dtype=torch.float32, non_blocking=True)
                labels_batch = cache["labels"][indices].to(self.device, non_blocking=True)
                output = self.model.forward_features(cls, patches)
                batch_classes = torch.unique(labels_batch, sorted=True)
                query_indices = torch.arange(labels_batch.numel(), device=self.device)
                global_scores = leave_one_out_cosine_logits(
                    output["global_features"], labels_batch, query_indices,
                    batch_classes, self.logit_scale,
                )
                local_scores = leave_one_out_cosine_logits(
                    output["local_features"], labels_batch, query_indices,
                    batch_classes, self.logit_scale,
                )
                joint_scores = self.model.fusion(global_scores, local_scores)
                targets = torch.searchsorted(batch_classes, labels_batch)
                lg = F.cross_entropy(global_scores, targets)
                ll = F.cross_entropy(local_scores, targets)
                lj = F.cross_entropy(joint_scores, targets)
                loss = wg * lg + wl * ll + wj * lj
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite Base loss.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for name, parameter in self.model.named_parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        if not torch.isfinite(parameter.grad).all():
                            raise FloatingPointError("Non-finite gradient: " + name)
                optimizer.step()
                totals["total"] += float(loss.detach())
                totals["global"] += float(lg.detach())
                totals["local"] += float(ll.detach())
                totals["joint"] += float(lj.detach())
                steps += 1
            scheduler.step()
            record = {"stage": "base_train", "epoch": epoch + 1, "steps": steps}
            record.update({k: v / max(steps, 1) for k, v in totals.items()})
            record["fusion_alpha"] = float(self.model.fusion.alpha.detach())
            record["lr"] = scheduler.get_last_lr()[0]
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            record["epoch_seconds"] = time.perf_counter() - epoch_started
            if self.device.type == "cuda":
                record["cuda_peak_allocated_mb"] = (
                    torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 2)
                )
            self._append_log(record)

        self.model.freeze_for_incremental()
        # Select exactly five deterministic training samples per base class.
        selected = []
        rng = np.random.default_rng(self.seed)
        for class_id in range(int(self.args.base_class)):
            candidates = np.flatnonzero(labels == class_id)
            if candidates.size < self.memory_per_class:
                raise RuntimeError("Base class {} has too few samples.".format(class_id))
            selected.extend(np.sort(rng.choice(
                candidates, self.memory_per_class, replace=False
            )).tolist())
        selected = torch.tensor(sorted(selected), dtype=torch.long)
        memory_cls = cache["cls"][selected].float().to(self.device)
        memory_patches = cache["patches"][selected].float().to(self.device)
        with torch.no_grad():
            keys, values = self.model.context.project_patches(memory_patches)
            selected_labels = cache["labels"][selected]
            sample_ids = ["base:index{}".format(int(i)) for i in selected.tolist()]
            self.memory.append(
                memory_cls, keys, values, selected_labels, sample_ids, 0, is_base=True
            )
            cls, all_keys, all_values, memory_labels = self._all_memory()
            local_features = []
            global_features = []
            for start in range(0, len(self.memory), batch_size):
                end = min(start + batch_size, len(self.memory))
                output = self.model.forward_projected(
                    cls[start:end], all_keys[start:end], all_values[start:end]
                )
                local_features.append(output["local_features"])
                global_features.append(output["global_features"])
            global_features = torch.cat(global_features)
            local_features = torch.cat(local_features)
            self.classes, self.global_prototypes, self.local_prototypes = self._build_prototypes(
                global_features, local_features, memory_labels
            )
            for index, record in enumerate(self.memory.records):
                self.sample_birth[record.sample_id] = local_features[index].detach().cpu().clone()
            for row, class_id in enumerate(self.classes.tolist()):
                self.prototype_birth[int(class_id)] = self.local_prototypes[row].detach().cpu().clone()
        self._save_checkpoint("continual_vsmr_base.pth", stage=0, include_memory=True)
        memory_bytes = sum(
            r.cls.numel() * r.cls.element_size()
            + r.keys.numel() * r.keys.element_size()
            + r.values.numel() * r.values.element_size()
            for r in self.memory.records
        )
        self._append_log({
            "stage": "base_memory", "memory_samples": len(self.memory),
            "memory_per_class": self.memory_per_class, "memory_bytes": int(memory_bytes),
            "base_train_seconds": time.time() - started,
        })

    def _incremental_session(self, support_set, session):
        started = time.time()
        start_class = int(self.args.base_class) + (session - 1) * int(self.args.way)
        expected = np.arange(start_class, start_class + int(self.args.way))
        support_labels = np.asarray(support_set.targets, dtype=np.int64)
        counts = {int(c): int((support_labels == c).sum()) for c in expected}
        if not np.array_equal(np.unique(support_labels), expected):
            raise RuntimeError("Session {} support classes mismatch: {}".format(session, counts))
        if any(v != int(self.args.shot) for v in counts.values()):
            raise RuntimeError("Session {} requires exact K-shot: {}".format(session, counts))
        previous_count = len(self.memory)
        new = self._extract_dataset(support_set, store_projected=True)
        self.memory.append(
            new["cls"], new["keys"], new["values"], new["labels"],
            ["session{}:index{}".format(session, i) for i in new["ids"]],
            session, is_base=False,
        )
        expected_added = int(self.args.way) * int(self.args.shot)
        if len(self.memory) - previous_count != expected_added:
            raise RuntimeError("Incremental memory did not append exactly K-shot samples.")

        cls, keys, values, labels = self._all_memory()
        prompt = self.model.context.shared_prompt
        self.model.assert_incremental_parameter_contract()
        historical_indices = torch.arange(previous_count, device=self.device)
        new_indices = torch.arange(previous_count, len(self.memory), device=self.device)
        historical_ids = [self.memory.records[i].sample_id for i in range(previous_count)]
        missing = [sid for sid in historical_ids if sid not in self.sample_birth]
        if missing:
            raise RuntimeError("Missing immutable sample Birth Reference: " + missing[0])
        sample_reference = torch.stack([
            self.sample_birth[sid].to(self.device) for sid in historical_ids
        ])
        old_classes = torch.unique(labels[:previous_count], sorted=True)
        prototype_reference = torch.stack([
            self.prototype_birth[int(c)].to(self.device) for c in old_classes.tolist()
        ])
        # The global branch is frozen after Base. Its features and LOO logits
        # are invariant throughout this Session, so compute them once rather
        # than repeating the full-memory projection on every Prompt epoch.
        classes = torch.unique(labels, sorted=True)
        query_indices = new_indices
        targets = torch.searchsorted(classes, labels[query_indices])
        with torch.no_grad():
            global_features = self.model.context.global_features(cls)
            global_scores = leave_one_out_cosine_logits(
                global_features, labels, query_indices, classes, self.logit_scale
            )
        optimizer = torch.optim.AdamW(
            [prompt],
            lr=float(getattr(self.args, "continual_prompt_lr", 1e-4)),
            weight_decay=float(getattr(self.args, "continual_prompt_weight_decay", 1e-4)),
        )
        epochs = int(self.args.epochs_new)
        wr = float(getattr(self.args, "continual_history_rep_weight", 0.1))
        wp = float(getattr(self.args, "continual_history_proto_weight", 0.1))
        feature_batch = int(getattr(self.args, "continual_feature_batch_size", 128))
        previous_prompt = prompt.detach().clone()

        for epoch in range(epochs):
            epoch_started = time.perf_counter()
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            # Compute the current local representation for every stored sample
            # once. Frozen K/V are already cached; only Prompt attention is rerun.
            local_chunks = []
            for start in range(0, len(self.memory), feature_batch):
                end = min(start + feature_batch, len(self.memory))
                local, _ = self.model.context.aggregate_projected(
                    keys[start:end], values[start:end]
                )
                local_chunks.append(local)
            local_features = torch.cat(local_chunks, dim=0)
            # New support samples are queries; every seen class is a candidate.
            # Only local scores depend on the changing Prompt. Global LOO scores
            # are the precomputed, immutable Base-space scores above.
            local_scores = leave_one_out_cosine_logits(
                local_features, labels, query_indices, classes, self.logit_scale
            )
            joint_scores = self.model.fusion(global_scores, local_scores)
            classification_loss = F.cross_entropy(joint_scores, targets)
            rep_loss = cosine_consistency_loss(
                local_features[historical_indices], sample_reference
            )
            _, current_old_prototypes = normalized_class_means(
                local_features[historical_indices], labels[historical_indices], old_classes
            )
            proto_loss = cosine_consistency_loss(
                current_old_prototypes, prototype_reference
            )
            loss = classification_loss + wr * rep_loss + wp * proto_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite incremental objective.")
            loss.backward()
            if prompt.grad is None or not torch.isfinite(prompt.grad).all():
                raise FloatingPointError("Shared Prompt gradient is absent or non-finite.")
            for name, parameter in self.model.named_parameters():
                if name != "context.shared_prompt" and parameter.grad is not None:
                    raise RuntimeError("Frozen parameter received gradient: " + name)
            optimizer.step()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self._append_log({
                "stage": "incremental_train", "session": session, "epoch": epoch + 1,
                "epoch_seconds": time.perf_counter() - epoch_started,
                "loss": float(loss.detach()),
                "classification": float(classification_loss.detach()),
                "history_representation": float(rep_loss.detach()),
                "history_prototype": float(proto_loss.detach()),
                "prompt_delta_from_previous": float((prompt.detach() - previous_prompt).norm()),
            })

        # Recalibrate all seen-class local prototypes in the final Prompt space.
        with torch.no_grad():
            local_chunks = []
            for start in range(0, len(self.memory), feature_batch):
                end = min(start + feature_batch, len(self.memory))
                local, _ = self.model.context.aggregate_projected(
                    keys[start:end], values[start:end]
                )
                local_chunks.append(local)
            local_features = torch.cat(local_chunks)
            self.classes, self.local_prototypes = normalized_class_means(
                local_features, labels
            )
            # The global branch is frozen and its historical prototypes are
            # immutable; append only prototypes for classes introduced now.
            new_global_features = self.model.context.global_features(cls[new_indices])
            _, new_global_prototypes = normalized_class_means(
                new_global_features, labels[new_indices]
            )
            self.global_prototypes = torch.cat(
                [self.global_prototypes, new_global_prototypes], dim=0
            )
            if self.global_prototypes.shape[0] != self.classes.numel():
                raise RuntimeError("Global prototype bank does not cover all seen classes.")
            for index in range(previous_count, len(self.memory)):
                sid = self.memory.records[index].sample_id
                self.sample_birth[sid] = local_features[index].cpu().clone()
            for row, class_id in enumerate(self.classes.tolist()):
                if int(class_id) not in self.prototype_birth:
                    self.prototype_birth[int(class_id)] = self.local_prototypes[row].cpu().clone()
        self._save_checkpoint("continual_vsmr_S{}.pth".format(session), stage=session, include_memory=False)
        self.timings["incremental_training_seconds"] = (
            self.timings.get("incremental_training_seconds", 0.0) + time.time() - started
        )
        self._append_log({
            "stage": "memory_update", "session": session,
            "memory_samples": len(self.memory), "seen_classes": int(self.classes.numel()),
            "historical_samples": previous_count,
            "new_samples": len(self.memory) - previous_count,
            "prompt_delta": float((prompt.detach() - previous_prompt).norm()),
            "incremental_session_seconds": time.time() - started,
        })

    def _prepare_test_cache(self, testset):
        """Encode the complete held-out test split once; never use it for updates."""
        started = time.time()
        self.test_cache = self._extract_dataset(testset, store_projected=True)
        self.timings["test_feature_cache_seconds"] = time.time() - started
        cache_bytes = sum(
            value.numel() * value.element_size()
            for value in self.test_cache.values() if torch.is_tensor(value)
        )
        self._append_log({
            "stage": "test_feature_cache",
            "test_samples": int(self.test_cache["labels"].numel()),
            "cache_bytes": int(cache_bytes),
            "cache_gib": float(cache_bytes / (1024.0 ** 3)),
            "seconds": self.timings["test_feature_cache_seconds"],
        })

    @torch.no_grad()
    def _evaluate_cached(self, session):
        cache = self.test_cache
        seen = int(self.args.base_class) + session * int(self.args.way)
        selected = torch.nonzero(cache["labels"] < seen, as_tuple=False).flatten()
        if selected.numel() == 0:
            raise RuntimeError("No cached test samples for Session {}.".format(session))
        predictions, labels_all = [], []
        batch_size = int(getattr(self.args, "continual_eval_batch_size", 128))
        self.model.eval()
        for start in range(0, selected.numel(), batch_size):
            idx = selected[start:start + batch_size]
            cls = cache["cls"][idx].to(self.device, dtype=torch.float32, non_blocking=True)
            keys = cache["keys"][idx].to(self.device, dtype=torch.float16, non_blocking=True)
            values = cache["values"][idx].to(self.device, dtype=torch.float16, non_blocking=True)
            output = self.model.forward_projected(cls, keys, values)
            global_scores = F.normalize(output["global_features"], dim=-1) @ self.global_prototypes.t()
            local_scores = F.normalize(output["local_features"], dim=-1) @ self.local_prototypes.t()
            scores = self.model.fusion(
                global_scores * self.logit_scale, local_scores * self.logit_scale
            )
            predictions.append(scores.argmax(-1).cpu())
            labels_all.append(cache["labels"][idx].cpu())
        labels = torch.cat(labels_all)
        predictions = torch.cat(predictions)
        metrics = evaluate_session_predictions(
            labels, predictions, int(self.args.base_class), seen, session, int(self.args.way)
        )
        metrics["stage"] = "evaluation"
        metrics["method"] = "ContinualVSMR-Prompt-V2"
        self.confusion_matrices[session] = metrics.pop("confusion_matrix")
        self.session_records.append(metrics)
        self._append_log(metrics)
        write_confusion_csv(
            os.path.join(self.save_dir, "confusion_S{}.csv".format(session)),
            self.confusion_matrices[session],
        )
        return metrics

    @torch.no_grad()
    def _evaluate(self, testset, session):
        original = getattr(testset, "transform", None)
        testset.transform = self._eval_transform()
        loader = self._loader(testset, int(getattr(self.args, "continual_eval_batch_size", 128)))
        predictions, labels_all = [], []
        self.model.eval()
        try:
            for images, labels, _ in tqdm(loader, desc="Evaluate S{}".format(session)):
                cls, patches = self.model.backbone(images.to(self.device, non_blocking=True))
                output = self.model.forward_features(cls, patches)
                global_scores = F.normalize(output["global_features"], dim=-1) @ self.global_prototypes.t()
                local_scores = F.normalize(output["local_features"], dim=-1) @ self.local_prototypes.t()
                scores = self.model.fusion(
                    global_scores * self.logit_scale, local_scores * self.logit_scale
                )
                predictions.append(scores.argmax(-1).cpu())
                labels_all.append(torch.as_tensor(labels, dtype=torch.long).cpu())
        finally:
            testset.transform = original
        labels = torch.cat(labels_all)
        predictions = torch.cat(predictions)
        seen = int(self.args.base_class) + session * int(self.args.way)
        metrics = evaluate_session_predictions(
            labels, predictions, int(self.args.base_class), seen, session, int(self.args.way)
        )
        metrics["stage"] = "evaluation"
        metrics["method"] = "ContinualVSMR-Prompt-V2"
        self.confusion_matrices[session] = metrics.pop("confusion_matrix")
        self.session_records.append(metrics)
        self._append_log(metrics)
        write_confusion_csv(
            os.path.join(self.save_dir, "confusion_S{}.csv".format(session)),
            self.confusion_matrices[session],
        )
        return metrics

    def _save_checkpoint(self, filename, stage, include_memory=False):
        state = {
            "stage": int(stage), "model": self.model.state_dict(),
            "global_prototypes": self.global_prototypes.detach().cpu(),
            "local_prototypes": self.local_prototypes.detach().cpu(),
            "sample_birth": self.sample_birth,
            "prototype_birth": self.prototype_birth,
            "session_records": self.session_records,
            "timings": self.timings,
            "memory_included": bool(include_memory),
        }
        # Avoid serializing hundreds of MB of unchanged replay K/V after every
        # session. The Base and final checkpoints remain self-contained.
        if include_memory:
            state["memory_metadata"] = self.memory.metadata()
            state["feature_memory"] = self.memory.state_dict()
        torch.save(state, os.path.join(self.save_dir, filename))

    def _write_final_report(self, elapsed):
        if len(self.session_records) != int(self.args.sessions):
            raise RuntimeError("Refusing to report incomplete S0..ST metrics.")
        summary = summarize_session_metrics(self.session_records)
        base_drop = (
            float(self.session_records[0]["base_class_average_accuracy"])
            - float(self.session_records[-1]["base_class_average_accuracy"])
        )
        summary.update({
            "stage": "summary", "method": "ContinualVSMR-Prompt-V2",
            "elapsed_minutes": float(elapsed / 60.0),
            "base_class_accuracy_drop": base_drop,
            "base_class_accuracy_drop_percentage_points": 100.0 * base_drop,
            "memory_per_class": self.memory_per_class,
            "memory_samples_final": len(self.memory),
            "timings": self.timings,
            "AA_definition": "Arithmetic mean of all-seen-class macro Top-1 accuracy over S0..ST.",
            "PD_definition": "S0 all-seen-class macro accuracy minus ST all-seen-class macro accuracy.",
            "base_class_drop_definition": "S0 base-class macro accuracy minus ST base-class macro accuracy.",
        })
        self._append_log(summary)
        write_session_csv(os.path.join(self.save_dir, "session_metrics.csv"), self.session_records)
        self._write_json(os.path.join(self.save_dir, "session_metrics.json"), self.session_records)
        self._write_json(os.path.join(self.save_dir, "summary.json"), summary)
        with open(os.path.join(self.save_dir, "summary.txt"), "w") as handle:
            handle.write(format_session_table(self.session_records) + "\n")
        self._save_checkpoint("continual_vsmr_final.pth", stage=int(self.args.sessions) - 1, include_memory=True)
        print("\n" + format_session_table(self.session_records))
        return summary

    def train(self):
        started = time.time()
        trainset, _, testloader = get_base_dataloader(self.args)
        self._train_base(trainset)
        # Cache the complete held-out test split once. All later session
        # evaluations reuse frozen CLS/K/V features; only Prompt aggregation
        # and prototype scoring are recomputed.
        _, _, final_testloader = get_new_dataloader(self.args, int(self.args.sessions) - 1)
        self._prepare_test_cache(final_testloader.dataset)
        self._evaluate_cached(0)
        for session in range(1, int(self.args.sessions)):
            support_set, _, _ = get_new_dataloader(self.args, session)
            self._incremental_session(support_set, session)
            self._evaluate_cached(session)
        summary = self._write_final_report(time.time() - started)
        print("[ContinualVSMR] summary={}".format(os.path.join(self.save_dir, "summary.txt")))
        return summary
