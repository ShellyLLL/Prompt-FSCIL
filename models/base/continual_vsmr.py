"""Efficient CPU feature memory and prototype utilities for Continual VSMR."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class FeatureRecord:
    cls: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    label: int
    sample_id: str
    session: int


class FrozenFeatureMemory:
    """CPU FP16 cache of frozen ViT CLS and projected Patch K/V features."""

    def __init__(self, max_base_per_class=5):
        if int(max_base_per_class) < 1:
            raise ValueError("max_base_per_class must be positive.")
        self.max_base_per_class = int(max_base_per_class)
        self.records = []
        self._ids = set()

    def append(self, cls_features, keys, values, labels, sample_ids, session,
               is_base=False):
        if cls_features.ndim != 2 or keys.ndim != 3 or values.shape != keys.shape:
            raise ValueError("Expected CLS [B,D] and matching projected K/V [B,N,D].")
        n = int(cls_features.shape[0])
        if keys.shape[0] != n or labels.ndim != 1 or labels.numel() != n or len(sample_ids) != n:
            raise ValueError("Feature, label and sample-id counts differ.")
        if cls_features.shape[1] != keys.shape[2]:
            raise ValueError("CLS and projected Patch dimensions differ.")
        if not all(torch.isfinite(t).all() for t in (cls_features, keys, values)):
            raise ValueError("Feature memory cannot store non-finite features.")
        counts = {}
        for record in self.records:
            counts[record.label] = counts.get(record.label, 0) + 1
        for i, raw_label in enumerate(labels.detach().cpu().tolist()):
            label, sid = int(raw_label), str(sample_ids[i])
            if sid in self._ids:
                raise RuntimeError("Duplicate sample id: " + sid)
            if is_base and counts.get(label, 0) >= self.max_base_per_class:
                continue
            self.records.append(FeatureRecord(
                cls=cls_features[i].detach().cpu().half().contiguous(),
                keys=keys[i].detach().cpu().half().contiguous(),
                values=values[i].detach().cpu().half().contiguous(),
                label=label, sample_id=sid, session=int(session)))
            self._ids.add(sid)
            counts[label] = counts.get(label, 0) + 1

    def tensors(self, device):
        if not self.records:
            raise RuntimeError("Feature memory is empty.")
        cls = torch.stack([r.cls for r in self.records]).to(device=device, dtype=torch.float32)
        keys = torch.stack([r.keys for r in self.records]).to(device=device, dtype=torch.float16)
        values = torch.stack([r.values for r in self.records]).to(device=device, dtype=torch.float16)
        labels = torch.tensor([r.label for r in self.records], device=device, dtype=torch.long)
        return cls, keys, values, labels

    def metadata(self):
        return [{"label": r.label, "sample_id": r.sample_id, "session": r.session}
                for r in self.records]

    def state_dict(self):
        return [{
            "cls": r.cls.clone(), "keys": r.keys.clone(), "values": r.values.clone(),
            "label": int(r.label), "sample_id": str(r.sample_id), "session": int(r.session),
        } for r in self.records]

    def __len__(self):
        return len(self.records)


def normalized_class_means(features, labels, classes=None):
    """Differentiable normalized mean prototypes in sorted class order."""
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.numel():
        raise ValueError("Expected features [N,D] and labels [N].")
    classes = torch.unique(labels, sorted=True) if classes is None else torch.as_tensor(
        classes, dtype=torch.long, device=labels.device
    )
    prototypes = []
    for class_id in classes.tolist():
        selected = features[labels == int(class_id)]
        if selected.shape[0] == 0:
            raise ValueError("Class {} has no feature samples.".format(class_id))
        prototypes.append(F.normalize(selected.mean(0), dim=-1))
    return classes, torch.stack(prototypes)


def leave_one_out_cosine_logits(features, labels, query_indices, classes, scale=16.0):
    """Vectorized cosine logits with each query removed from its own class mean."""
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.numel():
        raise ValueError("Expected features [N,D] and labels [N].")
    query_indices = torch.as_tensor(
        query_indices, dtype=torch.long, device=labels.device
    ).reshape(-1)
    classes = torch.as_tensor(classes, dtype=torch.long, device=labels.device).reshape(-1)
    if query_indices.numel() == 0 or torch.any(query_indices < 0) or torch.any(query_indices >= labels.numel()):
        raise ValueError("Query indices must be non-empty and in range.")
    if torch.unique(query_indices).numel() != query_indices.numel():
        raise ValueError("Query indices must be unique.")
    observed = torch.unique(labels, sorted=True)
    if not torch.equal(classes, observed):
        raise ValueError("Classes must exactly match labels.")
    inverse = torch.searchsorted(classes, labels)
    counts = torch.bincount(inverse, minlength=classes.numel())
    if torch.any(counts < 2):
        raise ValueError("LOO classification requires at least two samples per class.")

    # One class-wise reduction replaces per-query/per-class Python loops.
    sums = features.new_zeros((classes.numel(), features.shape[-1])).index_add(
        0, inverse, features
    )
    prototypes = F.normalize(sums / counts.to(features.dtype).unsqueeze(1), dim=-1)
    query_features = F.normalize(features[query_indices], dim=-1)
    scores = (query_features @ prototypes.t()) * float(scale)

    query_classes = inverse[query_indices]
    own_sums = sums[query_classes] - features[query_indices]
    own_counts = (counts[query_classes] - 1).to(features.dtype).unsqueeze(1)
    own_prototypes = F.normalize(own_sums / own_counts, dim=-1)
    own_scores = (query_features * own_prototypes).sum(-1) * float(scale)
    return scores.scatter(1, query_classes.unsqueeze(1), own_scores.unsqueeze(1))


def cosine_consistency_loss(student, reference):
    """Mean sample-wise cosine drift; reference is always detached."""
    if student.shape != reference.shape or student.ndim < 2:
        raise ValueError("Student and reference tensors must have equal batched shapes.")
    if student.shape[0] == 0:
        return student.sum() * 0.0
    student = F.normalize(student.reshape(student.shape[0], -1), dim=-1)
    reference = F.normalize(reference.detach().reshape(reference.shape[0], -1), dim=-1)
    return (1.0 - (student * reference).sum(-1)).mean()
