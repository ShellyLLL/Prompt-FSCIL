"""Training-free diagnostics for Continual VSMR-Prompt.

All statistics are descriptive diagnostics. They do not add losses or alter
model parameters, so enabling them cannot change the optimization objective.
"""
import math

import torch
import torch.nn.functional as F

from .vsmr_losses import ot_distance


@torch.no_grad()
def component_set_diagnostics(parts, attention, labels, epsilon=0.05,
                              iterations=30, max_samples=48,
                              max_pairs=512):
    """Measure component diversity, attention use, and class-conditional OT.

    Args:
        parts: [N,K,D] decomposed visual components.
        attention: [N,K,P] prompt-to-patch distributions.
        labels: [N] labels from the training/replay memory only.
    """
    if parts.ndim != 3 or attention.ndim != 3 or labels.ndim != 1:
        raise ValueError("Expected parts [N,K,D], attention [N,K,P], labels [N].")
    if parts.shape[0] != attention.shape[0] or parts.shape[0] != labels.numel():
        raise ValueError("Diagnostic tensors must share their sample axis.")
    if parts.shape[1] < 2 or attention.shape[1] != parts.shape[1]:
        raise ValueError("At least two aligned components are required.")
    if attention.shape[-1] < 1 or parts.shape[0] < 1:
        raise ValueError("Diagnostic inputs cannot be empty.")
    if not torch.isfinite(parts).all() or not torch.isfinite(attention).all():
        raise ValueError("Diagnostic inputs contain non-finite values.")

    n = parts.shape[0]
    if n > int(max_samples):
        indices = torch.linspace(
            0, n - 1, steps=int(max_samples), device=parts.device
        ).round().long().unique()
        parts, attention, labels = parts[indices], attention[indices], labels[indices]
    labels = labels.to(device=parts.device, dtype=torch.long)
    parts = F.normalize(parts, dim=-1)
    attention = attention.clamp_min(0.0)
    attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    k = parts.shape[1]
    off_diagonal = ~torch.eye(k, dtype=torch.bool, device=parts.device)
    cosine = torch.matmul(parts, parts.transpose(1, 2))
    component_cosine = cosine[:, off_diagonal].mean()

    patch_count = attention.shape[-1]
    entropy = -(
        attention.clamp_min(1e-12) * attention.clamp_min(1e-12).log()
    ).sum(dim=-1)
    normalized_entropy = entropy / max(math.log(max(patch_count, 2)), 1e-12)

    overlap_matrix = torch.minimum(
        attention.unsqueeze(2), attention.unsqueeze(1)
    ).sum(dim=-1)
    attention_overlap = overlap_matrix[:, off_diagonal].mean()

    top_count = max(1, int(math.ceil(0.10 * patch_count)))
    top_indices = attention.topk(top_count, dim=-1).indices
    covered = torch.zeros(
        attention.shape[0], patch_count, dtype=torch.bool, device=parts.device
    )
    covered.scatter_(1, top_indices.reshape(attention.shape[0], -1), True)
    patch_coverage = covered.float().mean()

    same_distances, different_distances = [], []
    pair_count = 0
    sample_count = parts.shape[0]
    for left in range(sample_count):
        for right in range(left + 1, sample_count):
            distance = ot_distance(
                parts[left:left + 1], parts[right:right + 1],
                epsilon=epsilon, iterations=iterations,
            )[0]
            target = same_distances if labels[left] == labels[right] else different_distances
            target.append(distance)
            pair_count += 1
            if pair_count >= int(max_pairs):
                break
        if pair_count >= int(max_pairs):
            break

    same_mean = (
        float(torch.stack(same_distances).mean().item())
        if same_distances else None
    )
    different_mean = (
        float(torch.stack(different_distances).mean().item())
        if different_distances else None
    )
    separation = (
        different_mean - same_mean
        if same_mean is not None and different_mean is not None else None
    )
    return {
        "diagnostic_samples": int(sample_count),
        "component_count": int(k),
        "component_pairwise_cosine_mean": float(component_cosine.item()),
        "attention_entropy_normalized_mean": float(normalized_entropy.mean().item()),
        "attention_overlap_mean": float(attention_overlap.item()),
        "patch_coverage_top10_union_mean": float(patch_coverage.item()),
        "within_class_component_ot_distance_mean": same_mean,
        "between_class_component_ot_distance_mean": different_mean,
        "between_minus_within_component_ot_distance": separation,
        "component_ot_pairs": int(pair_count),
    }


def prompt_gradient_diagnostics(classification_loss, history_loss, prompt,
                                history_weight, epsilon=1e-12):
    """Measure the relative size and directional agreement of two Prompt gradients."""
    classification_grad = torch.autograd.grad(
        classification_loss, prompt, retain_graph=True, allow_unused=True
    )[0]
    history_grad = torch.autograd.grad(
        history_loss, prompt, retain_graph=True, allow_unused=True
    )[0]
    if classification_grad is None:
        classification_grad = torch.zeros_like(prompt)
    if history_grad is None:
        history_grad = torch.zeros_like(prompt)

    classification_norm = classification_grad.norm()
    history_norm = history_grad.norm()
    weighted_history_norm = abs(float(history_weight)) * history_norm
    ratio = weighted_history_norm / classification_norm.clamp_min(epsilon)
    cosine = F.cosine_similarity(
        classification_grad.reshape(1, -1),
        history_grad.reshape(1, -1),
        dim=-1,
        eps=epsilon,
    )[0]
    values = {
        "classification_prompt_grad_norm": classification_norm.detach(),
        "history_prompt_grad_norm_unweighted": history_norm.detach(),
        "history_prompt_grad_norm_weighted": weighted_history_norm.detach(),
        "weighted_history_to_classification_grad_ratio": ratio.detach(),
        "history_classification_grad_cosine": cosine.detach(),
    }
    return {key: float(value.item()) for key, value in values.items()}
