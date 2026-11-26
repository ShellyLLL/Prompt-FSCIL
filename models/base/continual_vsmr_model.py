"""Continual VSMR-Prompt: frozen ViT with one shared visual context Prompt."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vsmr_modules import FrozenViTBackbone


class PromptGuidedVisualContext(nn.Module):
    """One shared Prompt pools frozen Patch tokens into a contextual feature.

    Q/K/V and the global projection are learned in Base and frozen thereafter.
    Only shared_prompt is updated during incremental sessions.
    """

    def __init__(self, embed_dim=768, temperature=1.0):
        super().__init__()
        if int(embed_dim) <= 0 or float(temperature) <= 0:
            raise ValueError("embed_dim and temperature must be positive.")
        self.embed_dim = int(embed_dim)
        self.temperature = float(temperature)
        self.shared_prompt = nn.Parameter(torch.empty(1, self.embed_dim))
        nn.init.trunc_normal_(self.shared_prompt, std=0.02)
        self.query_projection = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.key_projection = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.value_projection = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.global_projection = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def project_patches(self, patch_features):
        if patch_features.ndim != 3 or patch_features.shape[-1] != self.embed_dim:
            raise ValueError("Expected Patch features [B,N,D] with configured D.")
        return self.key_projection(patch_features), self.value_projection(patch_features)

    def aggregate_projected(self, keys, values):
        if keys.ndim != 3 or values.shape != keys.shape:
            raise ValueError("Projected keys and values must share [B,N,D].")
        if keys.shape[-1] != self.embed_dim or keys.shape[1] < 1:
            raise ValueError("Projected Patch feature dimensions are invalid.")
        query = self.query_projection(self.shared_prompt).squeeze(0)
        # Half precision projected memory reduces memory traffic; softmax stays FP32.
        query_for_attention = query.to(dtype=keys.dtype)
        logits = torch.einsum("d,bnd->bn", query_for_attention, keys)
        logits = logits.float() / (math.sqrt(self.embed_dim) * self.temperature)
        attention = torch.softmax(logits, dim=-1)
        context = torch.einsum(
            "bn,bnd->bd", attention.to(dtype=values.dtype), values
        ).float()
        return F.normalize(context, p=2, dim=-1), attention

    def forward(self, cls_features, patch_features):
        if cls_features.ndim != 2 or cls_features.shape[-1] != self.embed_dim:
            raise ValueError("Expected CLS features [B,D].")
        if patch_features.ndim != 3 or patch_features.shape[0] != cls_features.shape[0]:
            raise ValueError("CLS and Patch batch sizes differ or Patch shape is invalid.")
        keys, values = self.project_patches(patch_features)
        local_features, attention = self.aggregate_projected(keys, values)
        global_features = self.global_features(cls_features)
        return global_features, local_features, attention

    def global_features(self, cls_features):
        return F.normalize(self.global_projection(cls_features), p=2, dim=-1)


class GlobalLocalFusion(nn.Module):
    """Learned convex fusion, trained in Base and frozen incrementally."""

    def __init__(self, global_init=0.5):
        super().__init__()
        value = float(global_init)
        if not 0.0 < value < 1.0:
            raise ValueError("global_init must be strictly between 0 and 1.")
        self.logit_alpha = nn.Parameter(torch.tensor(math.log(value / (1.0 - value))))

    @property
    def alpha(self):
        return torch.sigmoid(self.logit_alpha)

    def forward(self, global_scores, local_scores):
        if global_scores.shape != local_scores.shape:
            raise ValueError("Global and local score tensors must have equal shapes.")
        alpha = self.alpha
        return alpha * global_scores + (1.0 - alpha) * local_scores


class ContinualVSMRViT(nn.Module):
    """Frozen ViT-B/16 plus Prompt-guided visual context and prototype fusion."""

    def __init__(self, args, encoder=None):
        super().__init__()
        if bool(getattr(args, "scratch", False)):
            raise ValueError("Continual VSMR requires pretrained ImageNet-21K ViT-B/16.")
        self.backbone = FrozenViTBackbone(pretrained=True, encoder=encoder)
        self.context = PromptGuidedVisualContext(
            embed_dim=self.backbone.embed_dim,
            temperature=float(getattr(args, "vsmr_prompt_temperature", 1.0)),
        )
        self.fusion = GlobalLocalFusion(
            global_init=float(getattr(args, "continual_global_init",
                                      getattr(args, "vsmr_global_weight", 0.5)))
        )
        self._incremental_prompt_only = False
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if self._incremental_prompt_only:
            self.context.eval()
            self.fusion.eval()
        return self

    def forward_features(self, cls_features, patch_features):
        global_features, local_features, attention = self.context(
            cls_features, patch_features
        )
        return {
            "global_features": global_features,
            "local_features": local_features,
            "attention": attention,
        }

    def forward_projected(self, cls_features, keys, values):
        global_features = self.context.global_features(cls_features)
        local_features, attention = self.context.aggregate_projected(keys, values)
        return {
            "global_features": global_features,
            "local_features": local_features,
            "attention": attention,
        }

    def forward(self, images):
        cls_features, patch_features = self.backbone(images)
        output = self.forward_features(cls_features, patch_features)
        output["cls_features"] = cls_features
        output["patch_features"] = patch_features
        return output

    def freeze_for_incremental(self):
        """Freeze every parameter except the single shared visual Prompt."""
        self._incremental_prompt_only = True
        for parameter in self.parameters():
            parameter.grad = None
            parameter.requires_grad_(False)
        self.context.shared_prompt.requires_grad_(True)
        self.eval()
        self.assert_incremental_parameter_contract()

    def assert_incremental_parameter_contract(self):
        trainable = [
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        if trainable != ["context.shared_prompt"]:
            raise RuntimeError(
                "Only context.shared_prompt may be trainable; got {}.".format(trainable)
            )
