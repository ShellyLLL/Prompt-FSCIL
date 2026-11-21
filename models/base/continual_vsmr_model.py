"""Continual VSMR-Prompt model: frozen ViT and fixed shared visual Prompt."""
import torch
import torch.nn as nn

from .vsmr_modules import (
    FrozenViTBackbone,
    VisualSemanticDecomposer,
    QueryConditionedReconstructor,
    GlobalLocalFusion,
)


class ContinualVSMRViT(nn.Module):
    """Frozen ViT with a shared Prompt as the sole incremental trainable block."""

    def __init__(self, args, encoder=None):
        super().__init__()
        if bool(getattr(args, "scratch", False)):
            raise ValueError(
                "Continual VSMR requires the pretrained ImageNet-21K ViT-B/16."
            )
        self.backbone = FrozenViTBackbone(pretrained=True, encoder=encoder)
        dim = self.backbone.embed_dim
        self.decomposer = VisualSemanticDecomposer(
            embed_dim=dim,
            num_parts=int(getattr(args, "vsmr_num_parts", 6)),
            temperature=float(getattr(args, "vsmr_prompt_temperature", 1.0)),
        )
        self.reconstructor = QueryConditionedReconstructor(embed_dim=dim)
        self.fusion = GlobalLocalFusion(
            global_init=float(getattr(args, "vsmr_global_weight", 0.5)),
            local_init=float(getattr(args, "vsmr_local_weight_eval", 0.5)),
        )
        self._incremental_prompt_only = False
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if self._incremental_prompt_only:
            self.decomposer.eval()
            self.reconstructor.eval()
            self.fusion.eval()
        return self

    def decompose_features(self, cls_features, patch_features):
        global_features, parts, attention = self.decomposer(
            cls_features, patch_features
        )
        return {
            "cls_features": cls_features,
            "patch_features": patch_features,
            "global_features": global_features,
            "parts": parts,
            "attention": attention,
        }

    def forward(self, images):
        cls_features, patch_features = self.backbone(images)
        return self.decompose_features(cls_features, patch_features)

    def freeze_for_incremental(self):
        """Freeze all learned modules except the shared Prompt matrix."""
        self._incremental_prompt_only = True
        for parameter in self.parameters():
            # Base optimizer gradients may still be attached after its final
            # step. Freeze must clear them so incremental checks observe only
            # gradients created by the Prompt-only objective.
            parameter.grad = None
            parameter.requires_grad_(False)
        self.decomposer.shared_prompts.requires_grad_(True)
        self.eval()
        trainable = [
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        if trainable != ["decomposer.shared_prompts"]:
            raise RuntimeError(
                "Incremental trainable-parameter contract violated: {}".format(
                    trainable
                )
            )

    def assert_incremental_parameter_contract(self):
        trainable = [
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        if trainable != ["decomposer.shared_prompts"]:
            raise RuntimeError(
                "Only decomposer.shared_prompts may be trainable; got {}".format(
                    trainable
                )
            )
