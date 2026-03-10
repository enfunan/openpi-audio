"""LoRA module for PyTorch.

Provides LoRA with optional audio position routing and gradient isolation.
Audio LoRA is applied at audio token positions; task LoRA everywhere else.
Gradient modes control which LoRA branch receives gradients during training.
"""

import dataclasses
import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)

# Gradient mode constants.
GRAD_MODE_NONE = 0
GRAD_MODE_FLOW_MATCHING = 1  # Stop gradient on audio LoRA
GRAD_MODE_ASR = 2  # Stop gradient on task LoRA
GRAD_MODE_BYPASS = 3  # Skip LoRA entirely, use base weights only (teacher)


@dataclasses.dataclass
class LoRAConfig:
    rank: int = 16
    alpha: float = 16.0
    init_std: float = 0.01

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear with optional dual (task + audio) LoRA.

    Forward signature accepts optional ``audio_mask`` and ``gradient_mode``
    kwargs so it is a drop-in replacement after injection.  When neither
    audio LoRA nor audio_mask is provided it falls back to task-only LoRA.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        task_cfg: LoRAConfig | None = None,
        audio_cfg: LoRAConfig | None = None,
    ):
        super().__init__()
        self.base_linear = base_linear
        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.task_cfg = task_cfg
        self.audio_cfg = audio_cfg

        if task_cfg is not None:
            # Keep LoRA params in float32 for numerical stability
            self.task_A = nn.Linear(in_features, task_cfg.rank, bias=False, dtype=torch.float32)
            self.task_B = nn.Linear(task_cfg.rank, out_features, bias=False, dtype=torch.float32)
            nn.init.normal_(self.task_A.weight, std=task_cfg.init_std)
            nn.init.zeros_(self.task_B.weight)  # Standard LoRA: B=0 → zero initial contribution
            self.task_scaling = task_cfg.scaling

        if audio_cfg is not None:
            self.audio_A = nn.Linear(in_features, audio_cfg.rank, bias=False, dtype=torch.float32)
            self.audio_B = nn.Linear(audio_cfg.rank, out_features, bias=False, dtype=torch.float32)
            nn.init.normal_(self.audio_A.weight, std=audio_cfg.init_std)
            nn.init.zeros_(self.audio_B.weight)
            self.audio_scaling = audio_cfg.scaling

    # Proxy properties so callers that inspect the old nn.Linear still work.
    @property
    def weight(self):
        return self.base_linear.weight

    @property
    def bias(self):
        return self.base_linear.bias

    @property
    def in_features(self):
        return self.base_linear.in_features

    @property
    def out_features(self):
        return self.base_linear.out_features

    def forward(self, x, audio_mask=None, gradient_mode=None):
        # Base projection (frozen weights)
        base_out = self.base_linear(x)

        if self.task_cfg is None or gradient_mode == GRAD_MODE_BYPASS:
            return base_out

        # Compute LoRA outputs in float32 for numerical stability
        x_f32 = x.float()

        # Task LoRA
        task_out = self.task_B(self.task_A(x_f32)) * self.task_scaling

        if self.audio_cfg is not None and audio_mask is not None:
            # Split LoRA mode: separate audio path
            audio_out = self.audio_B(self.audio_A(x_f32)) * self.audio_scaling

            # Gradient isolation
            if gradient_mode is not None:
                if gradient_mode == GRAD_MODE_FLOW_MATCHING:
                    audio_out = audio_out.detach()
                elif gradient_mode == GRAD_MODE_ASR:
                    task_out = task_out.detach()

            # Position routing: audio_mask is (B, S) bool → (B, S, 1)
            mask = audio_mask.unsqueeze(-1)
            lora_out = torch.where(mask, audio_out, task_out)
        else:
            lora_out = task_out

        return base_out + lora_out.to(base_out.dtype)


def inject_lora(
    paligemma_language_model,
    action_expert_model=None,
    task_cfg: LoRAConfig = None,
    audio_cfg: LoRAConfig = None,
    paligemma_targets: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ),
    expert_targets: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ),
):
    """Replace nn.Linear projections in-place with LoRALinear wrappers.

    PaliGemma gets split LoRA (task + audio) on ``paligemma_targets``.
    Action expert (if provided) gets task-only LoRA on ``expert_targets``.
    """
    count = 0

    # PaliGemma language model — split LoRA (task + audio)
    for layer in paligemma_language_model.layers:
        for name in paligemma_targets:
            if name in ("gate_proj", "up_proj", "down_proj"):
                parent = layer.mlp
            else:
                parent = layer.self_attn
            base_linear = getattr(parent, name)
            lora_linear = LoRALinear(base_linear, task_cfg=task_cfg, audio_cfg=audio_cfg)
            setattr(parent, name, lora_linear)
            count += 1

    # Action expert — task-only LoRA (no audio)
    if action_expert_model is not None:
        for layer in action_expert_model.layers:
            for name in expert_targets:
                if name in ("gate_proj", "up_proj", "down_proj"):
                    parent = layer.mlp
                else:
                    parent = layer.self_attn
                base_linear = getattr(parent, name)
                lora_linear = LoRALinear(base_linear, task_cfg=task_cfg, audio_cfg=None)
                setattr(parent, name, lora_linear)
                count += 1

    logger.info(f"Injected LoRA into {count} linear projections")
    return count
