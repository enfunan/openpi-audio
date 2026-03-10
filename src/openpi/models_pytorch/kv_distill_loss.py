"""KV cache distillation loss for audio-conditioned model training.

Computes alignment loss between teacher (text) and student (audio) KV caches
from PaliGemma prefix-only forward passes.

Loss has two components:
  1. Image region (per-position MSE): positions 0..num_image_tokens-1
     Both teacher and student see identical images, but KVs differ because
     self-attention mixes in text vs audio information.

  2. Text/audio region (pooled MSE): positions num_image_tokens..end
     Teacher has ~10-40 valid text tokens; student has 32 audio tokens.
     Cannot align per-position, so pool each side and compare.
"""

import torch
from torch import Tensor
import torch.nn.functional as F  # noqa: N812


def kv_cache_to_tensors(kv_cache, num_layers: int = 18):
    """Convert HuggingFace DynamicCache to stacked (keys, values) tensors.

    Args:
        kv_cache: DynamicCache with ``cache[layer_idx]`` → (K, V),
            each of shape ``(B, num_kv_heads, seq_len, head_dim)``.
        num_layers: Number of transformer layers.

    Returns:
        keys:   (num_layers, B, num_kv_heads, seq_len, head_dim)
        values: (num_layers, B, num_kv_heads, seq_len, head_dim)
    """
    keys = torch.stack([kv_cache[i][0] for i in range(num_layers)])
    values = torch.stack([kv_cache[i][1] for i in range(num_layers)])
    return keys, values


def _masked_mean_pool(x: Tensor, mask: Tensor) -> Tensor:
    """Pool over sequence dimension using a validity mask.

    Args:
        x:    (L, B, S, D)  — L=layers, B=batch, S=seq, D=head_dim
        mask: (B, S)        — True for valid positions

    Returns:
        (L, B, D)
    """
    mask_expanded = mask[None, :, :, None].float()  # (1, B, S, 1)
    pooled = (x * mask_expanded).sum(dim=2) / mask_expanded.sum(dim=2).clamp(min=1.0)
    return pooled


def compute_kv_distill_loss(
    teacher_kv,
    student_kv,
    teacher_valid_mask: Tensor,
    student_valid_mask: Tensor,
    num_image_tokens: int = 768,
    alpha: float = 1.0,
    num_layers: int = 18,
) -> tuple[Tensor, dict[str, float]]:
    """Compute KV cache alignment loss between teacher and student.

    Args:
        teacher_kv: DynamicCache from teacher (text) prefix forward.
        student_kv: DynamicCache from student (audio) prefix forward.
        teacher_valid_mask: (B, seq_len) bool — True where teacher tokens valid.
        student_valid_mask: (B, seq_len) bool — True where student tokens valid.
        num_image_tokens: Number of image positions (e.g. 3×256 = 768).
        alpha: Weight for semantic (text/audio region) loss.
        num_layers: Number of PaliGemma transformer layers.

    Returns:
        total_loss: scalar tensor
        metrics: dict with per-component loss values
    """
    teacher_keys, teacher_values = kv_cache_to_tensors(teacher_kv, num_layers)
    student_keys, student_values = kv_cache_to_tensors(student_kv, num_layers)
    # Each: (L, B, num_kv_heads, S, head_dim) — num_kv_heads=1 for Gemma GQA

    # Squeeze kv_heads dim → (L, B, S, D)
    teacher_keys = teacher_keys.squeeze(2).float()
    teacher_values = teacher_values.squeeze(2).float()
    student_keys = student_keys.squeeze(2).float()
    student_values = student_values.squeeze(2).float()

    head_dim = teacher_keys.shape[-1]

    # ===== Image region: per-position MSE =====
    t_img_k = teacher_keys[:, :, :num_image_tokens, :]
    s_img_k = student_keys[:, :, :num_image_tokens, :]
    t_img_v = teacher_values[:, :, :num_image_tokens, :]
    s_img_v = student_values[:, :, :num_image_tokens, :]

    # Mask: valid in BOTH teacher and student
    img_valid = (
        teacher_valid_mask[:, :num_image_tokens] & student_valid_mask[:, :num_image_tokens]
    )  # (B, 768)
    img_valid_expanded = img_valid[None, :, :, None].float()  # (1, B, 768, 1)
    num_valid_img = img_valid_expanded.sum().clamp(min=1.0)

    img_k_loss = ((t_img_k - s_img_k) ** 2 * img_valid_expanded).sum() / num_valid_img / head_dim
    img_v_loss = ((t_img_v - s_img_v) ** 2 * img_valid_expanded).sum() / num_valid_img / head_dim
    image_loss = (img_k_loss + img_v_loss) / 2

    # ===== Text/audio region: pooled MSE =====
    t_text_k = teacher_keys[:, :, num_image_tokens:, :]
    s_text_k = student_keys[:, :, num_image_tokens:, :]
    t_text_v = teacher_values[:, :, num_image_tokens:, :]
    s_text_v = student_values[:, :, num_image_tokens:, :]

    teacher_text_valid = teacher_valid_mask[:, num_image_tokens:]
    student_text_valid = student_valid_mask[:, num_image_tokens:]

    # Pool each side → (L, B, D), then MSE
    t_k_pooled = _masked_mean_pool(t_text_k, teacher_text_valid)
    s_k_pooled = _masked_mean_pool(s_text_k, student_text_valid)
    t_v_pooled = _masked_mean_pool(t_text_v, teacher_text_valid)
    s_v_pooled = _masked_mean_pool(s_text_v, student_text_valid)

    semantic_k_loss = F.mse_loss(s_k_pooled, t_k_pooled.detach())
    semantic_v_loss = F.mse_loss(s_v_pooled, t_v_pooled.detach())
    semantic_loss = (semantic_k_loss + semantic_v_loss) / 2

    total_loss = image_loss + alpha * semantic_loss

    metrics = {
        "kv/image_k_loss": img_k_loss.item(),
        "kv/image_v_loss": img_v_loss.item(),
        "kv/image_loss": image_loss.item(),
        "kv/semantic_k_loss": semantic_k_loss.item(),
        "kv/semantic_v_loss": semantic_v_loss.item(),
        "kv/semantic_loss": semantic_loss.item(),
        "kv/total_loss": total_loss.item(),
    }
    return total_loss, metrics
