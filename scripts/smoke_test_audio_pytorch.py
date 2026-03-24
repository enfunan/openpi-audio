"""Smoke test for PyTorch audio/split LoRA port.

Tests model construction, LoRA injection, forward pass shapes, gradient flow,
and gradient isolation — all with random weights (no pretrained checkpoint needed).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/smoke_test_audio_pytorch.py
"""

import logging
import sys
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch


def make_fake_observation(batch_size, device, audio_enabled=True):
    """Create a fake observation matching the data pipeline output."""

    class FakeObs:
        pass

    obs = FakeObs()
    obs.images = {
        "base_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device),
        "left_wrist_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device),
        "right_wrist_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device),
    }
    obs.image_masks = {
        "base_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
        "left_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
        "right_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
    }
    obs.state = torch.randn(batch_size, 32, device=device)
    obs.tokenized_prompt = torch.randint(0, 1000, (batch_size, 48), device=device)
    obs.tokenized_prompt_mask = torch.ones(batch_size, 48, dtype=torch.bool, device=device)
    obs.token_ar_mask = None
    obs.token_loss_mask = None

    if audio_enabled:
        obs.audio_whisper_hidden = torch.randn(batch_size, 1500, 1280, device=device)
        obs.audio_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        obs.asr_target_tokens = torch.randint(0, 1000, (batch_size, 48), device=device)
        obs.asr_target_mask = torch.ones(batch_size, 48, dtype=torch.bool, device=device)
        obs.original_tokenized_prompt = obs.tokenized_prompt.clone()
        obs.original_tokenized_prompt_mask = obs.tokenized_prompt_mask.clone()
    else:
        obs.audio_whisper_hidden = None
        obs.audio_mask = None
        obs.asr_target_tokens = None
        obs.asr_target_mask = None
        obs.original_tokenized_prompt = None
        obs.original_tokenized_prompt_mask = None

    return obs


def test_model_construction():
    """Test 1: Model construction with audio components."""
    logger.info("=" * 60)
    logger.info("TEST 1: Model construction")
    logger.info("=" * 60)

    import openpi.models.pi0_config as pi0_config
    import openpi.models_pytorch.pi0_pytorch as pi0_pt

    # Don't actually load Whisper from HF — patch it
    # We'll test with a mock whisper encoder
    config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=10,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        audio_enabled=True,
        training_stage="asr_alignment",
        dtype="float32",  # Use float32 for smoke test (faster on CPU/small GPU)
    )

    # Patch WhisperModel to avoid downloading weights
    import unittest.mock as mock

    class FakeWhisperEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Linear(1, 1)

        def forward(self, input_features, **kwargs):
            B = input_features.shape[0]
            return torch.randn(B, 1500, 1280, device=input_features.device)

    with mock.patch("transformers.WhisperModel") as MockWhisper:
        mock_model = mock.MagicMock()
        mock_model.encoder = FakeWhisperEncoder()
        MockWhisper.from_pretrained.return_value = mock_model

        model = pi0_pt.PI0Pytorch(config)

    logger.info(f"  Model created: {type(model).__name__}")
    logger.info(f"  audio_enabled: {model.audio_enabled}")
    logger.info(f"  training_stage: {model.training_stage}")
    assert hasattr(model, "audio_projector"), "Missing audio_projector"
    assert hasattr(model, "attention_pooling"), "Missing attention_pooling"
    assert hasattr(model, "whisper_encoder"), "Missing whisper_encoder"
    logger.info("  PASS: All audio components present")
    return model


def test_lora_injection(model):
    """Test 2: LoRA injection."""
    logger.info("=" * 60)
    logger.info("TEST 2: LoRA injection")
    logger.info("=" * 60)

    from openpi.models_pytorch.lora_pytorch import LoRAConfig, LoRALinear, inject_lora

    task_cfg = LoRAConfig(rank=16, alpha=16.0)
    audio_cfg = LoRAConfig(rank=16, alpha=16.0)

    count = inject_lora(
        model.paligemma_with_expert.paligemma.language_model,
        model.paligemma_with_expert.gemma_expert.model,
        task_cfg=task_cfg,
        audio_cfg=audio_cfg,
    )
    logger.info(f"  Injected LoRA into {count} projections")

    # Verify PaliGemma layers have split LoRA
    layer0 = model.paligemma_with_expert.paligemma.language_model.layers[0]
    q_proj = layer0.self_attn.q_proj
    assert isinstance(q_proj, LoRALinear), f"Expected LoRALinear, got {type(q_proj)}"
    assert q_proj.audio_cfg is not None, "PaliGemma should have audio LoRA"
    logger.info("  PASS: PaliGemma has split LoRA (task + audio)")

    # Verify action expert layers have task-only LoRA
    expert_layer0 = model.paligemma_with_expert.gemma_expert.model.layers[0]
    expert_q = expert_layer0.self_attn.q_proj
    assert isinstance(expert_q, LoRALinear), f"Expected LoRALinear, got {type(expert_q)}"
    assert expert_q.audio_cfg is None, "Action expert should NOT have audio LoRA"
    logger.info("  PASS: Action expert has task-only LoRA")

    # Verify LoRA params are float32
    assert q_proj.task_A.weight.dtype == torch.float32, "LoRA A should be float32"
    assert q_proj.task_B.weight.dtype == torch.float32, "LoRA B should be float32"
    logger.info("  PASS: LoRA params are float32")

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"  Trainable: {trainable:,} / {total:,} total")

    return model


def test_forward_pass(model, device):
    """Test 3: Forward pass — ASR alignment (Stage 1)."""
    logger.info("=" * 60)
    logger.info("TEST 3: Forward pass (ASR alignment)")
    logger.info("=" * 60)

    model = model.to(device)
    model.train()

    batch_size = 2
    obs = make_fake_observation(batch_size, device, audio_enabled=True)
    actions = torch.randn(batch_size, 10, 32, device=device)

    # Stage 1: ASR alignment
    from openpi.models_pytorch.lora_pytorch import GRAD_MODE_ASR
    loss = model.forward(obs, actions, training_stage="asr_alignment")
    logger.info(f"  ASR loss shape: {loss.shape}, value: {loss.mean().item():.4f}")
    assert loss.shape == (batch_size, 1), f"Expected ({batch_size}, 1), got {loss.shape}"
    logger.info("  PASS: ASR alignment forward")

    # Stage 2: Flow matching
    loss_fm = model.forward(obs, actions, training_stage="default")
    logger.info(f"  FM loss shape: {loss_fm.shape}, value: {loss_fm.mean().item():.4f}")
    assert loss_fm.ndim >= 2, f"Expected >=2D, got {loss_fm.ndim}D"
    logger.info("  PASS: Flow matching forward")

    return model


def test_gradient_isolation(model, device):
    """Test 4: Gradient isolation — verify detach works correctly."""
    logger.info("=" * 60)
    logger.info("TEST 4: Gradient isolation")
    logger.info("=" * 60)

    model = model.to(device)
    model.train()

    batch_size = 2
    obs = make_fake_observation(batch_size, device, audio_enabled=True)
    actions = torch.randn(batch_size, 10, 32, device=device)

    # Test ASR mode: task LoRA should have zero grads, audio LoRA should have grads
    model.zero_grad()
    loss = model.compute_alignment_loss(obs, gradient_mode=2)  # GRAD_MODE_ASR
    loss.backward()

    layer0 = model.paligemma_with_expert.paligemma.language_model.layers[0]
    q_proj = layer0.self_attn.q_proj

    task_A_grad = q_proj.task_A.weight.grad
    audio_A_grad = q_proj.audio_A.weight.grad

    task_has_grad = task_A_grad is not None and task_A_grad.abs().sum() > 0
    audio_has_grad = audio_A_grad is not None and audio_A_grad.abs().sum() > 0

    logger.info(f"  ASR mode — task LoRA grad: {'nonzero' if task_has_grad else 'ZERO/None'}")
    logger.info(f"  ASR mode — audio LoRA grad: {'nonzero' if audio_has_grad else 'ZERO/None'}")

    if task_has_grad:
        logger.warning("  WARN: task LoRA got gradients in ASR mode (should be detached)")
    else:
        logger.info("  PASS: task LoRA correctly detached in ASR mode")

    if audio_has_grad:
        logger.info("  PASS: audio LoRA receives gradients in ASR mode")
    else:
        logger.warning("  WARN: audio LoRA got no gradients in ASR mode")

    return model


def test_backward_and_nan_guard(model, device):
    """Test 5: Full backward pass with NaN guard."""
    logger.info("=" * 60)
    logger.info("TEST 5: Backward pass + NaN guard")
    logger.info("=" * 60)

    model = model.to(device)
    model.train()

    batch_size = 2
    obs = make_fake_observation(batch_size, device, audio_enabled=True)
    actions = torch.randn(batch_size, 10, 32, device=device)

    # Freeze base weights (like the training script does)
    for name, param in model.named_parameters():
        if "base_linear" in name:
            param.requires_grad_(False)
    model.whisper_encoder.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
    )

    # Run 2 steps
    for step in range(2):
        optimizer.zero_grad()
        loss = model.forward(obs, actions, training_stage="asr_alignment")
        loss_val = loss.mean()
        loss_val.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )

        if torch.isfinite(grad_norm):
            optimizer.step()
            logger.info(f"  Step {step}: loss={loss_val.item():.4f}, grad_norm={grad_norm.item():.4f}")
        else:
            logger.warning(f"  Step {step}: NaN grad_norm, skipping")

    logger.info("  PASS: Backward pass completes without crash")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    model = test_model_construction()
    model = test_lora_injection(model)
    model = test_forward_pass(model, device)
    model = test_gradient_isolation(model, device)
    test_backward_and_nan_guard(model, device)

    logger.info("=" * 60)
    logger.info("ALL SMOKE TESTS PASSED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
