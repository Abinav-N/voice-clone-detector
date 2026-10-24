"""
Unit tests for VoiceCloneCNN model architecture and forward pass shapes (Task 22).
"""
import torch
import pytest

from model import VoiceCloneCNN


def test_model_instantiation():
    model = VoiceCloneCNN()
    assert isinstance(model, torch.nn.Module)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params > 100_000
    assert total_params == trainable_params


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_forward_pass_shape(batch_size):
    model = VoiceCloneCNN()
    model.eval()
    dummy_input = torch.randn(batch_size, 3, 40, 300)
    with torch.no_grad():
        out = model(dummy_input)
    assert out.shape == (batch_size, 2)
    assert out.dtype == torch.float32

    # Verify softmax output sums to 1.0 per sample
    probs = torch.softmax(out, dim=1)
    sums = probs.sum(dim=1).numpy()
    for s in sums:
        assert pytest.approx(s, rel=1e-5) == 1.0


def test_gradcam_backward():
    """Verify that gradients propagate to the last convolutional layer for Grad-CAM."""
    model = VoiceCloneCNN()
    model.eval()

    target_layer = None
    for module in reversed(list(model.features.modules())):
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
            break
    assert target_layer is not None, "Could not find Conv2d layer in features"

    gradients = []
    def hook_grad(grad):
        gradients.append(grad)

    x = torch.randn(1, 3, 40, 300, requires_grad=True)
    # Forward hook
    acts = []
    def hook_act(module, inp, out):
        acts.append(out)
        out.register_hook(hook_grad)

    h = target_layer.register_forward_hook(hook_act)
    logits = model(x)
    loss = logits[0, 1]  # fake class logit
    loss.backward()
    h.remove()

    assert len(acts) == 1
    assert len(gradients) == 1
    assert acts[0].shape == gradients[0].shape
    assert acts[0].ndim == 4
